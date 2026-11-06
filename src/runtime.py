"""PyTorch-free serving runtime: image loading, ONNX Runtime models, visual helpers.

Everything the web server imports lives here (plus config, preprocessing and
ensemble), so a deployment needs only numpy, Pillow and onnxruntime - no
PyTorch, SciPy or scikit-learn. That keeps the Vercel function at ~215 MB.
"""
from __future__ import annotations

import base64
import importlib
import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps

from .config import CLASS_NAMES, MODEL_SPECS, env_number
from .preprocessing import PreprocessConfig, preprocess_image, to_rgb

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ImageInput = Union[str, Path, Image.Image, np.ndarray]


# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #
def normalize_image(img: Image.Image) -> Image.Image:
    """Upright 8-bit RGB copy: first frame, EXIF orientation applied, any mode."""
    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)
    img = ImageOps.exif_transpose(img)  # phone photos are often stored rotated
    return to_rgb(img)


def load_image(source: ImageInput) -> Image.Image:
    if isinstance(source, (str, Path)):
        with Image.open(source) as img:
            return normalize_image(img)
    if isinstance(source, np.ndarray):
        return to_rgb(Image.fromarray(source))
    if isinstance(source, Image.Image):
        return normalize_image(source)
    raise ValueError(f"Unsupported image type: {type(source)}")


def to_input_array(image: Image.Image, cfg: PreprocessConfig) -> np.ndarray:
    """(1, 3, H, W) float32 - identical to torchvision ToTensor() + Normalize(ImageNet)."""
    prepared = preprocess_image(image, cfg)
    arr = np.asarray(prepared, dtype=np.float32) / np.float32(255.0)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------------------- #
# Grad-CAM visuals
# --------------------------------------------------------------------------- #
_COLORMAP_ANCHORS = np.array(
    [[0.00, 0, 0, 128], [0.25, 0, 96, 255], [0.50, 0, 220, 200], [0.75, 255, 210, 0], [1.00, 220, 20, 20]],
    dtype=np.float32,
)


def normalize_cam(cam: np.ndarray, size: int) -> np.ndarray:
    """Scale a raw CAM to [0, 1] and upsample (bilinear) to size x size."""
    cam = np.maximum(np.asarray(cam, dtype=np.float32), 0.0)
    peak = float(cam.max()) if cam.size else 0.0
    if not np.isfinite(peak) or peak <= 1e-8:
        return np.zeros((size, size), dtype=np.float32)
    up = Image.fromarray((cam / np.float32(peak)).astype(np.float32)).resize((size, size), Image.BILINEAR)
    return np.clip(np.asarray(up, dtype=np.float32), 0.0, 1.0)


def colorize(heatmap: np.ndarray) -> np.ndarray:
    """Map [0,1] -> RGB uint8 with a perceptually ordered blue->cyan->yellow->red ramp."""
    flat = heatmap.ravel()
    channels = [np.interp(flat, _COLORMAP_ANCHORS[:, 0], _COLORMAP_ANCHORS[:, i]) for i in (1, 2, 3)]
    return np.stack(channels, axis=-1).reshape(*heatmap.shape, 3).astype(np.uint8)


def heatmap_rgba(heatmap: np.ndarray) -> Image.Image:
    """Transparent where attention is low so the browser can blend it over the scan."""
    rgb = colorize(heatmap)
    alpha = (np.clip(heatmap, 0, 1) ** 0.8 * 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]).astype(np.uint8))


def overlay(image: Image.Image, heatmap: np.ndarray, strength: float = 0.45) -> Image.Image:
    base = np.asarray(image.convert("RGB").resize(heatmap.shape[::-1]), dtype=np.float32)
    colored = colorize(heatmap).astype(np.float32)
    alpha = (strength * np.clip(heatmap, 0, 1))[..., None]
    return Image.fromarray(((1 - alpha) * base + alpha * colored).clip(0, 255).astype(np.uint8))


def to_data_uri(image: Image.Image, fmt: str = "JPEG") -> str:
    buffer = io.BytesIO()
    if fmt == "JPEG":
        image.convert("RGB").save(buffer, format="JPEG", quality=90)
        mime = "image/jpeg"
    else:
        image.save(buffer, format="PNG", optimize=True)
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@dataclass
class ServingModel:
    """Backend-independent description of a loaded model."""

    key: str
    display_name: str
    description: str
    arch: str
    temperature: float
    preprocess: PreprocessConfig
    val_metrics: Dict[str, Any] = field(default_factory=dict)
    checkpoint_format: int = 1
    supports_gradcam: bool = False
    parameters_total: int = 0
    source_file: str = ""
    source_size_mb: float = 0.0
    weight: float = 1.0
    backend: str = "onnx"

    @property
    def is_legacy(self) -> bool:
        return self.checkpoint_format < 2

    def analyze(self, image: Image.Image, tta: bool = True) -> np.ndarray:
        """Calibrated class probabilities, shape (classes,)."""
        raise NotImplementedError

    def explain(self, image: Image.Image, class_idx: int) -> Optional[np.ndarray]:
        """Grad-CAM heatmap in [0, 1] at the model's input resolution, or None."""
        return None

    def public_info(self) -> Dict[str, Any]:
        return {
            "arch": self.arch,
            "backend": self.backend,
            "format_version": self.checkpoint_format,
            "temperature": round(float(self.temperature), 4),
            "preprocess": self.preprocess.to_dict(),
            "val_metrics": self.val_metrics,
            "parameters_total": self.parameters_total,
            "checkpoint_size_mb": self.source_size_mb,
            "supports_gradcam": self.supports_gradcam,
        }


class OnnxModel(ServingModel):
    def __init__(self, session: Any, **kwargs: Any):
        super().__init__(**kwargs)
        self.session = session

    def _run(self, batch: np.ndarray):
        return self.session.run(None, {"input": batch})

    def analyze(self, image: Image.Image, tta: bool = True) -> np.ndarray:
        x = to_input_array(image, self.preprocess)
        batch = np.ascontiguousarray(np.concatenate([x, x[..., ::-1]])) if tta else x
        logits = self._run(batch)[0].astype(np.float64)
        mean_logits = logits.mean(axis=0) if tta else logits[0]
        return softmax(mean_logits / self.temperature)

    def explain(self, image: Image.Image, class_idx: int) -> Optional[np.ndarray]:
        if not self.supports_gradcam:
            return None
        cams = self._run(to_input_array(image, self.preprocess))[1]
        return normalize_cam(cams[0, class_idx], self.preprocess.image_size)


def _session_options():
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    threads = env_number("NEUROSCAN_ORT_THREADS", 0, cast=int, minimum=0)  # 0: onnxruntime decides
    if threads > 0:
        options.intra_op_num_threads = threads
    return options


def onnx_runtime_available() -> bool:
    try:
        importlib.import_module("onnxruntime")
    except ImportError:
        return False
    return True


def load_onnx_models(onnx_dir: Path) -> Tuple[Dict[str, OnnxModel], Dict[str, str]]:
    """Load every model listed in ``onnx_dir/manifest.json``. Never raises."""
    onnx_dir = Path(onnx_dir)
    models: Dict[str, OnnxModel] = {}
    unavailable: Dict[str, str] = {}
    try:
        manifest = json.loads((onnx_dir / "manifest.json").read_text())
        entries = manifest.get("models", {})
        if list(manifest.get("class_names", CLASS_NAMES)) != list(CLASS_NAMES):
            raise ValueError(f"class order {manifest.get('class_names')} differs from {CLASS_NAMES}")
    except (OSError, ValueError, AttributeError) as exc:
        reason = f"ONNX manifest unreadable: {exc.__class__.__name__}"
        logger.error("%s (%s)", reason, onnx_dir / "manifest.json")
        return {}, {key: reason for key in MODEL_SPECS}

    import onnxruntime as ort

    options = _session_options()
    excluded = manifest.get("excluded") or {}
    for key, spec in MODEL_SPECS.items():
        entry = entries.get(key)
        if entry is None:
            unavailable[key] = ("not deployed: too large for a serverless function" if key in excluded
                                else f"no ONNX export for {spec['checkpoint']}")
            continue
        path = onnx_dir / str(entry.get("file", ""))
        if not path.is_file():
            unavailable[key] = f"{entry.get('file')} not found"
            continue
        try:
            session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
            models[key] = OnnxModel(
                session=session,
                key=key,
                display_name=spec["display_name"],
                description=spec["description"],
                arch=str(entry.get("arch", key)),
                temperature=max(float(entry.get("temperature", 1.0)), 1e-3),
                preprocess=PreprocessConfig.from_dict(entry.get("preprocess")),
                val_metrics=dict(entry.get("val_metrics") or {}),
                checkpoint_format=int(entry.get("checkpoint_format", 1)),
                supports_gradcam=bool(entry.get("supports_gradcam")) and len(session.get_outputs()) > 1,
                parameters_total=int(entry.get("parameters_total", 0)),
                source_file=path.name,
                source_size_mb=round(path.stat().st_size / 2**20, 2),
                backend="onnx",
            )
        except Exception as exc:  # corrupt file, bad metadata: report, never crash the app
            logger.exception("Failed to load %s", path)
            unavailable[key] = f"failed to load: {exc.__class__.__name__}"
    return models, unavailable
