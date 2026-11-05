"""Inference service shared by the FastAPI server, the Gradio app and the CLI.

Fixes relative to the original implementation:
* only models with a real checkpoint are loaded - a missing ViT checkpoint no
  longer turns into a randomly initialised network that votes in the consensus;
* each model uses the preprocessing and calibration temperature stored in its
  checkpoint, plus optional flip test-time augmentation;
* the ensemble is reliability-weighted and reports when it is uncertain;
* Grad-CAM is computed natively (no OpenCV / grad-cam dependency), for the most
  reliable CNN and for the class the *ensemble* predicted, and is thread-safe.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .config import CLASS_METADATA, CLASS_NAMES, MEDICAL_DISCLAIMER, MODELS_DIR, RESULTS_DIR, get_device
from .data import get_test_transforms
from .ensemble import assess_uncertainty, reliability_weight, weighted_average
from .models import MODEL_REGISTRY, CheckpointInfo, gradcam_target_layer, load_model_from_checkpoint
from .preprocessing import LEGACY_PREPROCESS, PreprocessConfig, assess_image_quality, preprocess_image

logger = logging.getLogger(__name__)
ImageInput = Union[str, Path, Image.Image, np.ndarray]


def to_pil(image_input: ImageInput) -> Image.Image:
    if isinstance(image_input, (str, Path)):
        with Image.open(image_input) as img:
            return img.convert("RGB")
    if isinstance(image_input, np.ndarray):
        return Image.fromarray(image_input).convert("RGB")
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    raise ValueError(f"Unsupported image type: {type(image_input)}")


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #
def compute_gradcam(model: nn.Module, input_tensor: torch.Tensor, class_idx: int) -> Optional[np.ndarray]:
    """Grad-CAM (Selvaraju et al. 2017) -> heatmap in [0, 1] at input resolution."""
    layer = gradcam_target_layer(model)
    if layer is None:
        return None
    store: Dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        store["activations"] = output
        output.register_hook(lambda grad: store.__setitem__("gradients", grad))

    handle = layer.register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(input_tensor.detach().clone().requires_grad_(True))
            logits[0, class_idx].backward()
        activations = store["activations"].detach()[0]
        gradients = store["gradients"].detach()[0]
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)

    weights = gradients.mean(dim=(1, 2))
    cam = F.relu((weights[:, None, None] * activations).sum(dim=0))
    if float(cam.max()) <= 1e-8:
        return np.zeros(input_tensor.shape[-2:], dtype=np.float32)
    cam = cam / cam.max()
    cam = F.interpolate(cam[None, None], size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return cam.cpu().numpy().astype(np.float32)


_COLORMAP_ANCHORS = np.array(
    [[0.00, 0, 0, 128], [0.25, 0, 96, 255], [0.50, 0, 220, 200], [0.75, 255, 210, 0], [1.00, 220, 20, 20]],
    dtype=np.float32,
)


def colorize(heatmap: np.ndarray) -> np.ndarray:
    """Map [0,1] -> RGB uint8 with a perceptually ordered blue->cyan->yellow->red ramp."""
    flat = heatmap.ravel()
    channels = [np.interp(flat, _COLORMAP_ANCHORS[:, 0], _COLORMAP_ANCHORS[:, i]) for i in (1, 2, 3)]
    return np.stack(channels, axis=-1).reshape(*heatmap.shape, 3).astype(np.uint8)


def heatmap_rgba(heatmap: np.ndarray) -> Image.Image:
    """Transparent where attention is low so the browser can blend it over the scan."""
    rgb = colorize(heatmap)
    alpha = (np.clip(heatmap, 0, 1) ** 0.8 * 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")


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
# Loaded model wrapper
# --------------------------------------------------------------------------- #
@dataclass
class LoadedModel:
    key: str
    display_name: str
    description: str
    module: nn.Module
    info: CheckpointInfo
    checkpoint_path: Path
    weight: float

    def __post_init__(self) -> None:
        self.transform = get_test_transforms(self.info.preprocess.image_size, self.info.preprocess)

    @property
    def preprocess(self) -> PreprocessConfig:
        return self.info.preprocess

    def input_tensor(self, image: Image.Image, device: torch.device) -> torch.Tensor:
        return self.transform(image).unsqueeze(0).to(device)

    @torch.no_grad()
    def probabilities(self, image: Image.Image, device: torch.device, tta: bool = True) -> np.ndarray:
        x = self.input_tensor(image, device)
        logits = self.module(x)
        if tta:
            logits = 0.5 * (logits + self.module(torch.flip(x, dims=[3])))
        return F.softmax(logits / self.info.temperature, dim=1)[0].cpu().numpy()


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
class InferenceService:
    def __init__(
        self,
        models_dir: Path = MODELS_DIR,
        device: Optional[torch.device] = None,
        benchmark_path: Optional[Path] = RESULTS_DIR / "model_comparison_metrics.json",
        tta: bool = True,
    ):
        self.models_dir = Path(models_dir)
        self.device = device or get_device()
        self.benchmark_path = benchmark_path
        self.tta = tta
        self.models: Dict[str, LoadedModel] = {}
        self.unavailable: Dict[str, str] = {}
        self._lock = threading.RLock()

    # ---------------------------------------------------------------- loading
    def _benchmark(self) -> Dict[str, Any]:
        if self.benchmark_path and Path(self.benchmark_path).exists():
            try:
                with open(self.benchmark_path) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                logger.warning("Could not read %s", self.benchmark_path)
        return {}

    def load(self) -> "InferenceService":
        """(Re)load every available checkpoint. Builds a new registry, then swaps it
        in atomically so in-flight requests never see a half-loaded state."""
        models: Dict[str, LoadedModel] = {}
        unavailable: Dict[str, str] = {}
        benchmark = self._benchmark()
        for key, spec in MODEL_REGISTRY.items():
            path = self.models_dir / spec["checkpoint"]
            if not path.exists():
                unavailable[key] = f"checkpoint {spec['checkpoint']} not found"
                continue
            try:
                module, info = load_model_from_checkpoint(key, path, self.device)
            except Exception as exc:  # corrupt / incompatible file: skip, never crash the app
                logger.exception("Failed to load %s", path)
                unavailable[key] = f"failed to load: {exc.__class__.__name__}"
                continue
            weight = reliability_weight(info.val_metrics, benchmark.get(key))
            models[key] = LoadedModel(key, spec["display_name"], spec["description"], module, info, path, weight)
            logger.info("Loaded %s (%s, T=%.2f, weight=%.2f)", spec["display_name"], info.arch, info.temperature, weight)
        with self._lock:
            self.models, self.unavailable = models, unavailable
        return self

    # ---------------------------------------------------------------- status
    def status(self) -> Dict[str, Any]:
        with self._lock:
            loaded, unavailable = dict(self.models), dict(self.unavailable)
        total_weight = sum(m.weight for m in loaded.values()) or 1.0
        rows = []
        for key, spec in MODEL_REGISTRY.items():
            row = {
                "key": key,
                "display_name": spec["display_name"],
                "description": spec["description"],
                "checkpoint": spec["checkpoint"],
                "is_trained": key in loaded,
            }
            if key in loaded:
                m = loaded[key]
                params = sum(p.numel() for p in m.module.parameters())
                row.update({
                    "checkpoint_size_mb": round(m.checkpoint_path.stat().st_size / 2**20, 2),
                    "parameters_total": params,
                    "ensemble_weight": round(m.weight / total_weight, 4),
                    "supports_gradcam": gradcam_target_layer(m.module) is not None,
                    **m.info.to_public_dict(),
                })
            else:
                row["unavailable_reason"] = unavailable.get(key, "not loaded")
            rows.append(row)
        return {
            "device": str(self.device),
            "tta": self.tta,
            "classes": CLASS_NAMES,
            "class_metadata": CLASS_METADATA,
            "models": rows,
            "loaded_models": len(loaded),
            "total_models": len(MODEL_REGISTRY),
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    # ---------------------------------------------------------------- predict
    def _gradcam_model(self, models: Dict[str, LoadedModel]) -> Optional[LoadedModel]:
        candidates = [m for m in models.values() if gradcam_target_layer(m.module) is not None]
        return max(candidates, key=lambda m: m.weight) if candidates else None

    def predict(self, image: ImageInput, with_gradcam: bool = True) -> Dict[str, Any]:
        start = time.perf_counter()
        pil = to_pil(image)
        quality = assess_image_quality(pil)
        with self._lock:
            models = dict(self.models)
            unavailable = dict(self.unavailable)
            if not models:
                raise RuntimeError("No trained model checkpoints are available on the server.")

            per_model, prob_rows, weights, votes = [], [], [], []
            own_prediction: Dict[str, int] = {}
            for key, m in models.items():
                probs = m.probabilities(pil, self.device, self.tta)
                idx = int(np.argmax(probs))
                own_prediction[key] = idx
                meta = CLASS_METADATA[CLASS_NAMES[idx]]
                prob_rows.append(probs)
                weights.append(m.weight)
                votes.append(idx)
                per_model.append({
                    "model_key": key,
                    "display_name": m.display_name,
                    "predicted_class": CLASS_NAMES[idx],
                    "confidence_pct": round(float(probs[idx]) * 100.0, 2),
                    "confidences": {c: round(float(p) * 100.0, 2) for c, p in zip(CLASS_NAMES, probs)},
                    "color": meta["color"],
                    "severity": meta["severity"],
                    "description": meta["simple_summary"],
                    "weight": m.weight,
                    "temperature": round(m.info.temperature, 3),
                    "calibrated": abs(m.info.temperature - 1.0) > 1e-6,
                    "legacy_checkpoint": m.info.is_legacy,
                    "is_checkpoint_loaded": True,
                })

            ensemble = weighted_average(prob_rows, weights)
            verdict = assess_uncertainty(ensemble, votes, weights)
            total_weight = sum(weights)
            for row in per_model:
                row["weight"] = round(row["weight"] / total_weight, 4)

            gradcam = None
            cam_model = self._gradcam_model(models) if with_gradcam else None
            if cam_model is not None:
                # Explain what this model itself predicted: a heatmap for a class
                # the model did not choose is hard to interpret.
                gradcam = self._gradcam_payload(cam_model, pil, own_prediction[cam_model.key])

        return self._build_response(pil, quality, per_model, ensemble, verdict, gradcam, unavailable,
                                    (time.perf_counter() - start) * 1000.0)

    def _gradcam_payload(self, m: LoadedModel, pil: Image.Image, class_idx: int) -> Optional[Dict[str, Any]]:
        try:
            heat = compute_gradcam(m.module, m.input_tensor(pil, self.device), class_idx)
        except Exception:  # explainability must never break a prediction
            logger.exception("Grad-CAM failed for %s", m.key)
            return None
        if heat is None:
            return None
        model_view = preprocess_image(pil, m.preprocess)
        return {
            "model_key": m.key,
            "display_name": m.display_name,
            "target_class": CLASS_NAMES[class_idx],
            "input_b64": to_data_uri(model_view),
            "heatmap_b64": to_data_uri(heatmap_rgba(heat), fmt="PNG"),
            "overlay_b64": to_data_uri(overlay(model_view, heat)),
        }

    def _build_response(self, pil, quality, per_model, ensemble, verdict, gradcam, unavailable, elapsed_ms):
        top = verdict["predicted_class"]
        meta = CLASS_METADATA[top]
        total = verdict["total_models"]
        agree = verdict["agreement_count"]
        warnings = list(quality["warnings"])

        if verdict["inconclusive"]:
            title = "Inconclusive \u2014 expert review needed"
            summary = (
                f"The AI models could not reach a confident result for this scan "
                f"({'; '.join(verdict['reasons'])}). The most likely pattern is "
                f"'{meta['short_label']}' at {verdict['probability'] * 100:.0f}%, but this should "
                "not be relied on. Please have the scan reviewed by a radiologist."
            )
        else:
            title = meta["display_name"]
            agreement = "All" if agree == total else f"{agree} of"
            summary = f"{agreement} {total} AI model{'s' if total != 1 else ''} matched this scan to the '{meta['short_label']}' pattern. "
            summary += f"{meta['simple_summary']} {meta['what_it_means']}"
        if warnings:
            summary += " Note: " + " ".join(warnings)

        ensemble_pct = {c: round(float(p) * 100.0, 2) for c, p in zip(CLASS_NAMES, ensemble)}
        guide = []
        for cls in CLASS_NAMES:
            m = CLASS_METADATA[cls]
            guide.append({
                "class_key": cls,
                "display_name": m["display_name"],
                "color": m["color"],
                "icon": m["icon"],
                "severity": m["severity"],
                "severity_level": m["severity_level"],
                "simple_summary": m["simple_summary"],
                "plain_details": m["plain_details"],
                "what_it_means": m["what_it_means"],
                "average_confidence_pct": ensemble_pct[cls],
                "is_detected": cls == top,
                "model_votes": sum(1 for r in per_model if r["predicted_class"] == cls),
                "total_models": total,
            })

        fallback_view = to_data_uri(pil.resize((224, 224)))
        return {
            "success": True,
            "consensus_prediction": top,
            "consensus_display_name": title,
            "consensus_confidence": round(verdict["probability"] * 100.0, 2),
            "confidence_level": verdict["confidence_level"],
            "inconclusive": verdict["inconclusive"],
            "uncertainty": verdict,
            "agreement_count": agree,
            "total_models": total,
            "agreement_text": f"{agree} of {total} models agree",
            "plain_summary": summary,
            "severity": meta["severity"],
            "severity_level": "unknown" if verdict["inconclusive"] else meta["severity_level"],
            "ensemble_probabilities": ensemble_pct,
            "categories_guide": guide,
            "model_results": per_model,
            "unavailable_models": [
                {"model_key": k, "display_name": MODEL_REGISTRY[k]["display_name"], "reason": reason}
                for k, reason in unavailable.items()
            ],
            "gradcam": gradcam,
            "original_image_b64": gradcam["input_b64"] if gradcam else fallback_view,
            "gradcam_image_b64": gradcam["overlay_b64"] if gradcam else None,
            "image_quality": quality,
            "warnings": warnings,
            "tta": self.tta,
            "disclaimer": MEDICAL_DISCLAIMER,
            "inference_ms": round(elapsed_ms, 1),
        }


# --------------------------------------------------------------------------- #
# Backwards-compatible functional API
# --------------------------------------------------------------------------- #
def prepare_image(image_input: ImageInput, preprocess: PreprocessConfig = LEGACY_PREPROCESS):
    pil = to_pil(image_input)
    return get_test_transforms(preprocess.image_size, preprocess)(pil).unsqueeze(0), pil


def predict_single_image(
    model: nn.Module,
    image_input: ImageInput,
    device: torch.device,
    class_names: List[str] = CLASS_NAMES,
    preprocess: PreprocessConfig = LEGACY_PREPROCESS,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    tensor, _ = prepare_image(image_input, preprocess)
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(tensor.to(device)) / temperature, dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return {
        "predicted_class": class_names[idx],
        "confidence": float(probs[idx]),
        "confidences": {class_names[i]: float(p) for i, p in enumerate(probs)},
    }


def generate_gradcam_overlay(
    model: nn.Module,
    image_input: ImageInput,
    device: torch.device,
    target_category: Optional[int] = None,
    preprocess: PreprocessConfig = LEGACY_PREPROCESS,
) -> Optional[np.ndarray]:
    tensor, pil = prepare_image(image_input, preprocess)
    tensor = tensor.to(device)
    if target_category is None:
        with torch.no_grad():
            target_category = int(model(tensor).argmax(dim=1))
    heat = compute_gradcam(model, tensor, target_category)
    if heat is None:
        return None
    return np.asarray(overlay(preprocess_image(pil, preprocess), heat))


def get_available_trained_models(device: Optional[torch.device] = None, models_dir: Path = MODELS_DIR) -> Dict[str, nn.Module]:
    service = InferenceService(models_dir=models_dir, device=device, benchmark_path=None).load()
    return {m.display_name: m.module for m in service.models.values()}
