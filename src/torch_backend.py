"""PyTorch serving backend (local development / evaluation).

Used when no ONNX export is present or when NEUROSCAN_BACKEND=torch. The web
deployment uses the ONNX backend in ``src.runtime`` instead, because PyTorch
does not fit into a serverless function.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .config import CLASS_NAMES, MODEL_SPECS, MODELS_DIR, get_device
from .data import get_test_transforms
from .models import count_parameters, gradcam_target_layer, load_model_from_checkpoint
from .preprocessing import LEGACY_PREPROCESS, PreprocessConfig, preprocess_image
from .runtime import ImageInput, ServingModel, load_image, overlay

logger = __import__("logging").getLogger(__name__)


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


class TorchModel(ServingModel):
    def __init__(self, module: nn.Module, device: torch.device, **kwargs: Any):
        super().__init__(**kwargs)
        self.module = module
        self.device = device
        self.transform = get_test_transforms(self.preprocess.image_size, self.preprocess)

    def input_tensor(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def analyze(self, image: Image.Image, tta: bool = True) -> np.ndarray:
        x = self.input_tensor(image)
        logits = self.module(x)
        if tta:
            logits = 0.5 * (logits + self.module(torch.flip(x, dims=[3])))
        return F.softmax(logits / self.temperature, dim=1)[0].cpu().numpy()

    def explain(self, image: Image.Image, class_idx: int) -> Optional[np.ndarray]:
        if not self.supports_gradcam:
            return None
        return compute_gradcam(self.module, self.input_tensor(image), class_idx)


def load_torch_models(models_dir: Path, device: Optional[torch.device] = None) -> Tuple[Dict[str, TorchModel], Dict[str, str]]:
    device = device or get_device()
    models: Dict[str, TorchModel] = {}
    unavailable: Dict[str, str] = {}
    for key, spec in MODEL_SPECS.items():
        path = Path(models_dir) / spec["checkpoint"]
        if not path.exists():
            unavailable[key] = f"checkpoint {spec['checkpoint']} not found"
            continue
        try:
            module, info = load_model_from_checkpoint(key, path, device)
        except Exception as exc:  # corrupt / incompatible file: skip, never crash the app
            logger.exception("Failed to load %s", path)
            unavailable[key] = f"failed to load: {exc.__class__.__name__}"
            continue
        models[key] = TorchModel(
            module=module, device=device, key=key,
            display_name=spec["display_name"], description=spec["description"],
            arch=info.arch, temperature=info.temperature, preprocess=info.preprocess,
            val_metrics=info.val_metrics, checkpoint_format=info.format_version,
            supports_gradcam=gradcam_target_layer(module) is not None,
            parameters_total=count_parameters(module)["total"], source_file=path.name,
            source_size_mb=round(path.stat().st_size / 2**20, 2), backend="torch",
        )
    return models, unavailable


# --------------------------------------------------------------------------- #
# Backwards-compatible functional API (notebooks / older scripts)
# --------------------------------------------------------------------------- #
def prepare_image(image_input: ImageInput, preprocess: PreprocessConfig = LEGACY_PREPROCESS):
    pil = load_image(image_input)
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
    models, _ = load_torch_models(models_dir, device)
    return {m.display_name: m.module for m in models.values()}
