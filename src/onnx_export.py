"""Export trained checkpoints to ONNX for PyTorch-free serving (e.g. on Vercel).

PyTorch does not fit into a serverless function, so the web app runs the models
with ONNX Runtime. Each exported graph returns

* ``logits`` - shape (batch, classes);
* ``cams``   - shape (batch, classes, h, w): Grad-CAM maps for *every* class.

Grad-CAM normally needs back-propagation, which ONNX Runtime cannot do. For the
supported CNNs the explained layer is followed by global average pooling and a
small head, so the Grad-CAM channel weights are exactly the head's Jacobian:

    alpha_c = d logit_c / d pooled  (divided by h*w, which normalisation removes)
    cam_c   = ReLU(sum_k alpha_ck * A_k)

That Jacobian is computed in closed form inside the graph (Linear -> activation
-> Linear heads), so the exported maps match autograd Grad-CAM (verified by
``verify_export`` and the test-suite).
"""
from __future__ import annotations

import copy
import datetime as _dt
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models

from .config import CLASS_NAMES, MODEL_SPECS
from .models import BrainTumorCNN, BrainTumorCNNv2, count_parameters, load_model_from_checkpoint

ONNX_OPSET = 18
MANIFEST_VERSION = 1


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _mlp_jacobian(pooled: torch.Tensor, first: nn.Linear, act: str, last: nn.Linear) -> torch.Tensor:
    """d logits / d pooled for Linear -> act -> (Dropout) -> Linear, shape (B, K, C)."""
    hidden = F.linear(pooled, first.weight, first.bias)                       # (B, H)
    if act == "relu":
        slope = (hidden > 0).to(pooled.dtype)
    elif act == "hardswish":  # derivative used by torch.nn.functional.hardswish backward
        slope = torch.where(hidden < -3, torch.zeros_like(hidden),
                            torch.where(hidden > 3, torch.ones_like(hidden), hidden / 3 + 0.5))
    else:
        raise ValueError(act)
    # (K, H) * (B, 1, H) -> (B, K, H) @ (H, C) -> (B, K, C)
    return torch.matmul(last.weight.unsqueeze(0) * slope.unsqueeze(1), first.weight)


class ExplainableClassifier(nn.Module):
    """Wraps a trained model so one forward pass yields logits and Grad-CAM maps."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()
        if isinstance(model, tv_models.ResNet):
            self.kind = "resnet"
        elif isinstance(model, tv_models.MobileNetV3):
            self.kind = "mobilenet"
        elif isinstance(model, BrainTumorCNNv2):
            self.kind = "custom_v2"
        else:
            raise TypeError(f"Grad-CAM export not supported for {type(model).__name__}")

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        m = self.model
        if self.kind == "resnet":
            x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
            return m.layer4(m.layer3(m.layer2(m.layer1(x))))
        if self.kind == "mobilenet":
            return m.features(x)
        return m.stages(m.stem(x))

    def forward(self, x: torch.Tensor):
        feats = self._features(x)                       # (B, C, h, w) - the explained layer
        pooled = feats.mean(dim=(2, 3))                 # == AdaptiveAvgPool2d(1) + flatten
        m = self.model
        if self.kind == "resnet":
            first, last, act = m.fc[0], m.fc[3], "relu"
            logits = m.fc(pooled)
            jac = _mlp_jacobian(pooled, first, act, last)
        elif self.kind == "mobilenet":
            first, last, act = m.classifier[0], m.classifier[3], "hardswish"
            logits = m.classifier(pooled)
            jac = _mlp_jacobian(pooled, first, act, last)
        else:
            linear = m.head[2]
            logits = linear(pooled)
            jac = linear.weight.unsqueeze(0).expand(pooled.shape[0], -1, -1)
        cams = F.relu(torch.einsum("bkc,bchw->bkhw", jac, feats))
        return logits, cams


class StaticAdaptiveAvgPool2d(nn.Module):
    """AdaptiveAvgPool2d written as fixed slices so it exports to ONNX.

    ONNX cannot express adaptive pooling when the output size does not divide
    the input size (the legacy custom CNN pools 14x14 -> 4x4). With a fixed
    input resolution the bins are static: [floor(i*H/o), ceil((i+1)*H/o)).
    """

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = int(x.shape[-2]), int(x.shape[-1])
        out_h, out_w = self.output_size
        rows = []
        for i in range(out_h):
            top, bottom = (i * height) // out_h, -((-(i + 1) * height) // out_h)
            cols = []
            for j in range(out_w):
                left, right = (j * width) // out_w, -((-(j + 1) * width) // out_w)
                cols.append(x[:, :, top:bottom, left:right].mean(dim=(2, 3)))
            rows.append(torch.stack(cols, dim=-1))
        return torch.stack(rows, dim=-2)


def exportable_copy(model: nn.Module) -> nn.Module:
    """Deep copy with ONNX-unfriendly layers replaced by exact equivalents."""
    clone = copy.deepcopy(model).eval()
    if isinstance(clone, BrainTumorCNN):
        clone.adaptive_pool = StaticAdaptiveAvgPool2d(clone.adaptive_pool.output_size)
    return clone


class LogitsOnly(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()

    def forward(self, x: torch.Tensor):
        return self.model(x)


def supports_gradcam(model: nn.Module) -> bool:
    return isinstance(model, (tv_models.ResNet, tv_models.MobileNetV3, BrainTumorCNNv2))


def export_onnx(model: nn.Module, out_path: Path, image_size: int = 224) -> bool:
    """Export with a dynamic batch dimension (TTA runs image + mirror in one call)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with_cams = supports_gradcam(model)
    model = exportable_copy(model)
    # A new nn.Module defaults to training=True and the exporter restores the
    # wrapper's original mode afterwards - which would flip the shared model
    # into training mode (BatchNorm batch statistics, active dropout).
    wrapper = (ExplainableClassifier(model) if with_cams else LogitsOnly(model)).eval()
    dummy = torch.randn(2, 3, image_size, image_size)
    output_names = ["logits", "cams"] if with_cams else ["logits"]
    dynamic_axes = {"input": {0: "batch"}, **{name: {0: "batch"} for name in output_names}}
    tmp = out_path.with_suffix(".onnx.tmp")
    import warnings

    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tracer warnings about static shapes are expected
        torch.onnx.export(
            wrapper, (dummy,), str(tmp), dynamo=False, opset_version=ONNX_OPSET,
            input_names=["input"], output_names=output_names, dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
    tmp.replace(out_path)
    model.eval()
    return with_cams


def autograd_gradcam(model: nn.Module, x: torch.Tensor, class_idx: int) -> np.ndarray:
    """Reference Grad-CAM via autograd at feature resolution (for verification)."""
    from .models import gradcam_target_layer

    layer = gradcam_target_layer(model)
    store: Dict[str, torch.Tensor] = {}

    def hook(_m, _i, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    handle = layer.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(x.clone().requires_grad_(True))
            logits[0, class_idx].backward()
    finally:
        handle.remove()
    weights = store["g"][0].mean(dim=(1, 2))
    return F.relu((weights[:, None, None] * store["a"][0]).sum(0)).detach().numpy()


def verify_export(model: nn.Module, onnx_path: Path, image_size: int = 224, seed: int = 0) -> Dict[str, float]:
    """Compare ONNX Runtime against PyTorch: logits and (normalised) Grad-CAM maps."""
    import onnxruntime as ort

    model.eval()
    torch.manual_seed(seed)
    x = torch.randn(2, 3, image_size, image_size)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"input": x.numpy()})
    with torch.no_grad():
        ref = model(x).numpy()
    report = {"max_abs_logit_diff": float(np.abs(outputs[0] - ref).max())}
    if len(outputs) > 1:
        worst = 0.0
        for k in range(len(CLASS_NAMES)):
            ref_cam = autograd_gradcam(model, x[:1], k)
            onnx_cam = outputs[1][0, k]
            ref_n = ref_cam / ref_cam.max() if ref_cam.max() > 1e-8 else ref_cam
            onnx_n = onnx_cam / onnx_cam.max() if onnx_cam.max() > 1e-8 else onnx_cam
            worst = max(worst, float(np.abs(ref_n - onnx_n).max()))
        report["max_abs_cam_diff"] = worst
    return report


def export_checkpoint(key: str, checkpoint: Path, out_dir: Path, verify: bool = True) -> Dict[str, Any]:
    """Export one checkpoint and return its manifest entry."""
    model, info = load_model_from_checkpoint(key, checkpoint, torch.device("cpu"))
    out_path = Path(out_dir) / f"{key}.onnx"
    with_cams = export_onnx(model, out_path, info.preprocess.image_size)
    entry: Dict[str, Any] = {
        "file": out_path.name,
        "sha256": sha256_file(out_path),
        "source_checkpoint": Path(checkpoint).name,
        "source_sha256": sha256_file(Path(checkpoint)),
        "arch": info.arch,
        "display_name": MODEL_SPECS[key]["display_name"],
        "description": MODEL_SPECS[key]["description"],
        "temperature": float(info.temperature),
        "preprocess": info.preprocess.to_dict(),
        "val_metrics": info.val_metrics,
        "checkpoint_format": info.format_version,
        "supports_gradcam": with_cams,
        "parameters_total": count_parameters(model)["total"],
        "outputs": ["logits", "cams"] if with_cams else ["logits"],
    }
    if verify:
        entry["verification"] = verify_export(model, out_path, info.preprocess.image_size)
    return entry


def build_manifest(entries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "exported_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "torch_version": str(torch.__version__),
        "opset": ONNX_OPSET,
        "class_names": list(CLASS_NAMES),
        "models": entries,
    }


def stale_models(manifest: Dict[str, Any], models_dir: Path) -> Dict[str, str]:
    """Models whose ONNX export no longer matches the .pth checkpoint next to it."""
    problems: Dict[str, str] = {}
    excluded = manifest.get("excluded", {})
    for key, spec in MODEL_SPECS.items():
        if key in excluded:
            continue
        ckpt = Path(models_dir) / spec["checkpoint"]
        entry: Optional[Dict[str, Any]] = manifest.get("models", {}).get(key)
        if ckpt.exists() and entry is None:
            problems[key] = "checkpoint has no ONNX export"
        elif ckpt.exists() and entry["source_sha256"] != sha256_file(ckpt):
            problems[key] = "checkpoint changed since export"
    return problems
