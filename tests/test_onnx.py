"""ONNX export: numerical parity with PyTorch (logits and Grad-CAM) and backend equivalence."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("torch")  # PyTorch tests are skipped in the serving-only (Vercel) environment
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import torch  # noqa: E402

from src.config import MODELS_DIR, PUBLIC_DIR  # noqa: E402
from src.inference import InferenceService  # noqa: E402
from src.models import BrainTumorCNN, build_model  # noqa: E402
from src.onnx_export import StaticAdaptiveAvgPool2d, export_onnx, verify_export  # noqa: E402


def _randomize_batchnorm(model: torch.nn.Module, seed: int = 0) -> torch.nn.Module:
    """Non-trivial BN statistics so parity checks are meaningful for random weights."""
    gen = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.weight.data = torch.rand(m.weight.shape, generator=gen) + 0.5
            m.bias.data = torch.rand(m.bias.shape, generator=gen) - 0.5
            m.running_mean = torch.rand(m.running_mean.shape, generator=gen) * 0.4 - 0.2
            m.running_var = torch.rand(m.running_var.shape, generator=gen) + 0.5
    return model.eval()


@pytest.mark.parametrize("name,arch,has_cam", [
    ("custom_cnn", "custom_cnn_v1", False),
    ("custom_cnn", "custom_cnn_v2", True),
    ("resnet18", None, True),
    ("mobilenet_v3", None, True),
])
def test_export_matches_pytorch(tmp_path, name, arch, has_cam):
    torch.manual_seed(0)
    model = _randomize_batchnorm(build_model(name, pretrained=False, arch=arch))
    path = tmp_path / "model.onnx"
    assert export_onnx(model, path) is has_cam
    assert not model.training, "export must leave the model in eval mode"
    report = verify_export(model, path)
    assert report["max_abs_logit_diff"] < 1e-3
    if has_cam:
        assert report["max_abs_cam_diff"] < 1e-3  # analytic Grad-CAM == autograd Grad-CAM


def test_static_adaptive_pool_matches_torch():
    x = torch.randn(3, 5, 14, 14)
    for size in [(4, 4), (3, 5), (1, 1), (14, 14)]:
        ref = torch.nn.functional.adaptive_avg_pool2d(x, size)
        assert torch.allclose(StaticAdaptiveAvgPool2d(size)(x), ref, atol=1e-6)


def test_legacy_cnn_is_exported_through_the_static_pool(tmp_path):
    model = BrainTumorCNN().eval()
    export_onnx(model, tmp_path / "v1.onnx")
    assert isinstance(model.adaptive_pool, torch.nn.AdaptiveAvgPool2d), "original model must not be modified"


@pytest.mark.skipif(not (MODELS_DIR / "onnx" / "manifest.json").exists(), reason="no ONNX export")
def test_onnx_and_pytorch_backends_agree_on_every_sample():
    onnx = InferenceService(backend="onnx", benchmark_path=None).load()
    torch_svc = InferenceService(backend="torch", device=torch.device("cpu"), benchmark_path=None).load()
    assert onnx.backend == "onnx" and torch_svc.backend == "torch"
    assert set(onnx.models) == set(torch_svc.models)
    samples = json.loads((PUBLIC_DIR / "samples" / "index.json").read_text())["samples"]
    for sample in samples:
        path = PUBLIC_DIR / "samples" / sample["id"]
        a, b = onnx.predict(path), torch_svc.predict(path, with_gradcam=False)
        assert a["consensus_prediction"] == b["consensus_prediction"], sample["id"]
        for cls, value in a["ensemble_probabilities"].items():
            assert abs(value - b["ensemble_probabilities"][cls]) < 0.05, (sample["id"], cls)


@pytest.mark.skipif(not (MODELS_DIR / "onnx" / "manifest.json").exists(), reason="no ONNX export")
def test_onnx_gradcam_matches_pytorch_gradcam():
    from src.runtime import load_image

    onnx = InferenceService(backend="onnx", benchmark_path=None).load()
    torch_svc = InferenceService(backend="torch", device=torch.device("cpu"), benchmark_path=None).load()
    image = load_image(PUBLIC_DIR / "samples" / "glioma" / "Te-gl_10.jpg")
    for key in ("resnet18", "mobilenet_v3"):
        for cls in range(4):
            heat_onnx = onnx.models[key].explain(image, cls)
            heat_torch = torch_svc.models[key].explain(image, cls)
            assert heat_onnx.shape == heat_torch.shape
            assert float(np.abs(heat_onnx - heat_torch).max()) < 0.02, (key, cls)
