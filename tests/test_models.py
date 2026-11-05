from pathlib import Path

import pytest
import torch

from src.config import CLASS_NAMES, MODELS_DIR
from src.models import (
    BrainTumorCNN,
    BrainTumorCNNv2,
    build_model,
    gradcam_target_layer,
    load_model_from_checkpoint,
    parameter_groups,
    read_checkpoint,
    save_checkpoint,
    set_backbone_trainable,
)
from src.preprocessing import DEFAULT_PREPROCESS, LEGACY_PREPROCESS


@pytest.mark.parametrize("name,arch", [("custom_cnn", "custom_cnn_v1"), ("custom_cnn", "custom_cnn_v2"),
                                       ("resnet18", None), ("mobilenet_v3", None)])
def test_builders_output_four_logits(name, arch):
    model = build_model(name, pretrained=False, arch=arch).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 224, 224)).shape == (2, len(CLASS_NAMES))
    assert gradcam_target_layer(model) is not None


def test_custom_cnn_v2_is_compact_and_uses_global_pooling():
    v1, v2 = BrainTumorCNN(), BrainTumorCNNv2()
    n1, n2 = (sum(p.numel() for p in m.parameters()) for m in (v1, v2))
    assert n2 < 2_000_000
    # v1 spends >70% of its parameters on one dense layer; v2's head is tiny
    assert v1.fc1.weight.numel() / n1 > 0.7
    assert sum(p.numel() for p in v2.head.parameters()) / n2 < 0.01


def test_parameter_groups_split_head_and_backbone():
    model = build_model("resnet18", pretrained=False)
    groups = {g["group_name"]: g for g in parameter_groups(model, "resnet18", lr=1e-3, backbone_lr_mult=0.1, weight_decay=0.05)}
    assert groups["head"]["lr"] == pytest.approx(1e-3)
    assert groups["backbone"]["lr"] == pytest.approx(1e-4)
    assert groups["backbone/no_decay"]["weight_decay"] == 0.0  # BN and biases are not decayed
    total = sum(p.numel() for g in groups.values() for p in g["params"])
    assert total == sum(p.numel() for p in model.parameters())


def test_freezing_backbone_leaves_only_head_trainable():
    model = build_model("mobilenet_v3", pretrained=False)
    set_backbone_trainable(model, "mobilenet_v3", False)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("classifier.") for n in trainable)


def test_checkpoint_roundtrip_with_metadata(tmp_path):
    model = BrainTumorCNNv2().eval()
    path = tmp_path / "ckpt.pth"
    import numpy as np
    save_checkpoint(path, model, "custom_cnn", "custom_cnn_v2", DEFAULT_PREPROCESS, temperature=1.7,
                    val_metrics={"val_acc": np.float64(0.9), "per_class": np.array([0.1, 0.2])}, epoch=3)
    # weights_only=True must accept everything we stored
    torch.load(path, map_location="cpu", weights_only=True)
    loaded, info = load_model_from_checkpoint("custom_cnn", path, torch.device("cpu"))
    assert info.arch == "custom_cnn_v2" and info.temperature == pytest.approx(1.7)
    assert info.preprocess == DEFAULT_PREPROCESS and not info.is_legacy
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x))


@pytest.mark.skipif(not (MODELS_DIR / "custom_cnn_final_model.pth").exists(), reason="shipped checkpoint not present")
def test_shipped_legacy_checkpoint_still_loads():
    _, info = read_checkpoint(MODELS_DIR / "custom_cnn_final_model.pth", "custom_cnn")
    assert info.arch == "custom_cnn_v1" and info.is_legacy
    assert info.preprocess == LEGACY_PREPROCESS and info.temperature == 1.0
    model, _ = load_model_from_checkpoint("custom_cnn", MODELS_DIR / "custom_cnn_final_model.pth", torch.device("cpu"))
    assert isinstance(model, BrainTumorCNN)


def test_class_order_mismatch_is_rejected(tmp_path):
    path = tmp_path / "bad.pth"
    torch.save({"model_state_dict": BrainTumorCNNv2().state_dict(), "class_names": list(reversed(CLASS_NAMES))}, path)
    with pytest.raises(ValueError, match="class order"):
        read_checkpoint(path, "custom_cnn")
