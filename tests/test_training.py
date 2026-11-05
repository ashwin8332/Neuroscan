"""End-to-end smoke test of the training engine on synthetic data (CPU, ~seconds)."""
import dataclasses
import json

import torch

from src.config import TRAIN_RECIPES
from src.data import build_data
from src.models import build_model, load_model_from_checkpoint
from src.preprocessing import PreprocessConfig
from src.training import _schedule, train_model


def test_schedule_warmup_then_cosine_and_delayed_backbone():
    assert _schedule(0, 0, 10, 100) < _schedule(9, 0, 10, 100) == 1.0
    assert _schedule(99, 0, 10, 100) < 0.05
    assert _schedule(5, 20, 10, 100) == 0.0      # frozen backbone gets no LR before unfreezing
    assert _schedule(20, 20, 10, 100) > 0.0


def test_train_model_writes_calibrated_checkpoint_and_history(synthetic_dataset, tmp_path):
    torch.set_num_threads(2)
    preprocess = PreprocessConfig(image_size=64)
    data = build_data(train_dir=synthetic_dataset / "Training", test_dir=None, batch_size=16,
                      num_workers=0, preprocess=preprocess, cache=True)
    recipe = dataclasses.replace(TRAIN_RECIPES["custom_cnn"], epochs=2, warmup_epochs=0.5, patience=5)
    model = build_model("custom_cnn", pretrained=False, arch="custom_cnn_v2")
    ckpt = tmp_path / "custom_cnn_final_model.pth"
    result = train_model(model, "custom_cnn", data, recipe, torch.device("cpu"), output_path=ckpt,
                         preprocess=preprocess, arch="custom_cnn_v2", results_dir=tmp_path)

    history = json.loads((tmp_path / "custom_cnn_history.json").read_text())
    assert len(history["train_loss"]) == 2 and len(history["val_loss"]) == 2
    assert history["split"]["grouped"] is True
    assert 1 <= result.best_epoch <= 2
    loaded, info = load_model_from_checkpoint("custom_cnn", ckpt, torch.device("cpu"))
    assert info.preprocess == preprocess and info.arch == "custom_cnn_v2"
    assert 1.0 <= info.temperature <= 10.0
    assert {"val_acc", "val_ece", "val_ece_calibrated", "best_epoch"} <= set(info.val_metrics)
    assert "interim" not in info.val_metrics  # the calibrated final checkpoint replaced the interim one
    with torch.no_grad():
        assert loaded(torch.randn(1, 3, 64, 64)).shape == (1, 4)
