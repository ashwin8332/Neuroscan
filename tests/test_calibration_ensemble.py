import numpy as np
import pytest
import torch

from src.calibration import fit_temperature, nll
from src.ensemble import assess_uncertainty, reliability_weight, weighted_average
from src.metrics import compute_comprehensive_metrics, expected_calibration_error, screening_metrics


def test_temperature_scaling_fixes_overconfidence():
    torch.manual_seed(0)
    n, k = 4000, 4
    labels = torch.randint(0, k, (n,))
    clean = torch.randn(n, k)
    clean[torch.arange(n), labels] += 1.5   # informative but noisy logits
    overconfident = clean * 4.0              # same ranking, far too confident
    t = fit_temperature(overconfident, labels)
    assert 2.5 < t < 6.0
    assert nll(overconfident, labels, t) < nll(overconfident, labels)
    probs = torch.softmax(overconfident, 1).numpy()
    probs_cal = torch.softmax(overconfident / t, 1).numpy()
    assert expected_calibration_error(labels.numpy(), probs_cal) < expected_calibration_error(labels.numpy(), probs)


def test_temperature_never_sharpens():
    """Under-confident (or separable) validation logits would ask for T < 1; policy clamps to 1."""
    labels = torch.tensor([0, 1, 2, 3] * 10)
    logits = torch.nn.functional.one_hot(labels, 4).float() * 2
    assert fit_temperature(logits, labels) == pytest.approx(1.0)


def test_reliability_weight_orders_models_and_prefers_validation():
    weak, strong = reliability_weight({"val_acc": 0.80}), reliability_weight({"val_acc": 0.97})
    assert strong > weak > 0
    assert reliability_weight({}, {"test_accuracy_pct": 95.0}) == pytest.approx(np.log(0.95 / 0.05))
    assert reliability_weight(None) == 1.0


def test_weighted_average_is_a_distribution():
    avg = weighted_average([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]], [3.0, 1.0])
    assert avg.sum() == pytest.approx(1.0) and avg[0] == pytest.approx(0.55)


def test_uncertainty_levels():
    confident = assess_uncertainty([0.92, 0.04, 0.02, 0.02], [0, 0, 0], [1, 1, 1])
    assert confident["confidence_level"] == "high" and not confident["inconclusive"]
    # the real ensemble output for sample glioma/Te-gl_1.jpg with the shipped checkpoints
    split = assess_uncertainty([0.291, 0.175, 0.388, 0.147], [2, 0, 2], [2.1, 4.3, 3.6])
    assert split["predicted_class"] == "notumor" and split["inconclusive"]
    assert len(split["reasons"]) == 2  # low top probability + close runner-up
    assert split["agreement_count"] == 2 and split["total_models"] == 3


def test_metrics_include_screening_and_calibration():
    y_true = np.array([0, 0, 1, 2, 2, 3])
    y_pred = np.array([0, 2, 1, 2, 2, 3])  # one glioma missed as "no tumor"
    probs = np.eye(4)[y_pred] * 0.9 + 0.025
    m = compute_comprehensive_metrics(y_true, y_pred, probs)
    assert m["missed_tumors"] == 1 and m["false_alarms"] == 0
    assert m["tumor_sensitivity"] == pytest.approx(0.75) and m["tumor_specificity"] == 1.0
    assert 0 <= m["ece"] <= 1 and m["glioma_f1"] < 1.0
    assert screening_metrics(y_true, y_true)["tumor_sensitivity"] == 1.0
