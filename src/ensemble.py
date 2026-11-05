"""Reliability-weighted ensembling and uncertainty assessment.

The original server averaged the four models with equal weight - including a
*randomly initialised* ViT whenever its checkpoint was missing - and reported
"N of 4 models agree" as if every vote were equally trustworthy.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .config import CLASS_METADATA, CLASS_NAMES

# Thresholds for presenting a result as confident. Below them the UI shows an
# "inconclusive - needs expert review" state instead of a diagnosis-like label.
HIGH_CONFIDENCE = 0.85
MODERATE_CONFIDENCE = 0.60
MIN_MARGIN = 0.20


def reliability_weight(
    val_metrics: Optional[Mapping[str, Any]], metrics_fallback: Optional[Mapping[str, Any]] = None
) -> float:
    """Log-odds of a model's accuracy: the optimal weight for weighted voting of
    independent classifiers (Nitzan & Paroush, 1982).

    Validation accuracy stored in the checkpoint is preferred; benchmark
    accuracy is only a fallback, and without either the weight is 1.
    """
    accuracy = None
    if val_metrics and val_metrics.get("val_acc") is not None:
        accuracy = float(val_metrics["val_acc"])
    elif metrics_fallback and metrics_fallback.get("test_accuracy_pct") is not None:
        accuracy = float(metrics_fallback["test_accuracy_pct"]) / 100.0
    if accuracy is None:
        return 1.0
    accuracy = min(max(accuracy, 0.55), 0.99)
    return math.log(accuracy / (1.0 - accuracy))


def weighted_average(prob_rows: Sequence[Sequence[float]], weights: Sequence[float]) -> np.ndarray:
    probs = np.asarray(prob_rows, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if probs.size == 0:
        return np.full(len(CLASS_NAMES), 1.0 / len(CLASS_NAMES))
    w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
    return w @ probs


def assess_uncertainty(
    ensemble_probs: Sequence[float],
    model_predictions: Sequence[int],
    weights: Sequence[float],
    class_names: Sequence[str] = CLASS_NAMES,
) -> Dict[str, Any]:
    """Summarise how much the result can be trusted."""
    probs = np.asarray(ensemble_probs, dtype=np.float64)
    order = np.argsort(probs)[::-1]
    top, second = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
    margin = float(probs[top] - probs[second])
    entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum() / math.log(len(probs)))

    w = np.asarray(weights, dtype=np.float64)
    votes = np.asarray(model_predictions)
    weighted_agreement = float(w[votes == top].sum() / w.sum()) if len(w) and w.sum() > 0 else 0.0
    agree_count = int((votes == top).sum())

    if probs[top] >= HIGH_CONFIDENCE and weighted_agreement >= 0.75 and margin >= MIN_MARGIN:
        level = "high"
    elif probs[top] >= MODERATE_CONFIDENCE and weighted_agreement >= 0.5 and margin >= MIN_MARGIN:
        level = "moderate"
    else:
        level = "low"

    def label(i: int) -> str:
        return CLASS_METADATA.get(class_names[i], {}).get("short_label", class_names[i])

    reasons: List[str] = []
    if probs[top] < MODERATE_CONFIDENCE:
        reasons.append(f"the top class has only {probs[top] * 100:.0f}% probability")
    if margin < MIN_MARGIN:
        reasons.append(f"'{label(top)}' and '{label(second)}' are close ({margin * 100:.0f} points apart)")
    if weighted_agreement < 0.5:
        reasons.append("the models disagree")

    return {
        "predicted_index": top,
        "predicted_class": class_names[top],
        "runner_up_class": class_names[second],
        "probability": float(probs[top]),
        "margin": round(margin, 4),
        "normalized_entropy": round(entropy, 4),
        "weighted_agreement": round(weighted_agreement, 4),
        "agreement_count": agree_count,
        "total_models": int(len(votes)),
        "confidence_level": level,
        "inconclusive": level == "low",
        "reasons": reasons,
    }
