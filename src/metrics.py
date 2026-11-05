"""Evaluation metrics: accuracy, error, calibration and clinical screening view."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm

from .config import CLASS_NAMES, CLASS_TO_IDX


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """ECE: average |confidence - accuracy| over equal-width confidence bins.

    0 means the displayed confidence can be taken at face value; the original
    models showed ~100% confidence even on misclassified scans.
    """
    confidences = y_prob.max(axis=1)
    correct = (y_prob.argmax(axis=1) == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > low) & (confidences <= high)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidences[in_bin].mean())
    return float(ece)


def screening_metrics(y_true: np.ndarray, y_pred: np.ndarray, normal_class: str = "notumor") -> Dict[str, float]:
    """Binary tumor-vs-no-tumor view - the question a patient actually asks.

    Sensitivity = share of tumor scans flagged as *some* tumor (missing a tumor
    is the costly error); specificity = share of normal scans called normal.
    """
    normal_idx = CLASS_TO_IDX[normal_class]
    is_tumor = y_true != normal_idx
    flagged = y_pred != normal_idx
    sensitivity = float(flagged[is_tumor].mean()) if is_tumor.any() else float("nan")
    specificity = float((~flagged[~is_tumor]).mean()) if (~is_tumor).any() else float("nan")
    return {
        "tumor_sensitivity": round(sensitivity, 4),
        "tumor_specificity": round(specificity, 4),
        "missed_tumors": int((is_tumor & ~flagged).sum()),
        "false_alarms": int((~is_tumor & flagged).sum()),
    }


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    latency_ms_per_sample: float = 0.0,
) -> Dict[str, Any]:
    """Error, accuracy, discrimination, calibration and efficiency metrics."""
    total_samples = len(y_true)
    num_classes = len(class_names)
    labels = list(range(num_classes))

    misclassified = int(np.sum(y_true != y_pred))
    error_rate_pct = (misclassified / max(total_samples, 1)) * 100.0

    clipped = np.clip(y_prob, 1e-7, 1.0)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    try:
        ce_loss = float(log_loss(y_true, clipped, labels=labels))
    except ValueError:
        ce_loss = float("nan")

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = {
        name: {
            "precision": float(p_class[i]),
            "recall": float(r_class[i]),
            "f1_score": float(f1_class[i]),
            "support": int(support_class[i]),
            "error_rate_pct": float((1.0 - r_class[i]) * 100.0),
        }
        for i, name in enumerate(class_names)
    }

    try:
        auc_macro = float(roc_auc_score(y_true, clipped, multi_class="ovr", average="macro", labels=labels))
        auc_weighted = float(roc_auc_score(y_true, clipped, multi_class="ovr", average="weighted", labels=labels))
    except ValueError:  # a class is absent from y_true (e.g. tiny or filtered test sets)
        auc_macro = auc_weighted = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0
    )

    metrics: Dict[str, Any] = {
        "total_samples": total_samples,
        "misclassified_samples": misclassified,
        "error_rate_pct": round(error_rate_pct, 2),
        "cross_entropy_loss": round(ce_loss, 4),
        "test_accuracy_pct": round(float(accuracy_score(y_true, y_pred) * 100.0), 2),
        "balanced_accuracy_pct": round(float(balanced_accuracy_score(y_true, y_pred) * 100.0), 2),
        "macro_f1": round(float(f1_macro), 4),
        "macro_precision": round(float(precision_macro), 4),
        "macro_recall": round(float(recall_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "auc_macro": round(auc_macro, 4),
        "auc_weighted": round(auc_weighted, 4),
        "ece": round(expected_calibration_error(y_true, clipped), 4),
        "mean_confidence": round(float(clipped.max(axis=1).mean()), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "latency_ms_per_image": round(latency_ms_per_sample, 2),
    }
    for name in class_names:
        metrics[f"{name}_f1"] = round(per_class[name]["f1_score"], 4)
    if "notumor" in class_names:
        metrics.update(screening_metrics(y_true, y_pred))
    return metrics


@torch.no_grad()
def collect_logits(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    tta: bool = False,
    desc: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the model over a loader and return (logits, labels) on CPU.

    With ``tta=True`` logits are averaged with those of the mirrored image
    (left/right flips are anatomically plausible).
    """
    model.eval()
    all_logits, all_labels = [], []
    iterator = tqdm(loader, desc=desc, unit="batch", leave=False, disable=None) if desc else loader
    for images, labels in iterator:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        if tta:
            logits = 0.5 * (logits + model(torch.flip(images, dims=[3])))
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


@torch.no_grad()
def measure_latency(
    model: torch.nn.Module,
    device: torch.device,
    image_size: int = 224,
    runs: int = 30,
    warmup: int = 5,
) -> float:
    """Median single-image forward latency in milliseconds (model only, no I/O)."""
    model.eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)
    for _ in range(warmup):
        model(x)
    timings = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(timings))


def evaluate_model_on_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str] = CLASS_NAMES,
    desc: str = "Evaluating",
    temperature: float = 1.0,
    tta: bool = False,
    measure_speed: bool = True,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Full inference over a loader -> (metrics, y_true, y_pred, y_prob)."""
    logits, labels = collect_logits(model, loader, device, tta=tta, desc=desc)
    probs = F.softmax(logits / temperature, dim=1).numpy()
    y_true = labels.numpy()
    y_pred = probs.argmax(axis=1)
    latency = measure_latency(model, device) if measure_speed else 0.0
    metrics = compute_comprehensive_metrics(y_true, y_pred, probs, class_names, latency)
    return metrics, y_true, y_pred, probs
