import time
from typing import Any, Dict, List, Tuple
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

from .config import CLASS_NAMES


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    latency_ms_per_sample: float = 0.0,
) -> Dict[str, Any]:
    """
    Computes an extensive set of error, accuracy, discrimination, and efficiency metrics.
    
    Args:
        y_true: True class labels (shape: [N])
        y_pred: Predicted class labels (shape: [N])
        y_prob: Softmax probability distribution (shape: [N, num_classes])
        class_names: List of class strings
        latency_ms_per_sample: Inference latency in milliseconds per image
    """
    total_samples = len(y_true)
    num_classes = len(class_names)

    # 1. Error Metrics
    misclassified = int(np.sum(y_true != y_pred))
    error_rate_pct = (misclassified / total_samples) * 100.0

    # Cross-Entropy / Log Loss
    # Clip probabilities for numerical stability
    clipped_probs = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
    clipped_probs /= clipped_probs.sum(axis=1, keepdims=True)
    try:
        ce_loss = float(log_loss(y_true, clipped_probs, labels=list(range(num_classes))))
    except Exception:
        ce_loss = float("nan")

    # 2. Accuracy Metrics
    accuracy_pct = float(accuracy_score(y_true, y_pred) * 100.0)
    balanced_acc_pct = float(balanced_accuracy_score(y_true, y_pred) * 100.0)

    # 3. Precision, Recall, F1
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-Class Precision, Recall, F1
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0
    )

    per_class_metrics = {}
    for i, name in enumerate(class_names):
        per_class_metrics[name] = {
            "precision": float(p_class[i]),
            "recall": float(r_class[i]),
            "f1_score": float(f1_class[i]),
            "support": int(support_class[i]),
            "error_rate_pct": float((1.0 - r_class[i]) * 100.0),  # Class-specific miss rate
        }

    # 4. Multi-class ROC-AUC (One-vs-Rest)
    try:
        auc_macro = float(roc_auc_score(y_true, clipped_probs, multi_class="ovr", average="macro"))
    except Exception:
        auc_macro = float("nan")

    try:
        auc_weighted = float(roc_auc_score(y_true, clipped_probs, multi_class="ovr", average="weighted"))
    except Exception:
        auc_weighted = float("nan")

    # 5. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Classification report dict
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        # Overall Error Metrics
        "total_samples": total_samples,
        "misclassified_samples": misclassified,
        "error_rate_pct": round(error_rate_pct, 2),
        "cross_entropy_loss": round(ce_loss, 4),
        
        # Overall Accuracy & Score Metrics
        "test_accuracy_pct": round(accuracy_pct, 2),
        "balanced_accuracy_pct": round(balanced_acc_pct, 2),
        "macro_f1": round(float(f1_macro), 4),
        "macro_precision": round(float(precision_macro), 4),
        "macro_recall": round(float(recall_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "auc_macro": round(auc_macro, 4),
        "auc_weighted": round(auc_weighted, 4),
        
        # Per-Class Metrics
        "per_class": per_class_metrics,
        "glioma_f1": round(float(per_class_metrics.get("glioma", {}).get("f1_score", 0)), 4),
        "meningioma_f1": round(float(per_class_metrics.get("meningioma", {}).get("f1_score", 0)), 4),
        "notumor_f1": round(float(per_class_metrics.get("notumor", {}).get("f1_score", 0)), 4),
        "pituitary_f1": round(float(per_class_metrics.get("pituitary", {}).get("f1_score", 0)), 4),
        
        # Matrices & Reports
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        
        # Latency
        "latency_ms_per_image": round(latency_ms_per_sample, 2),
    }


from tqdm import tqdm


def evaluate_model_on_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str] = CLASS_NAMES,
    desc: str = "Evaluating",
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """
    Run full inference on DataLoader and return metrics dictionary + raw arrays.
    """
    model.eval()
    y_true_list = []
    y_pred_list = []
    y_prob_list = []

    start_time = time.time()
    total_images = 0

    pbar = tqdm(loader, desc=desc, unit="batch", leave=False)
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            y_true_list.extend(labels.cpu().numpy())
            y_pred_list.extend(preds.cpu().numpy())
            y_prob_list.extend(probs.cpu().numpy())
            total_images += images.size(0)

    total_time = time.time() - start_time
    latency_ms = (total_time / max(total_images, 1)) * 1000.0

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_prob = np.array(y_prob_list)

    metrics = compute_comprehensive_metrics(
        y_true, y_pred, y_prob, class_names=class_names, latency_ms_per_sample=latency_ms
    )

    return metrics, y_true, y_pred, y_prob
