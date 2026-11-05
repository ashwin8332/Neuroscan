"""Publication-style comparison figures from results/*.json.

    python compare.py            # writes PNGs to results/plots/
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")  # headless: works on servers and CI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from src.config import CLASS_NAMES, PLOTS_DIR, RESULTS_DIR, ensure_output_dirs
from src.models import MODEL_REGISTRY

CLASS_COLORS = {"glioma": "#d9485f", "meningioma": "#e0912a", "notumor": "#2fa47c", "pituitary": "#3f7fd9"}
CLASS_LABELS = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "No tumor", "pituitary": "Pituitary"}

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 12.5, "axes.titleweight": "bold", "figure.dpi": 150,
                     "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})


def load_metrics_data(results_dir: Path = RESULTS_DIR) -> Dict[str, Dict]:
    path = Path(results_dir) / "model_comparison_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 'python evaluate.py' first.")
    with open(path) as f:
        return json.load(f)


def load_all_histories(results_dir: Path = RESULTS_DIR) -> Dict[str, Dict]:
    histories = {}
    for path in sorted(Path(results_dir).glob("*_history.json")):
        with open(path) as f:
            data = json.load(f)
        histories[data.get("model_name", path.stem.replace("_history", ""))] = data
    return histories


def _names(metrics: Dict[str, Dict]):
    return [m["display_name"].replace(" (", "\n(") for m in metrics.values()]


def _bar_labels(ax, bars, fmt):
    for bar in bars:
        h = bar.get_height()
        if np.isfinite(h):
            ax.annotate(fmt.format(h), (bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.5)


def _grouped(ax, metrics, series, ylim, fmt):
    x = np.arange(len(metrics))
    width = 0.8 / len(series)
    for i, (label, values, color) in enumerate(series):
        bars = ax.bar(x + (i - (len(series) - 1) / 2) * width, values, width, label=label, color=color)
        _bar_labels(ax, bars, fmt)
    ax.set_xticks(x)
    ax.set_xticklabels(_names(metrics), fontsize=9)
    ax.set_ylim(*ylim)
    ax.legend(frameon=False, ncol=len(series), loc="upper center", bbox_to_anchor=(0.5, -0.16))


def screening_from_cm(cm) -> Dict[str, float]:
    cm = np.asarray(cm)
    n = CLASS_NAMES.index("notumor")
    tumor_rows = [i for i in range(len(CLASS_NAMES)) if i != n]
    tumors = cm[tumor_rows].sum()
    missed = cm[tumor_rows, n].sum()
    normals = cm[n].sum()
    return {"tumor_sensitivity": 1 - missed / tumors if tumors else np.nan,
            "tumor_specificity": cm[n, n] / normals if normals else np.nan}


def plot_accuracy_and_error(metrics: Dict[str, Dict], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    _grouped(ax, metrics, [
        ("Test accuracy (%)", [m["test_accuracy_pct"] for m in metrics.values()], "#2b2d42"),
        ("Error rate (%)", [m["error_rate_pct"] for m in metrics.values()], "#d9485f"),
    ], (0, 108), "{:.1f}")
    ax.set_title("Test accuracy vs. error rate")
    ax.set_ylabel("%")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  -> {save_path}")


def plot_f1_and_auc(metrics: Dict[str, Dict], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    _grouped(ax, metrics, [
        ("Macro F1", [m["macro_f1"] for m in metrics.values()], "#2b2d42"),
        ("ROC-AUC (one-vs-rest)", [m["auc_macro"] for m in metrics.values()], "#8d99ae"),
    ], (0, 1.1), "{:.3f}")
    ax.set_title("Macro F1 and ROC-AUC")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  -> {save_path}")


def plot_per_class_f1(metrics: Dict[str, Dict], save_path: Path) -> None:
    matrix = np.array([[m.get(f"{c}_f1", np.nan) for c in CLASS_NAMES] for m in metrics.values()])
    fig, ax = plt.subplots(figsize=(8.5, 1.2 + 0.7 * len(metrics)))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="mako", vmin=0.5, vmax=1.0, linewidths=1,
                xticklabels=[CLASS_LABELS[c] for c in CLASS_NAMES],
                yticklabels=[m["display_name"] for m in metrics.values()], cbar_kws={"label": "F1"}, ax=ax)
    ax.set_title("Per-class F1 score")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  -> {save_path}")


def plot_confusion_matrices(metrics: Dict[str, Dict], save_path: Path) -> None:
    """Row-normalised (recall) matrices, annotated with % and raw counts."""
    n = len(metrics)
    if not n:
        return
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.6 * rows), squeeze=False)
    labels = [CLASS_LABELS[c] for c in CLASS_NAMES]
    for ax, m in zip(axes.flat, metrics.values()):
        cm = np.asarray(m["confusion_matrix"], dtype=float)
        norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        annot = np.array([[f"{p:.0%}\n{int(c)}" for p, c in zip(prow, crow)] for prow, crow in zip(norm, cm)])
        sns.heatmap(norm, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False,
                    xticklabels=labels, yticklabels=labels, ax=ax, annot_kws={"fontsize": 9})
        ax.set_title(f"{m['display_name']}\nacc {m['test_accuracy_pct']:.1f}%", fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Confusion matrices (row-normalised: each row sums to 100%)", y=1.0)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  -> {save_path}")


def plot_training_histories(histories: Dict[str, Dict], save_path: Path, metrics: Dict[str, Dict] = None) -> None:
    """Train vs. validation curves per model - the over/under-fitting diagnostic.

    A dashed line marks held-out test accuracy when available: a large drop
    from validation to test means the validation split was not independent.
    """
    if not histories:
        return
    n = len(histories)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.3 * n), squeeze=False)
    for (key, h), (ax_loss, ax_acc) in zip(histories.items(), axes):
        epochs = np.arange(1, len(h["train_loss"]) + 1)
        name = h.get("display_name") if h.get("display_name") not in (None, key) else MODEL_REGISTRY.get(key, {}).get("display_name", key)
        ax_loss.plot(epochs, h["train_loss"], "o-", ms=3, color="#2b2d42", label="train")
        ax_loss.plot(epochs, h["val_loss"], "o-", ms=3, color="#d9485f", label="validation")
        ax_loss.set_title(f"{name} - loss")
        ax_loss.set_xlabel("epoch")
        ax_acc.plot(epochs, h["train_acc"], "o-", ms=3, color="#2b2d42", label="train")
        ax_acc.plot(epochs, h["val_acc"], "o-", ms=3, color="#d9485f", label="validation")
        if metrics and key in metrics:
            ax_acc.axhline(metrics[key]["test_accuracy_pct"], ls="--", color="#3f7fd9", label="test (held-out)")
        if h.get("best_epoch"):
            for ax in (ax_loss, ax_acc):
                ax.axvline(h["best_epoch"], color="#8d99ae", lw=1, ls=":")
        ax_acc.set_title(f"{name} - accuracy (%)")
        ax_acc.set_xlabel("epoch")
        ax_acc.legend(frameon=False, fontsize=8.5, loc="lower right")
        ax_loss.legend(frameon=False, fontsize=8.5)
        for ax in (ax_loss, ax_acc):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle("Training vs. validation: gaps reveal over-fitting, low plateaus reveal under-fitting", y=1.0)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  -> {save_path}")


def plot_screening_and_calibration(metrics: Dict[str, Dict], save_path: Path) -> None:
    rows = []
    for m in metrics.values():
        screen = {k: m[k] for k in ("tumor_sensitivity", "tumor_specificity") if m.get(k) is not None} or screening_from_cm(m["confusion_matrix"])
        rows.append((screen["tumor_sensitivity"] * 100, screen["tumor_specificity"] * 100, m.get("ece", np.nan)))
    rows = np.array(rows, dtype=float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    _grouped(ax1, metrics, [("Tumor sensitivity (%)", rows[:, 0], "#2b2d42"),
                            ("Specificity (%)", rows[:, 1], "#2fa47c")], (0, 108), "{:.1f}")
    ax1.set_title("Screening view: tumor vs. no tumor")
    if np.isfinite(rows[:, 2]).any():
        _grouped(ax2, metrics, [("Expected calibration error", rows[:, 2], "#e0912a")],
                 (0, max(0.05, np.nanmax(rows[:, 2]) * 1.3)), "{:.3f}")
        ax2.set_title("Calibration (lower is better)")
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "ECE not in this metrics file -\nre-run evaluate.py", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  -> {save_path}")


def generate_all_comparison_visualizations(results_dir: Path = RESULTS_DIR, plots_dir: Path = PLOTS_DIR) -> None:
    ensure_output_dirs()
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics_data(results_dir)
    histories = load_all_histories(results_dir)
    print(f"Generating comparison figures in {plots_dir}")
    plot_accuracy_and_error(metrics, plots_dir / "accuracy_and_error_comparison.png")
    plot_f1_and_auc(metrics, plots_dir / "f1_and_auc_comparison.png")
    plot_per_class_f1(metrics, plots_dir / "per_class_f1_heatmap.png")
    plot_confusion_matrices(metrics, plots_dir / "confusion_matrices_grid.png")
    plot_screening_and_calibration(metrics, plots_dir / "screening_and_calibration.png")
    if histories:
        plot_training_histories(histories, plots_dir / "training_loss_accuracy_curves.png", metrics)


if __name__ == "__main__":
    generate_all_comparison_visualizations()
