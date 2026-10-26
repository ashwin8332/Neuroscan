import json
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import CLASS_NAMES, PLOTS_DIR, RESULTS_DIR

# Set clean scientific plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 15,
        "figure.dpi": 200,
    }
)


def load_metrics_data(results_dir: Path = RESULTS_DIR) -> Dict[str, Dict]:
    """Load model_comparison_metrics.json."""
    json_path = results_dir / "model_comparison_metrics.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found at: {json_path}. Please run 'python evaluate.py' first."
        )
    with open(json_path, "r") as f:
        return json.load(f)


def load_all_histories(results_dir: Path = RESULTS_DIR) -> Dict[str, Dict]:
    """Load all *_history.json files."""
    histories = {}
    for hist_file in results_dir.glob("*_history.json"):
        with open(hist_file, "r") as f:
            data = json.load(f)
            key = data.get("model_name", hist_file.stem.replace("_history", ""))
            histories[key] = data
    return histories


def plot_accuracy_and_error(metrics: Dict[str, Dict], save_path: Path) -> None:
    """Side-by-side comparison of Accuracy and Error Rate."""
    models = [data["display_name"] for data in metrics.values()]
    accuracies = [data["test_accuracy_pct"] for data in metrics.values()]
    errors = [data["error_rate_pct"] for data in metrics.values()]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Test Accuracy (%)", color="#2a9d8f")
    bars2 = ax.bar(x + width / 2, errors, width, label="Error Rate (%)", color="#e76f51")

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Brain Tumor Detection: Model Accuracy vs Error Rate Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(frameon=True)
    ax.set_ylim(0, 105)

    # Add text labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            color="#1b4d3e",
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            color="#8a2817",
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  -> Saved: {save_path}")


def plot_f1_and_auc(metrics: Dict[str, Dict], save_path: Path) -> None:
    """Grouped bar chart for Macro F1 and ROC-AUC."""
    models = [data["display_name"] for data in metrics.values()]
    f1_scores = [data["macro_f1"] for data in metrics.values()]
    auc_scores = [data["auc_macro"] for data in metrics.values()]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, f1_scores, width, label="Macro F1-Score", color="#457b9d")
    bars2 = ax.bar(x + width / 2, auc_scores, width, label="Multi-class ROC-AUC", color="#e9c46a")

    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.set_title("Macro F1-Score & ROC-AUC Across Model Architectures")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(frameon=True)
    ax.set_ylim(0, 1.1)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  -> Saved: {save_path}")


def plot_per_class_f1(metrics: Dict[str, Dict], save_path: Path) -> None:
    """Heatmap showing per-class F1-Score across all models."""
    models = [data["display_name"] for data in metrics.values()]
    data_matrix = []

    for key, data in metrics.items():
        row = [
            data.get("glioma_f1", 0.0),
            data.get("meningioma_f1", 0.0),
            data.get("notumor_f1", 0.0),
            data.get("pituitary_f1", 0.0),
        ]
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)

    fig, ax = plt.subplots(figsize=(9, 5 + len(models) * 0.4))
    sns.heatmap(
        data_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=[c.capitalize() for c in CLASS_NAMES],
        yticklabels=models,
        cbar_kws={"label": "F1-Score"},
        ax=ax,
        linewidths=1.0,
    )
    ax.set_title("Per-Class Tumor Detection F1-Score Breakdown")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  -> Saved: {save_path}")


def plot_confusion_matrices(metrics: Dict[str, Dict], save_path: Path) -> None:
    """Side-by-side confusion matrix subplots."""
    num_models = len(metrics)
    if num_models == 0:
        return

    cols = 2
    rows = (num_models + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    if num_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (key, data) in enumerate(metrics.items()):
        ax = axes[idx]
        cm = np.array(data["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[c[:4] for c in CLASS_NAMES],
            yticklabels=[c[:4] for c in CLASS_NAMES],
            cbar=False,
            ax=ax,
        )
        ax.set_title(f"{data['display_name']} (Acc: {data['test_accuracy_pct']}%)")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    for empty_idx in range(num_models, len(axes)):
        fig.delaxes(axes[empty_idx])

    plt.suptitle("Normalized Confusion Matrices Comparison", fontsize=15, y=0.99)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  -> Saved: {save_path}")


def plot_training_histories(histories: Dict[str, Dict], save_path: Path) -> None:
    """Plot convergence curves: Train & Val Loss, and Train & Val Accuracy."""
    if not histories:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for key, hist in histories.items():
        name = hist.get("display_name", key)
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax1.plot(epochs, hist["val_loss"], marker="o", label=f"{name} (Val)")
        ax2.plot(epochs, hist["val_acc"], marker="s", label=f"{name} (Val)")

    ax1.set_title("Validation Loss Curve (Cross-Entropy)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(frameon=True)

    ax2.set_title("Validation Accuracy Curve (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(frameon=True)

    plt.suptitle("Training Convergence Curves Across Architectures", fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  -> Saved: {save_path}")


def generate_all_comparison_visualizations(
    results_dir: Path = RESULTS_DIR,
    plots_dir: Path = PLOTS_DIR,
) -> None:
    """Generate the full comparison visual suite."""
    print(f"\n{'='*70}")
    print(f" GENERATING PUBLICATION-READY MODEL COMPARISON VISUALS")
    print(f"{'='*70}")

    metrics = load_metrics_data(results_dir)
    histories = load_all_histories(results_dir)

    plot_accuracy_and_error(metrics, plots_dir / "accuracy_and_error_comparison.png")
    plot_f1_and_auc(metrics, plots_dir / "f1_and_auc_comparison.png")
    plot_per_class_f1(metrics, plots_dir / "per_class_f1_heatmap.png")
    plot_confusion_matrices(metrics, plots_dir / "confusion_matrices_grid.png")

    if histories:
        plot_training_histories(histories, plots_dir / "training_loss_accuracy_curves.png")

    print(f"\nAll comparison figures successfully saved to: {plots_dir}\n")


if __name__ == "__main__":
    generate_all_comparison_visualizations()
