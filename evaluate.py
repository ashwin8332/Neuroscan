import argparse
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    MODELS_DIR,
    RESULTS_DIR,
    TEST_DIR,
    get_device,
)
from src.data import get_image_folder
from src.metrics import evaluate_model_on_loader
from src.models import MODEL_REGISTRY, build_model, count_parameters, load_checkpoint


def evaluate_all_models(
    test_dir: Path = TEST_DIR,
    models_dir: Path = MODELS_DIR,
    batch_size: int = BATCH_SIZE,
    device: torch.device = None,
    target_model: str = None,
) -> Dict[str, Dict]:
    """
    Evaluate trained models on the test set and compile comparative results.
    """
    if device is None:
        device = get_device()

    test_dataset = get_image_folder(Path(test_dir), is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"\n{'='*90}")
    print(f" EVALUATING MODELS ON TEST DATASET")
    print(f" Test Set: {len(test_dataset)} images across {len(test_dataset.classes)} classes")
    print(f" Classes: {test_dataset.classes}")
    print(f" Device: {device}")
    print(f"{'='*90}\n")

    registry_keys = [target_model] if target_model else list(MODEL_REGISTRY.keys())
    results_summary = {}

    for key in registry_keys:
        spec = MODEL_REGISTRY[key]
        display_name = spec["display_name"]
        ckpt_path = Path(models_dir) / spec["checkpoint"]

        print(f"Evaluating {display_name}...")
        if not ckpt_path.exists():
            print(f"  [!] Checkpoint not found at: {ckpt_path}. Skipping.")
            continue

        model = build_model(key, num_classes=len(CLASS_NAMES), pretrained=False)
        model = load_checkpoint(model, ckpt_path, device)
        params_info = count_parameters(model)

        metrics, _, _, _ = evaluate_model_on_loader(
            model=model,
            loader=test_loader,
            device=device,
            class_names=CLASS_NAMES,
            desc=f"Evaluating {display_name}",
        )
        metrics["parameters_total"] = params_info["total"]
        metrics["parameters_trainable"] = params_info["trainable"]
        metrics["display_name"] = display_name

        results_summary[key] = metrics

    if not results_summary:
        print("[!] No evaluated models found. Please train models first using 'python train.py'.")
        return {}

    # Print Pretty Comparative Table
    print("\n" + "=" * 105)
    print(f"{'Model Architecture':<28} {'Accuracy':>10} {'Error Rate':>12} {'Loss':>8} {'Macro F1':>10} {'AUC':>8} {'Latency':>10}")
    print("=" * 105)

    table_rows = []
    for key, data in results_summary.items():
        row = {
            "Model": data["display_name"],
            "Test Accuracy (%)": data["test_accuracy_pct"],
            "Error Rate (%)": data["error_rate_pct"],
            "CE Loss": data["cross_entropy_loss"],
            "Macro F1": data["macro_f1"],
            "ROC-AUC": data["auc_macro"],
            "Glioma F1": data["glioma_f1"],
            "Meningioma F1": data["meningioma_f1"],
            "No Tumor F1": data["notumor_f1"],
            "Pituitary F1": data["pituitary_f1"],
            "Latency (ms/img)": data["latency_ms_per_image"],
            "Total Parameters": data["parameters_total"],
        }
        table_rows.append(row)
        print(
            f"{data['display_name']:<28} "
            f"{data['test_accuracy_pct']:>9.2f}% "
            f"{data['error_rate_pct']:>11.2f}% "
            f"{data['cross_entropy_loss']:>8.4f} "
            f"{data['macro_f1']:>10.4f} "
            f"{data['auc_macro']:>8.4f} "
            f"{data['latency_ms_per_image']:>8.2f} ms"
        )

    print("=" * 105)

    # Export to CSV and JSON
    df = pd.DataFrame(table_rows)
    csv_path = RESULTS_DIR / "model_comparison_metrics.csv"
    json_path = RESULTS_DIR / "model_comparison_metrics.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nComparative results saved to:")
    print(f"  - CSV:  {csv_path}")
    print(f"  - JSON: {json_path}")

    # Best model highlights
    best_acc = max(results_summary.keys(), key=lambda k: results_summary[k]["test_accuracy_pct"])
    lowest_err = min(results_summary.keys(), key=lambda k: results_summary[k]["error_rate_pct"])
    fastest = min(results_summary.keys(), key=lambda k: results_summary[k]["latency_ms_per_image"])

    print(f"\nSummary Highlights:")
    print(f"  * Highest Accuracy: {results_summary[best_acc]['display_name']} ({results_summary[best_acc]['test_accuracy_pct']}%)")
    print(f"  * Lowest Error:    {results_summary[lowest_err]['display_name']} ({results_summary[lowest_err]['error_rate_pct']}%)")
    print(f"  * Fastest Latency: {results_summary[fastest]['display_name']} ({results_summary[fastest]['latency_ms_per_image']} ms/image)")

    return results_summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate brain tumor detection models on the test set.")
    parser.add_argument("--test-dir", default=str(TEST_DIR))
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--model", default=None, choices=["custom_cnn", "resnet18", "mobilenet_v3", "vit", None])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else get_device()
    evaluate_all_models(
        test_dir=Path(args.test_dir),
        models_dir=Path(args.models_dir),
        batch_size=args.batch_size,
        device=device,
        target_model=args.model,
    )


if __name__ == "__main__":
    main()
