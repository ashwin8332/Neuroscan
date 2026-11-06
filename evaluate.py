"""Evaluate trained checkpoints on the held-out test set and export a comparison.

Each checkpoint is evaluated with the preprocessing and calibration temperature
stored inside it, so legacy and retrained models can be compared side by side.

    python evaluate.py                         # all checkpoints in models/
    python evaluate.py --tta                   # + horizontal-flip test-time augmentation
    python evaluate.py --exclude-leaked        # only test images with no near-duplicate in Training/
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.config import BATCH_SIZE, CLASS_NAMES, MODELS_DIR, RESULTS_DIR, TEST_DIR, TRAIN_DIR, ensure_output_dirs, get_device
from src.data import MRIDataset, compute_image_hashes, get_eval_tensorize, make_loader, near_duplicate_pairs, scan_image_folder
from src.ensemble import reliability_weight
from src.jsonutil import write_json
from src.metrics import collect_logits, compute_comprehensive_metrics, measure_latency
from src.models import MODEL_REGISTRY, count_parameters, load_model_from_checkpoint

logger = logging.getLogger("evaluate")


def leaked_test_mask(train_dir: Path, test_samples, hash_threshold: int) -> np.ndarray:
    """True for test images that have a near-duplicate in the training folder."""
    train_samples = scan_image_folder(train_dir)
    train_hashes = compute_image_hashes([p for p, _ in train_samples])
    test_hashes = compute_image_hashes([p for p, _ in test_samples])
    pairs = near_duplicate_pairs(test_hashes[0], test_hashes[1], hash_threshold, other=train_hashes[0])
    mask = np.zeros(len(test_samples), dtype=bool)
    for i, _ in pairs:
        mask[i] = True
    return mask


def evaluate_all_models(
    test_dir: Path = TEST_DIR,
    models_dir: Path = MODELS_DIR,
    batch_size: int = BATCH_SIZE,
    device: Optional[torch.device] = None,
    target_model: Optional[str] = None,
    tta: bool = False,
    train_dir: Optional[Path] = TRAIN_DIR,
    exclude_leaked: bool = False,
    hash_threshold: int = 24,
    results_dir: Path = RESULTS_DIR,
    num_workers: int = 0,
) -> Dict[str, Dict]:
    device = device or get_device()
    ensure_output_dirs()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    samples = scan_image_folder(Path(test_dir))
    meta: Dict = {"test_dir": str(test_dir), "tta": tta, "evaluated_at": dt.datetime.now().isoformat(timespec="seconds")}

    if train_dir and Path(train_dir).is_dir():
        mask = leaked_test_mask(Path(train_dir), samples, hash_threshold)
        labels = np.array([label for _, label in samples])
        meta["test_leakage"] = {"overall": round(float(mask.mean()), 4)} | {
            name: round(float(mask[labels == i].mean()), 4) for i, name in enumerate(CLASS_NAMES) if (labels == i).any()
        }
        logger.info("Test images with a near-duplicate in Training/: %.1f%% %s",
                    100 * mask.mean(), meta["test_leakage"])
        if exclude_leaked:
            samples = [s for s, leaked in zip(samples, mask) if not leaked]
            meta["excluded_leaked"] = int(mask.sum())
            logger.info("Evaluating on %d leakage-free test images only", len(samples))
    elif exclude_leaked:
        raise FileNotFoundError("--exclude-leaked needs the training folder (--train-dir)")
    meta["num_test_images"] = len(samples)
    y_true = np.array([label for _, label in samples])

    keys = [target_model] if target_model else list(MODEL_REGISTRY)
    loaders: Dict[tuple, torch.utils.data.DataLoader] = {}
    results: Dict[str, Dict] = {}
    ensemble_probs: List[np.ndarray] = []
    ensemble_weights: List[float] = []

    for key in keys:
        spec = MODEL_REGISTRY[key]
        ckpt_path = Path(models_dir) / spec["checkpoint"]
        if not ckpt_path.exists():
            logger.warning("%s: checkpoint %s not found - skipped", spec["display_name"], ckpt_path)
            continue
        model, info = load_model_from_checkpoint(key, ckpt_path, device)
        cfg = info.preprocess
        if cfg not in loaders:  # one cached dataset per preprocessing variant
            dataset = MRIDataset(samples, get_eval_tensorize(), cfg, cfg.image_size, cache=True)
            loaders[cfg] = make_loader(dataset, batch_size, False, num_workers)

        logits, labels = collect_logits(model, loaders[cfg], device, tta=tta, desc=f"Evaluating {spec['display_name']}")
        assert np.array_equal(labels.numpy(), y_true)
        probs_raw = F.softmax(logits, dim=1).numpy()
        probs = F.softmax(logits / info.temperature, dim=1).numpy()
        latency = measure_latency(model, device, cfg.image_size)

        metrics = compute_comprehensive_metrics(y_true, probs.argmax(1), probs, CLASS_NAMES, latency)
        uncalibrated = compute_comprehensive_metrics(y_true, probs_raw.argmax(1), probs_raw, CLASS_NAMES)
        params = count_parameters(model)
        metrics.update({
            "display_name": spec["display_name"],
            "parameters_total": params["total"],
            "parameters_trainable": params["trainable"],
            "arch": info.arch,
            "checkpoint_format": info.format_version,
            "temperature": round(info.temperature, 4),
            "ece_uncalibrated": uncalibrated["ece"],
            "cross_entropy_loss_uncalibrated": uncalibrated["cross_entropy_loss"],
            "preprocess": cfg.to_dict(),
            "val_metrics": info.val_metrics,
            "tta": tta,
        })
        results[key] = metrics
        ensemble_probs.append(probs)
        ensemble_weights.append(reliability_weight(info.val_metrics, metrics_fallback=None))

    if not results:
        logger.error("No checkpoints found in %s. Train models first: python train.py --model all", models_dir)
        return {}

    if len(ensemble_probs) > 1:
        weights = np.array(ensemble_weights) / np.sum(ensemble_weights)
        probs = np.tensordot(weights, np.stack(ensemble_probs), axes=1)
        ens = compute_comprehensive_metrics(y_true, probs.argmax(1), probs, CLASS_NAMES,
                                            sum(r["latency_ms_per_image"] for r in results.values()))
        ens.update({
            "display_name": "Weighted Ensemble",
            "parameters_total": sum(r["parameters_total"] for r in results.values()),
            "parameters_trainable": 0,
            "arch": "ensemble",
            "ensemble_weights": {k: round(float(w), 4) for k, w in zip(results, weights)},
            "tta": tta,
        })
        results["ensemble"] = ens

    _print_table(results)
    rows = [{
        "Model": m["display_name"],
        "Test Accuracy (%)": m["test_accuracy_pct"],
        "Balanced Acc (%)": m["balanced_accuracy_pct"],
        "Error Rate (%)": m["error_rate_pct"],
        "CE Loss": m["cross_entropy_loss"],
        "ECE": m["ece"],
        "Macro F1": m["macro_f1"],
        "ROC-AUC": m["auc_macro"],
        "Tumor Sensitivity": m.get("tumor_sensitivity"),
        "Tumor Specificity": m.get("tumor_specificity"),
        "Glioma F1": m["glioma_f1"],
        "Meningioma F1": m["meningioma_f1"],
        "No Tumor F1": m["notumor_f1"],
        "Pituitary F1": m["pituitary_f1"],
        "Latency (ms/img)": m["latency_ms_per_image"],
        "Total Parameters": m["parameters_total"],
    } for m in results.values()]
    pd.DataFrame(rows).to_csv(results_dir / "model_comparison_metrics.csv", index=False)
    write_json(results_dir / "model_comparison_metrics.json", results)
    write_json(results_dir / "evaluation_meta.json", meta)
    logger.info("Saved %s, %s and %s", results_dir / "model_comparison_metrics.csv",
                results_dir / "model_comparison_metrics.json", results_dir / "evaluation_meta.json")
    return results


def _print_table(results: Dict[str, Dict]) -> None:
    header = f"{'Model':<32}{'Acc %':>8}{'BalAcc':>8}{'Loss':>8}{'ECE':>7}{'MacroF1':>9}{'AUC':>7}{'Sens':>7}{'Spec':>7}{'ms':>8}"
    print("\n" + header + "\n" + "-" * len(header))
    for m in results.values():
        print(f"{m['display_name'][:31]:<32}{m['test_accuracy_pct']:>8.2f}{m['balanced_accuracy_pct']:>8.2f}"
              f"{m['cross_entropy_loss']:>8.4f}{m['ece']:>7.3f}{m['macro_f1']:>9.4f}{m['auc_macro']:>7.3f}"
              f"{m.get('tumor_sensitivity', float('nan')):>7.3f}{m.get('tumor_specificity', float('nan')):>7.3f}"
              f"{m['latency_ms_per_image']:>8.1f}")
    print()


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Evaluate brain tumor models on the test set.")
    p.add_argument("--data-dir", type=Path, default=None, help="Dataset root with Training/ and Testing/")
    p.add_argument("--test-dir", type=Path, default=TEST_DIR)
    p.add_argument("--train-dir", type=Path, default=TRAIN_DIR, help="Used for the train/test leakage audit")
    p.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--model", default=None, choices=list(MODEL_REGISTRY))
    p.add_argument("--tta", action="store_true", help="Average predictions with the mirrored image")
    p.add_argument("--exclude-leaked", action="store_true", help="Drop test images duplicated in Training/")
    p.add_argument("--hash-threshold", type=int, default=24)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args(argv)

    test_dir, train_dir = args.test_dir, args.train_dir
    if args.data_dir:
        test_dir, train_dir = args.data_dir / "Testing", args.data_dir / "Training"
    if not Path(test_dir).is_dir():
        logger.error("Test folder not found: %s", test_dir)
        return 2
    results = evaluate_all_models(
        test_dir=test_dir, models_dir=args.models_dir, batch_size=args.batch_size,
        device=torch.device("cpu") if args.cpu else get_device(), target_model=args.model,
        tta=args.tta, train_dir=train_dir, exclude_leaked=args.exclude_leaked,
        hash_threshold=args.hash_threshold, results_dir=args.results_dir, num_workers=args.workers,
    )
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
