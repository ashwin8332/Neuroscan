"""Train brain-tumor MRI classifiers with the leak-free, regularised recipe.

Examples
--------
    python train.py --model resnet18                       # recommended recipe
    python train.py --model all --data-dir /path/to/brain-tumor-mri-dataset
    python train.py --model custom_cnn --epochs 5 --subset-fraction 0.1   # smoke test
    python train.py --model resnet18 --pretrained-weights resnet18.pth     # offline
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
from pathlib import Path

import torch

from src.config import (
    MODELS_DIR,
    RESULTS_DIR,
    SEED,
    TRAIN_DIR,
    TRAIN_RECIPES,
    ensure_output_dirs,
    get_device,
)
from src.data import build_data
from src.models import MODEL_REGISTRY, build_model, count_parameters, default_arch
from src.preprocessing import PreprocessConfig
from src.training import set_seed, train_model

logger = logging.getLogger("train")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train brain tumor MRI classification models.")
    p.add_argument("--model", default="custom_cnn", choices=[*MODEL_REGISTRY, "all"])
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Dataset root containing Training/ (overrides --train-dir)")
    p.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    p.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)

    recipe = p.add_argument_group("recipe overrides (defaults come from src/config.py TRAIN_RECIPES)")
    recipe.add_argument("--epochs", type=int)
    recipe.add_argument("--lr", type=float, help="Head learning rate")
    recipe.add_argument("--backbone-lr-mult", type=float)
    recipe.add_argument("--batch-size", type=int)
    recipe.add_argument("--weight-decay", type=float)
    recipe.add_argument("--label-smoothing", type=float)
    recipe.add_argument("--warmup-epochs", type=float)
    recipe.add_argument("--freeze-epochs", type=int, help="Linear-probe epochs before full fine-tuning")
    recipe.add_argument("--patience", type=int)
    recipe.add_argument("--freeze-backbone", action="store_true",
                        help="Keep the backbone frozen for the whole run (pure linear probe)")

    data = p.add_argument_group("data")
    data.add_argument("--val-ratio", type=float, default=0.15)
    data.add_argument("--subset-fraction", type=float, default=1.0, help="Stratified subsample for quick runs")
    data.add_argument("--image-size", type=int, default=224)
    data.add_argument("--no-crop", action="store_true", help="Disable brain cropping / square padding")
    data.add_argument("--random-split", action="store_true",
                      help="Plain stratified split without duplicate grouping (leaky, not recommended)")
    data.add_argument("--hash-threshold", type=int, default=24)
    data.add_argument("--no-cache", action="store_true", help="Do not cache preprocessed images in RAM")
    data.add_argument("--workers", type=int, default=None)

    p.add_argument("--no-ema", action="store_true", help="Disable weight EMA")
    p.add_argument("--pretrained-weights", type=Path, default=None,
                   help="Local ImageNet weights for the backbone (offline / air-gapped training)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--cpu", action="store_true", help="Force CPU training")
    p.add_argument("--threads", type=int, default=None, help="CPU threads for PyTorch (default: min(cores, 8))")
    return p.parse_args(argv)


def resolve_recipe(model_key: str, args: argparse.Namespace):
    recipe = TRAIN_RECIPES[model_key]
    overrides = {
        field: getattr(args, field)
        for field in ("epochs", "lr", "backbone_lr_mult", "batch_size", "weight_decay",
                      "label_smoothing", "warmup_epochs", "freeze_epochs", "patience")
        if getattr(args, field) is not None
    }
    recipe = dataclasses.replace(recipe, **overrides)
    if args.freeze_backbone and model_key != "custom_cnn":
        recipe = dataclasses.replace(recipe, freeze_epochs=recipe.epochs)
    return recipe


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    ensure_output_dirs()
    if args.threads or not torch.cuda.is_available():
        torch.set_num_threads(args.threads or min(os.cpu_count() or 4, 8))

    train_dir = args.data_dir / "Training" if args.data_dir else args.train_dir
    if not Path(train_dir).is_dir():
        logger.error("Training folder not found: %s (use --data-dir or set NEUROSCAN_DATA_DIR)", train_dir)
        return 2

    device = torch.device("cpu") if args.cpu else get_device()
    targets = list(MODEL_REGISTRY) if args.model == "all" else [args.model]
    preprocess = PreprocessConfig(image_size=args.image_size, crop_brain=not args.no_crop,
                                  pad_to_square=not args.no_crop)

    set_seed(args.seed)
    bundles = {}
    for model_key in targets:
        recipe = resolve_recipe(model_key, args)
        if recipe.batch_size not in bundles:
            logger.info("Building data (batch size %d, grouped split=%s)...", recipe.batch_size, not args.random_split)
            bundles[recipe.batch_size] = build_data(
                train_dir=train_dir, test_dir=None, batch_size=recipe.batch_size,
                val_ratio=args.val_ratio, subset_fraction=args.subset_fraction,
                num_workers=args.workers, seed=args.seed, preprocess=preprocess,
                grouped_split=not args.random_split, hash_threshold=args.hash_threshold,
                cache=not args.no_cache,
            )
            report = bundles[recipe.batch_size].split_report
            logger.info("Split: %d train / %d val | %d near-duplicate images in %d groups (largest %d)",
                        report.num_train, report.num_val, report.duplicate_images,
                        report.num_groups, report.largest_group)
        data = bundles[recipe.batch_size]

        spec = MODEL_REGISTRY[model_key]
        arch = default_arch(model_key)
        model = build_model(model_key, num_classes=len(data.class_names), pretrained=spec["pretrained"],
                            arch=arch, weights_path=args.pretrained_weights)
        logger.info("=" * 72)
        logger.info("%s | %s | device=%s | params=%s", spec["display_name"], arch, device,
                    f"{count_parameters(model)['total']:,}")
        logger.info("Recipe: %s", recipe)
        result = train_model(
            model, model_key, data, recipe, device,
            output_path=args.output_dir / spec["checkpoint"], preprocess=preprocess, arch=arch,
            results_dir=args.results_dir, seed=args.seed, use_ema=not args.no_ema,
            extra_config={"pretrained_weights": str(args.pretrained_weights) if args.pretrained_weights else "torchvision"},
        )
        vm = result.val_metrics
        logger.info("%s done: best epoch %d, val acc %.2f%%, val F1 %.4f, T=%.3f, ECE %.4f -> %.4f",
                    spec["display_name"], result.best_epoch, vm["val_acc"] * 100, vm["val_macro_f1"],
                    result.temperature, vm["val_ece"], vm["val_ece_calibrated"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
