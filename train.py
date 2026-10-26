import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    MODELS_DIR,
    RESULTS_DIR,
    SEED,
    TRAIN_DIR,
    get_device,
)
from src.data import get_data_loaders
from src.models import MODEL_REGISTRY, build_model, count_parameters


import os
from tqdm import tqdm

# Optimize CPU thread allocation for fast training on multicore systems
if not torch.cuda.is_available():
    torch.set_num_threads(min(os.cpu_count() or 4, 8))


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 1,
    total_epochs: int = 1,
) -> Tuple[float, float, float]:
    """Train for one epoch with live progress bar. Returns (avg_loss, accuracy, error_rate)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch:02d}/{total_epochs:02d} [Train]", unit="batch", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        batch_correct = (preds == labels).sum().item()
        correct += batch_correct
        total += batch_size

        running_loss = total_loss / max(total, 1)
        running_acc = (correct / max(total, 1)) * 100.0
        pbar.set_postfix({"loss": f"{running_loss:.4f}", "acc": f"{running_acc:.1f}%"})

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    error_rate = 1.0 - acc
    return avg_loss, acc, error_rate


def validate_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 1,
    total_epochs: int = 1,
) -> Tuple[float, float, float]:
    """Evaluate on validation set with live progress bar. Returns (avg_loss, accuracy, error_rate)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch:02d}/{total_epochs:02d} [Val]  ", unit="batch", leave=False)
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

            running_loss = total_loss / max(total, 1)
            running_acc = (correct / max(total, 1)) * 100.0
            pbar.set_postfix({"loss": f"{running_loss:.4f}", "acc": f"{running_acc:.1f}%"})

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    error_rate = 1.0 - acc
    return avg_loss, acc, error_rate


def train_model(
    model_name: str,
    train_dir: Path = TRAIN_DIR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = DEFAULT_LR,
    val_ratio: float = 0.15,
    subset_fraction: float = 1.0,
    freeze_backbone: bool = False,
    patience: int = 5,
    output_path: Path = None,
    device: torch.device = None,
    seed: int = SEED,
) -> Dict:
    """
    Train and validate a specified model architecture.
    """
    set_seed(seed)
    if device is None:
        device = get_device()

    normalized_name = model_name.lower().replace("-", "_")
    spec = MODEL_REGISTRY[normalized_name]
    display_name = spec["display_name"]

    if output_path is None:
        output_path = MODELS_DIR / spec["checkpoint"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f" Training Model: {display_name} ({normalized_name})")
    print(f" Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print(f"{'='*70}")

    # Data loaders
    train_loader, val_loader, _, class_names = get_data_loaders(
        train_dir=train_dir,
        test_dir=None,
        batch_size=batch_size,
        val_ratio=val_ratio,
        subset_fraction=subset_fraction,
        seed=seed,
    )

    train_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset)
    print(f"Dataset: {train_samples} training samples, {val_samples} validation samples")

    # Build model
    pretrained = normalized_name != "custom_cnn"
    model = build_model(
        normalized_name,
        num_classes=len(class_names),
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    ).to(device)

    params_info = count_parameters(model)
    print(f"Parameters: {params_info['trainable']:,} trainable / {params_info['total']:,} total")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {
        "model_name": normalized_name,
        "display_name": display_name,
        "parameters": params_info,
        "train_loss": [],
        "train_acc": [],
        "train_error": [],
        "val_loss": [],
        "val_acc": [],
        "val_error": [],
        "epoch_times": [],
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_no_improve = 0

    total_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        tr_loss, tr_acc, tr_err = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch=epoch, total_epochs=epochs
        )
        val_loss, val_acc, val_err = validate_one_epoch(
            model, val_loader, criterion, device, epoch=epoch, total_epochs=epochs
        )
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc * 100.0, 2))
        history["train_error"].append(round(tr_err * 100.0, 2))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc * 100.0, 2))
        history["val_error"].append(round(val_err * 100.0, 2))
        history["epoch_times"].append(round(epoch_time, 2))

        # Checkpoint if validation accuracy improved
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "model_name": normalized_name,
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "epoch": epoch,
                },
                output_path,
            )
            saved_tag = " -> Best Checkpoint Saved"
        else:
            epochs_no_improve += 1
            saved_tag = ""

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {tr_loss:.4f} (Acc: {tr_acc*100:5.2f}%, Err: {tr_err*100:5.2f}%) | "
            f"Val Loss: {val_loss:.4f} (Acc: {val_acc*100:5.2f}%, Err: {val_err*100:5.2f}%) | "
            f"Time: {epoch_time:4.1f}s{saved_tag}"
        )

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {patience} epochs).")
            break

    total_training_time = time.time() - total_start
    history["total_time_seconds"] = round(total_training_time, 2)
    history["best_val_acc"] = round(best_val_acc * 100.0, 2)
    history["best_val_loss"] = round(best_val_loss, 4)

    # Save training history
    history_file = RESULTS_DIR / f"{normalized_name}_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining Complete for {display_name}!")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}% | Best Val Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_path}")
    print(f"History saved to: {history_file}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train brain tumor classification models.")
    parser.add_argument(
        "--model",
        default="custom_cnn",
        choices=["custom_cnn", "resnet18", "mobilenet_v3", "vit", "all"],
        help="Model architecture to train",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--subset-fraction", type=float, default=1.0, help="Subsample dataset (e.g., 0.2 for fast demo)")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze pretrained backbone weights")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else get_device()
    targets = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]

    for model_target in targets:
        train_model(
            model_name=model_target,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_ratio=args.val_ratio,
            subset_fraction=args.subset_fraction,
            freeze_backbone=args.freeze_backbone,
            patience=args.patience,
            device=device,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
