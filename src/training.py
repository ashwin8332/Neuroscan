"""Training engine.

What changed versus the original loop and why
---------------------------------------------
* **LP-FT** - the randomly initialised head is trained first with the
  backbone frozen; fine-tuning a pretrained network through a random head
  distorts its features and is a known cause of over-fitting.
* **Discriminative learning rates** - backbone LR = head LR x 0.06-0.2.
* **Warm-up + cosine decay per step** (the backbone gets its own warm-up when it
  is unfrozen) instead of ReduceLROnPlateau on a noisy signal.
* **Label smoothing, weight decay without decaying norms/biases, gradient
  clipping and an EMA of the weights** - all reduce over-confidence and the
  epoch-to-epoch validation swings seen in the original histories.
* **Model selection on validation loss** (not accuracy) on a leak-free split,
  then **temperature calibration**; the checkpoint stores everything inference
  needs.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm

from .calibration import fit_temperature, nll
from .config import RESULTS_DIR, SEED, TrainRecipe
from .data import DataBundle
from .metrics import collect_logits, expected_calibration_error
from .models import (
    MODEL_REGISTRY,
    count_parameters,
    normalize_model_name,
    parameter_groups,
    save_checkpoint,
    set_backbone_trainable,
)
from .preprocessing import PreprocessConfig

logger = logging.getLogger(__name__)

MIN_CALIBRATION_SAMPLES = 100


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ModelEMA:
    """Exponential moving average of parameters and BatchNorm buffers."""

    def __init__(self, model: nn.Module, decay: float):
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        # Warm-up: track the live weights closely during the first steps.
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        live = model.state_dict()
        for name, value in self.module.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(live[name].detach(), alpha=1.0 - decay)
            else:
                value.copy_(live[name])


def _schedule(step: int, start: int, warmup: int, total: int, min_ratio: float = 0.01) -> float:
    """Linear warm-up from ``start`` then cosine decay to ``min_ratio`` at ``total``."""
    if step < start:
        return 0.0
    local = step - start
    if local < warmup:
        return max(1e-3, (local + 1) / max(1, warmup))
    progress = min(1.0, (local - warmup) / max(1, total - start - warmup))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass
class TrainResult:
    history: Dict[str, Any]
    checkpoint_path: Path
    temperature: float
    best_epoch: int
    val_metrics: Dict[str, Any] = field(default_factory=dict)


def _validate(model: nn.Module, loader, device: torch.device) -> Dict[str, float]:
    logits, labels = collect_logits(model, loader, device)
    probs = F.softmax(logits, dim=1).numpy()
    y_true = labels.numpy()
    y_pred = probs.argmax(axis=1)
    return {
        "loss": nll(logits, labels),
        "acc": float((y_pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "ece": expected_calibration_error(y_true, probs),
    }


def train_model(
    model: nn.Module,
    model_name: str,
    data: DataBundle,
    recipe: TrainRecipe,
    device: torch.device,
    output_path: Path,
    preprocess: PreprocessConfig,
    arch: str,
    results_dir: Path = RESULTS_DIR,
    seed: int = SEED,
    use_ema: bool = True,
    extra_config: Optional[Dict[str, Any]] = None,
) -> TrainResult:
    key = normalize_model_name(model_name)
    set_seed(seed)
    model.to(device)
    params = count_parameters(model)

    steps_per_epoch = max(1, len(data.train_loader))
    total_steps = steps_per_epoch * recipe.epochs
    warmup_steps = int(round(recipe.warmup_epochs * steps_per_epoch))
    unfreeze_step = recipe.freeze_epochs * steps_per_epoch

    groups = parameter_groups(model, key, recipe.lr, recipe.backbone_lr_mult, recipe.weight_decay)
    optimizer = torch.optim.AdamW(groups, lr=recipe.lr)
    lambdas = [
        (lambda s: _schedule(s, 0, warmup_steps, total_steps))
        if g["group_name"].startswith("head")
        else (lambda s: _schedule(s, unfreeze_step, warmup_steps, total_steps))
        for g in groups
    ]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)

    criterion = nn.CrossEntropyLoss(label_smoothing=recipe.label_smoothing)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema_decay = 0.5 ** (1.0 / max(1.0, recipe.ema_half_life_epochs * steps_per_epoch))
    ema = ModelEMA(model, ema_decay) if use_ema else None

    if recipe.freeze_epochs > 0:
        set_backbone_trainable(model, key, False)

    history: Dict[str, Any] = {
        "model_name": key,
        "display_name": MODEL_REGISTRY.get(key, {}).get("display_name", key),
        "arch": arch,
        "recipe": recipe.to_dict(),
        "parameters": params,
        "split": data.split_report.to_dict(),
        "train_loss": [], "train_acc": [], "train_error": [],
        "val_loss": [], "val_acc": [], "val_error": [], "val_macro_f1": [], "val_ece": [],
        "lr": [], "epoch_times": [],
    }

    best = {"loss": float("inf"), "epoch": 0, "state": None, "metrics": {}}
    epochs_without_improvement = 0
    step = 0
    start_time = time.time()
    logger.info("Training %s: %d train / %d val images, %d steps/epoch",
                key, len(data.train_loader.dataset), len(data.val_loader.dataset), steps_per_epoch)

    for epoch in range(1, recipe.epochs + 1):
        if recipe.freeze_epochs > 0 and epoch == recipe.freeze_epochs + 1:
            set_backbone_trainable(model, key, True)
            logger.info("Epoch %d: backbone unfrozen (LP-FT phase 2)", epoch)

        model.train()
        epoch_start = time.time()
        loss_sum, correct, seen = 0.0, 0, 0
        progress = tqdm(data.train_loader, desc=f"Epoch {epoch:02d}/{recipe.epochs}", unit="batch", leave=False, disable=None)
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if recipe.max_grad_norm:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), recipe.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            if ema is not None:
                ema.update(model)

            batch = labels.size(0)
            loss_sum += float(loss.item()) * batch
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += batch
            progress.set_postfix(loss=f"{loss_sum / seen:.4f}", acc=f"{100.0 * correct / seen:.1f}%")

        eval_model = ema.module if ema is not None else model
        val = _validate(eval_model, data.val_loader, device)
        train_loss, train_acc = loss_sum / max(seen, 1), correct / max(seen, 1)
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc * 100.0, 2))
        history["train_error"].append(round((1.0 - train_acc) * 100.0, 2))
        history["val_loss"].append(round(val["loss"], 4))
        history["val_acc"].append(round(val["acc"] * 100.0, 2))
        history["val_error"].append(round((1.0 - val["acc"]) * 100.0, 2))
        history["val_macro_f1"].append(round(val["macro_f1"], 4))
        history["val_ece"].append(round(val["ece"], 4))
        history["lr"].append([round(g["lr"], 8) for g in optimizer.param_groups])
        history["epoch_times"].append(round(epoch_time, 2))

        improved = val["loss"] < best["loss"] - 1e-4
        if improved or best["state"] is None:
            best.update(loss=val["loss"], epoch=epoch, metrics=val,
                        state={k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()})
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        logger.info(
            "Epoch %02d/%d | train loss %.4f acc %.2f%% | val loss %.4f acc %.2f%% F1 %.4f ECE %.4f | %.0fs%s",
            epoch, recipe.epochs, train_loss, train_acc * 100, val["loss"], val["acc"] * 100,
            val["macro_f1"], val["ece"], epoch_time, "  <- best" if improved else "",
        )
        if epochs_without_improvement >= recipe.patience:
            logger.info("Early stopping: no val-loss improvement for %d epochs", recipe.patience)
            break

    # ---- restore best weights, calibrate, save ------------------------------
    final_model = ema.module if ema is not None else model
    final_model.load_state_dict(best["state"])
    val_logits, val_labels = collect_logits(final_model, data.val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)
    # Temperature scaling is only trustworthy with enough validation data and a
    # real NLL gain; otherwise noise in a tiny split would mis-calibrate.
    if len(val_labels) < MIN_CALIBRATION_SAMPLES or nll(val_logits, val_labels, temperature) > 0.995 * nll(val_logits, val_labels):
        temperature = 1.0
    val_probs_raw = F.softmax(val_logits, dim=1).numpy()
    val_probs_cal = F.softmax(val_logits / temperature, dim=1).numpy()
    y_val = val_labels.numpy()
    val_metrics = {
        "val_loss": round(best["metrics"]["loss"], 4),
        "val_loss_calibrated": round(nll(val_logits, val_labels, temperature), 4),
        "val_acc": round(best["metrics"]["acc"], 4),
        "val_macro_f1": round(best["metrics"]["macro_f1"], 4),
        "val_ece": round(expected_calibration_error(y_val, val_probs_raw), 4),
        "val_ece_calibrated": round(expected_calibration_error(y_val, val_probs_cal), 4),
        "best_epoch": best["epoch"],
    }

    train_config = {"recipe": recipe.to_dict(), "seed": seed, "ema": use_ema, "arch": arch,
                    "split": data.split_report.to_dict(), **(extra_config or {})}
    save_checkpoint(
        output_path, final_model, key, arch, preprocess,
        temperature=temperature, val_metrics=val_metrics, train_config=train_config, epoch=best["epoch"],
    )

    history.update({
        "best_epoch": best["epoch"],
        "best_val_loss": round(best["loss"], 4),
        "best_val_acc": round(best["metrics"]["acc"] * 100.0, 2),
        "temperature": round(temperature, 4),
        "val_metrics": val_metrics,
        "total_time_seconds": round(time.time() - start_time, 2),
        "generalization_gap_pct": [round(t - v, 2) for t, v in zip(history["train_acc"], history["val_acc"])],
    })
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{key}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Best epoch %d | val loss %.4f | T=%.3f | val ECE %.4f -> %.4f | saved %s",
                best["epoch"], best["loss"], temperature, val_metrics["val_ece"],
                val_metrics["val_ece_calibrated"], output_path)
    return TrainResult(history, Path(output_path), temperature, best["epoch"], val_metrics)
