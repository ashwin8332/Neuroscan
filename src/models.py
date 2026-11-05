"""Model architectures, registry and checkpoint (de)serialisation.

Checkpoints written by this version carry everything needed to reproduce
inference: architecture variant, class order, preprocessing, calibration
temperature and validation metrics. Older checkpoints (plain state dicts or
``{"model_state_dict", "class_names", "val_acc"}``) still load and run with the
exact preprocessing they were trained with.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .config import CLASS_NAMES
from .preprocessing import LEGACY_PREPROCESS, PreprocessConfig

CHECKPOINT_FORMAT_VERSION = 2


# --------------------------------------------------------------------------- #
# Custom CNN - v1 (original, kept so the shipped checkpoint keeps working)
# --------------------------------------------------------------------------- #
class BrainTumorCNN(nn.Module):
    """Original 4-block CNN (v1).

    Diagnosis: single conv per stage, then a 4x4 adaptive pool feeding a
    4096x256 dense layer that holds ~73% of the parameters. Shallow features +
    a huge dense head = slow to train and easy to over-fit; with 5 epochs at
    lr=1e-4 it under-fitted (89% train / 79% test accuracy).
    """

    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.1))
        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.1))
        x = torch.flatten(self.adaptive_pool(x), 1)
        x = F.leaky_relu(self.bn_fc(self.fc1(x)), negative_slope=0.1)
        return self.fc2(self.dropout(x))


# --------------------------------------------------------------------------- #
# Custom CNN - v2 (residual + squeeze-excitation + global-pool head)
# --------------------------------------------------------------------------- #
class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        return x * torch.sigmoid(self.fc2(F.silu(self.fc1(scale))))


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SqueezeExcite(out_ch)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        return F.silu(out + self.shortcut(x))


class BrainTumorCNNv2(nn.Module):
    """Compact residual CNN trained from scratch (~1.6M params, ~1.4 GFLOPs).

    * two 3x3 convs per stage + identity shortcuts -> deeper features, stable
      gradients (fixes under-fitting);
    * squeeze-and-excitation channel attention;
    * global average pooling + dropout + one linear layer instead of a 1M
      parameter dense layer (removes the main over-fitting risk).
    """

    def __init__(self, num_classes: int = 4, widths: Tuple[int, ...] = (32, 48, 96, 192), dropout_rate: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.SiLU(inplace=True),
        )
        stages: List[nn.Module] = []
        in_ch = widths[0]
        for out_ch in widths[1:]:
            stages.append(nn.Sequential(ResidualBlock(in_ch, out_ch, stride=2), ResidualBlock(out_ch, out_ch)))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout_rate), nn.Linear(in_ch, num_classes))
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # Zero-init the last BN of each residual branch: blocks start as identity.
        for module in self.modules():
            if isinstance(module, ResidualBlock):
                nn.init.zeros_(module.bn2.weight)

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.stages[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(self.stem(x))
        return self.head(F.adaptive_avg_pool2d(x, 1))


# --------------------------------------------------------------------------- #
# Transfer-learning backbones
# --------------------------------------------------------------------------- #
def _load_local_backbone(model: nn.Module, weights_path: Optional[Path]) -> None:
    """Load ImageNet weights from disk (offline / air-gapped training)."""
    if weights_path is None:
        return
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Pretrained weights {weights_path} do not match the architecture "
            f"(missing={missing[:5]}, unexpected={unexpected[:5]})"
        )


def build_resnet18(
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.4,
    weights_path: Optional[Path] = None,
) -> nn.Module:
    """ResNet-18 with a small MLP head (head layout unchanged for checkpoint compatibility)."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained and weights_path is None else None
    model = models.resnet18(weights=weights)
    _load_local_backbone(model, weights_path if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate),
        nn.Linear(256, num_classes),
    )
    if freeze_backbone:
        set_backbone_trainable(model, "resnet18", False)
    return model


def build_mobilenet_v3(
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.3,
    weights_path: Optional[Path] = None,
) -> nn.Module:
    """MobileNetV3-Small: SE attention + hard-swish, built for CPU latency."""
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained and weights_path is None else None
    model = models.mobilenet_v3_small(weights=weights)
    _load_local_backbone(model, weights_path if pretrained else None)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=dropout_rate),
        nn.Linear(256, num_classes),
    )
    if freeze_backbone:
        set_backbone_trainable(model, "mobilenet_v3", False)
    return model


def build_vit_b16(
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.3,
    weights_path: Optional[Path] = None,
) -> nn.Module:
    """Vision Transformer ViT-B/16."""
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained and weights_path is None else None
    model = models.vit_b_16(weights=weights)
    _load_local_backbone(model, weights_path if pretrained else None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate),
        nn.Linear(256, num_classes),
    )
    if freeze_backbone:
        set_backbone_trainable(model, "vit", False)
    return model


# Name prefix of the newly initialised classification head for each model.
HEAD_PREFIXES = {
    "resnet18": ("fc.",),
    "mobilenet_v3": ("classifier.",),
    "vit": ("heads.",),
    "custom_cnn": (),  # trained from scratch: every parameter is "head"
}

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "custom_cnn": {
        "display_name": "Custom CNN",
        "description": "Residual CNN with squeeze-excitation, trained from scratch",
        "checkpoint": "custom_cnn_final_model.pth",
        "pretrained": False,
    },
    "resnet18": {
        "display_name": "ResNet-18 (Transfer Learning)",
        "description": "Deep residual network fine-tuned from ImageNet",
        "checkpoint": "resnet18_final_model.pth",
        "pretrained": True,
    },
    "mobilenet_v3": {
        "display_name": "MobileNetV3 (Efficient CNN)",
        "description": "Hardware-efficient CNN with squeeze-excitation and hard-swish",
        "checkpoint": "mobilenet_v3_final_model.pth",
        "pretrained": True,
    },
    "vit": {
        "display_name": "Vision Transformer (ViT-B/16)",
        "description": "Transformer with multi-head self-attention over 16x16 patches",
        "checkpoint": "best_vit_model.pth",
        "pretrained": True,
    },
}


def normalize_model_name(model_name: str) -> str:
    return model_name.lower().replace("-", "_").replace(" ", "_")


def build_model(
    model_name: str,
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    arch: Optional[str] = None,
    weights_path: Optional[Path] = None,
) -> nn.Module:
    """Instantiate a registered model. ``arch`` selects ``custom_cnn_v1``/``v2``."""
    key = normalize_model_name(model_name)
    if key == "custom_cnn":
        return BrainTumorCNN(num_classes=num_classes) if arch == "custom_cnn_v1" else BrainTumorCNNv2(num_classes=num_classes)
    builders = {"resnet18": build_resnet18, "mobilenet_v3": build_mobilenet_v3, "vit": build_vit_b16}
    if key not in builders:
        raise ValueError(f"Unknown model architecture '{model_name}'. Available: {list(MODEL_REGISTRY)}")
    return builders[key](num_classes=num_classes, pretrained=pretrained,
                         freeze_backbone=freeze_backbone, weights_path=weights_path)


def default_arch(model_name: str) -> str:
    key = normalize_model_name(model_name)
    return "custom_cnn_v2" if key == "custom_cnn" else key


def is_head_parameter(model_name: str, param_name: str) -> bool:
    prefixes = HEAD_PREFIXES.get(normalize_model_name(model_name), ())
    return not prefixes or param_name.startswith(prefixes)


def set_backbone_trainable(model: nn.Module, model_name: str, trainable: bool) -> None:
    """Freeze/unfreeze everything except the classification head (for LP-FT)."""
    for name, param in model.named_parameters():
        if not is_head_parameter(model_name, name):
            param.requires_grad = trainable


def parameter_groups(
    model: nn.Module, model_name: str, lr: float, backbone_lr_mult: float, weight_decay: float
) -> List[Dict[str, Any]]:
    """AdamW groups: head vs. backbone LR, and no weight decay on norms/biases."""
    groups: Dict[Tuple[bool, bool], List[nn.Parameter]] = {}
    for name, param in model.named_parameters():
        head = is_head_parameter(model_name, name)
        no_decay = param.ndim <= 1 or name.endswith(".bias") or "pos_embedding" in name or "class_token" in name
        groups.setdefault((head, no_decay), []).append(param)
    result = []
    for (head, no_decay), params in groups.items():
        group_lr = lr if head else lr * backbone_lr_mult
        result.append({
            "params": params,
            "lr": group_lr,
            "initial_lr": group_lr,
            "weight_decay": 0.0 if no_decay else weight_decay,
            "group_name": f"{'head' if head else 'backbone'}{'/no_decay' if no_decay else ''}",
        })
    return result


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def gradcam_target_layer(model: nn.Module) -> Optional[nn.Module]:
    """Last spatial feature layer used for Grad-CAM, or ``None`` (e.g. ViT)."""
    if isinstance(model, BrainTumorCNNv2):
        return model.gradcam_layer
    if isinstance(model, BrainTumorCNN):
        return model.conv4
    if isinstance(model, models.ResNet):
        return model.layer4[-1]
    if isinstance(model, models.MobileNetV3):
        return model.features[-1]
    return None


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
@dataclass
class CheckpointInfo:
    model_name: str
    arch: str
    class_names: List[str]
    preprocess: PreprocessConfig
    temperature: float = 1.0
    format_version: int = 1
    val_metrics: Dict[str, Any] = field(default_factory=dict)
    train_config: Dict[str, Any] = field(default_factory=dict)
    epoch: Optional[int] = None
    created_at: Optional[str] = None

    @property
    def is_legacy(self) -> bool:
        return self.format_version < CHECKPOINT_FORMAT_VERSION

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "arch": self.arch,
            "format_version": self.format_version,
            "temperature": round(float(self.temperature), 4),
            "preprocess": self.preprocess.to_dict(),
            "val_metrics": self.val_metrics,
            "epoch": self.epoch,
            "created_at": self.created_at,
        }


def _infer_arch(model_name: str, state_dict: Dict[str, torch.Tensor]) -> str:
    if model_name == "custom_cnn":
        return "custom_cnn_v1" if "fc1.weight" in state_dict else "custom_cnn_v2"
    return model_name


def read_checkpoint(path: Path, model_name: str) -> Tuple[Dict[str, torch.Tensor], CheckpointInfo]:
    """Load a checkpoint safely (``weights_only=True``: no arbitrary code execution)."""
    key = normalize_model_name(model_name)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict, meta = raw["model_state_dict"], raw
    elif isinstance(raw, dict) and "state_dict" in raw:
        state_dict, meta = raw["state_dict"], raw
    else:
        state_dict, meta = raw, {}

    val_metrics = dict(meta.get("val_metrics") or {})
    for legacy_key in ("val_acc", "val_loss"):
        if legacy_key in meta and legacy_key not in val_metrics:
            val_metrics[legacy_key] = float(meta[legacy_key])

    info = CheckpointInfo(
        model_name=key,
        arch=meta.get("arch") or _infer_arch(key, state_dict),
        class_names=list(meta.get("class_names") or CLASS_NAMES),
        preprocess=PreprocessConfig.from_dict(meta.get("preprocess")) if meta.get("preprocess") else LEGACY_PREPROCESS,
        temperature=float(meta.get("temperature", 1.0)),
        format_version=int(meta.get("format_version", 1)),
        val_metrics=val_metrics,
        train_config=dict(meta.get("train_config") or {}),
        epoch=meta.get("epoch"),
        created_at=meta.get("created_at"),
    )
    if info.class_names != list(CLASS_NAMES):
        raise ValueError(f"{path}: class order {info.class_names} differs from {CLASS_NAMES}")
    return state_dict, info


def load_model_from_checkpoint(
    model_name: str, checkpoint_path: Path, device: torch.device
) -> Tuple[nn.Module, CheckpointInfo]:
    state_dict, info = read_checkpoint(checkpoint_path, model_name)
    model = build_model(model_name, num_classes=len(info.class_names), pretrained=False, arch=info.arch)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, info


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Backwards-compatible helper: load weights into an existing model."""
    raw = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        raw = raw["model_state_dict"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    model.load_state_dict(raw)
    return model.to(device).eval()


def to_builtin(value: Any) -> Any:
    """Recursively convert numpy/torch scalars so ``torch.load(weights_only=True)`` accepts them."""
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return str(value)  # drops str subclasses such as torch.torch_version.TorchVersion
    if hasattr(value, "tolist"):  # numpy arrays/scalars and torch tensors
        return to_builtin(value.tolist())
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_name: str,
    arch: str,
    preprocess: PreprocessConfig,
    temperature: float = 1.0,
    val_metrics: Optional[Dict[str, Any]] = None,
    train_config: Optional[Dict[str, Any]] = None,
    epoch: Optional[int] = None,
    class_names: Iterable[str] = CLASS_NAMES,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_name": normalize_model_name(model_name),
        "arch": arch,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "class_names": list(class_names),
        "preprocess": preprocess.to_dict(),
        "temperature": float(temperature),
        "val_metrics": to_builtin(val_metrics or {}),
        "train_config": to_builtin(train_config or {}),
        "epoch": None if epoch is None else int(epoch),
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "torch_version": str(torch.__version__),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic: a crash mid-save never corrupts the served model
