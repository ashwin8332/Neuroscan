from pathlib import Path
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BrainTumorCNN(nn.Module):
    """
    Custom Deep Convolutional Neural Network designed specifically for
    Brain MRI multi-class tumor classification.
    """
    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.4):
        super().__init__()
        # Conv Block 1: 3 -> 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Conv Block 2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Conv Block 3: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Conv Block 4: 128 -> 256
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully Connected Head
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1))
        # Block 2
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1))
        # Block 3
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.1))
        # Block 4
        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.1))
        
        # Global pooling & classification
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.bn_fc(self.fc1(x)), negative_slope=0.1)
        x = self.dropout(x)
        return self.fc2(x)


def build_resnet18(
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.4,
) -> nn.Module:
    """ResNet-18 model with transfer learning and custom classification head."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate),
        nn.Linear(256, num_classes),
    )
    return model


def build_mobilenet_v3(
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.3,
) -> nn.Module:
    """MobileNetV3-Small: Ultra-efficient modern CNN with high accuracy & low latency."""
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=dropout_rate),
        nn.Linear(256, num_classes),
    )
    return model


def build_vit_b16(
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.3,
) -> nn.Module:
    """Vision Transformer (ViT-B/16) with self-attention for medical imaging."""
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vit_b_16(weights=weights)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("heads"):
                param.requires_grad = False

    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate),
        nn.Linear(256, num_classes),
    )
    return model


MODEL_REGISTRY = {
    "custom_cnn": {
        "builder": lambda nc, pt, fb: BrainTumorCNN(num_classes=nc),
        "display_name": "Custom CNN",
        "description": "4-Stage Deep CNN with BatchNorm, LeakyReLU & Adaptive Pooling",
        "checkpoint": "custom_cnn_final_model.pth",
    },
    "resnet18": {
        "builder": lambda nc, pt, fb: build_resnet18(num_classes=nc, pretrained=pt, freeze_backbone=fb),
        "display_name": "ResNet-18 (Transfer Learning)",
        "description": "Deep Residual Network with Skip Connections & Pretrained ImageNet Weights",
        "checkpoint": "resnet18_final_model.pth",
    },
    "mobilenet_v3": {
        "builder": lambda nc, pt, fb: build_mobilenet_v3(num_classes=nc, pretrained=pt, freeze_backbone=fb),
        "display_name": "MobileNetV3 (Efficient CNN)",
        "description": "Hardware-efficient CNN with Squeeze-and-Excitation & Hard-Swish",
        "checkpoint": "mobilenet_v3_final_model.pth",
    },
    "vit": {
        "builder": lambda nc, pt, fb: build_vit_b16(num_classes=nc, pretrained=pt, freeze_backbone=fb),
        "display_name": "Vision Transformer (ViT-B/16)",
        "description": "Pure Transformer Architecture with Multi-Head Self-Attention over Patches",
        "checkpoint": "best_vit_model.pth",
    },
}


def build_model(
    model_name: str,
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Factory method to instantiate any of the supported models."""
    normalized = model_name.lower().replace("-", "_").replace(" ", "_")
    if normalized in MODEL_REGISTRY:
        return MODEL_REGISTRY[normalized]["builder"](num_classes, pretrained, freeze_backbone)
    raise ValueError(f"Unknown model architecture '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Calculate trainable and non-trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load model weights from a .pth checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
