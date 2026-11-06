"""Central configuration: paths, class metadata and per-model training recipes.

Paths can be overridden with environment variables so the same code runs on a
laptop, in Colab and inside a container:

* ``NEUROSCAN_DATA_DIR``     - folder that contains ``Training/`` and ``Testing/``
* ``NEUROSCAN_MODELS_DIR``   - folder with ``*.pth`` checkpoints
* ``NEUROSCAN_RESULTS_DIR``  - folder for metrics, histories and plots
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

# NOTE: no torch import at module level. This module is imported by the web
# server, which runs on Vercel with ONNX Runtime only (PyTorch does not fit in
# a serverless function). get_device() imports torch lazily.

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parents[1]


def _resolve_data_dir() -> Path:
    env_dir = os.environ.get("NEUROSCAN_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    for candidate in (
        ROOT_DIR / "Dataset" / "brain-tumor-mri-dataset",
        ROOT_DIR / "brain-tumor-mri-dataset",
    ):
        if candidate.exists():
            return candidate
    return ROOT_DIR / "Dataset" / "brain-tumor-mri-dataset"


DATA_DIR = _resolve_data_dir()
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"

MODELS_DIR = Path(os.environ.get("NEUROSCAN_MODELS_DIR", ROOT_DIR / "models")).expanduser()
RESULTS_DIR = Path(os.environ.get("NEUROSCAN_RESULTS_DIR", ROOT_DIR / "results")).expanduser()
PLOTS_DIR = RESULTS_DIR / "plots"
PUBLIC_DIR = ROOT_DIR / "public"            # static site: served by Vercel's CDN
FRONTEND_DIR = PUBLIC_DIR                    # backwards-compatible alias
STATIC_DIR = PUBLIC_DIR / "static"
SAMPLE_SCANS_DIR = PUBLIC_DIR / "samples"
ONNX_DIR = Path(os.environ.get("NEUROSCAN_ONNX_DIR", MODELS_DIR / "onnx")).expanduser()
ON_VERCEL = bool(os.environ.get("VERCEL"))


def ensure_output_dirs() -> None:
    """Create output folders. Never fails on read-only deployments."""
    for directory in (MODELS_DIR, RESULTS_DIR, PLOTS_DIR):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass


ensure_output_dirs()

# --------------------------------------------------------------------------- #
# Dataset specification
# --------------------------------------------------------------------------- #
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 4
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASS_NAMES)}

# Folder names used by the public source datasets (Kaggle "Brain Tumor MRI",
# SARTAJ, Br35H) mapped onto the canonical class names above.
CLASS_ALIASES = {
    "glioma": "glioma",
    "glioma_tumor": "glioma",
    "meningioma": "meningioma",
    "meningioma_tumor": "meningioma",
    "notumor": "notumor",
    "no_tumor": "notumor",
    "no": "notumor",
    "normal": "notumor",
    "pituitary": "pituitary",
    "pituitary_tumor": "pituitary",
}

# Plain-English, non-diagnostic descriptions used by the UIs. The wording is
# deliberately cautious: the models recognise visual patterns in four classes
# and cannot rule out other conditions.
CLASS_METADATA = {
    "glioma": {
        "short_label": "glioma",
        "display_name": "Glioma Pattern",
        "color": "#f43f5e",
        "icon": "fa-brain",
        "severity": "Specialist review advised",
        "severity_level": "high",
        "simple_summary": "A tumor that starts in the glial (support) cells of the brain or spinal cord.",
        "plain_details": "Glial cells surround and nourish nerve cells. Gliomas range from slow-growing to fast-growing and can spread into nearby brain tissue.",
        "what_it_means": "The scan shows visual patterns the models associate with gliomas. Only a radiologist or neurologist can confirm this, usually with further imaging.",
    },
    "meningioma": {
        "short_label": "meningioma",
        "display_name": "Meningioma Pattern",
        "color": "#f59e0b",
        "icon": "fa-shield-halved",
        "severity": "Specialist review advised",
        "severity_level": "moderate",
        "simple_summary": "A tumor that grows from the meninges, the protective layers that cover the brain.",
        "plain_details": "Most meningiomas are non-cancerous (benign) and grow slowly, pressing on the brain from the outside rather than growing into it.",
        "what_it_means": "The scan shows visual patterns the models associate with meningiomas. Doctors often monitor these with repeat imaging or treat them if they press on nearby structures.",
    },
    "notumor": {
        "short_label": "no tumor",
        "display_name": "No Tumor Pattern Found",
        "color": "#10b981",
        "icon": "fa-circle-check",
        "severity": "No tumor pattern detected",
        "severity_level": "none",
        "simple_summary": "The models did not find the patterns of the three tumor types they were trained on.",
        "plain_details": "This is not a clean bill of health: the models only recognise glioma, meningioma and pituitary tumor patterns and cannot detect other conditions.",
        "what_it_means": "No glioma, meningioma or pituitary tumor pattern was recognised. Any symptoms should still be discussed with a doctor.",
    },
    "pituitary": {
        "short_label": "pituitary tumor",
        "display_name": "Pituitary Tumor Pattern",
        "color": "#3b82f6",
        "icon": "fa-dna",
        "severity": "Specialist review advised",
        "severity_level": "moderate",
        "simple_summary": "A growth in the pituitary gland, the small hormone-producing gland at the base of the brain.",
        "plain_details": "Most pituitary tumors are benign adenomas. Because the gland controls many hormones and sits next to the optic nerves, they can affect hormone levels or vision.",
        "what_it_means": "The scan shows visual patterns the models associate with pituitary tumors. Doctors typically follow up with hormone tests and an eye examination.",
    },
}

# Display metadata for every architecture. Kept here (not in models.py) so the
# serving code can describe models without importing PyTorch.
MODEL_SPECS: Dict[str, Dict[str, Any]] = {
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

MEDICAL_DISCLAIMER = (
    "NeuroScan AI is a research and education tool. It is not a medical device, "
    "has not been clinically validated and must not be used to diagnose, treat or "
    "rule out any condition. Always consult a qualified radiologist or physician."
)

# --------------------------------------------------------------------------- #
# Training recipes
# --------------------------------------------------------------------------- #
SEED = 42
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 15


@dataclass(frozen=True)
class TrainRecipe:
    """Hyper-parameters for one architecture.

    ``lr`` is the learning rate of the newly initialised classification head.
    Pretrained backbones use ``lr * backbone_lr_mult`` (discriminative fine-
    tuning) so the ImageNet features are adapted gently instead of being
    overwritten, which is the main cause of the over-fitting seen in the
    original training runs.
    """

    epochs: int
    lr: float
    backbone_lr_mult: float
    weight_decay: float
    warmup_epochs: float
    freeze_epochs: int
    label_smoothing: float = 0.1
    batch_size: int = 32
    patience: int = 6
    ema_half_life_epochs: float = 1.0
    max_grad_norm: float = 5.0

    def to_dict(self) -> Dict:
        return asdict(self)


TRAIN_RECIPES: Dict[str, TrainRecipe] = {
    # Trained from scratch: needs a higher LR and more epochs (it under-fitted
    # with 5 epochs at 1e-4) - regularised by the global-pool head + dropout.
    "custom_cnn": TrainRecipe(
        epochs=40, lr=2e-3, backbone_lr_mult=1.0, weight_decay=0.05,
        warmup_epochs=2, freeze_epochs=0, patience=10,
    ),
    # Transfer learning: linear-probe first (LP-FT), then fine-tune the
    # backbone at 10x lower LR.
    "resnet18": TrainRecipe(
        epochs=15, lr=1e-3, backbone_lr_mult=0.1, weight_decay=0.05,
        warmup_epochs=1, freeze_epochs=1, patience=5,
    ),
    "mobilenet_v3": TrainRecipe(
        epochs=15, lr=1e-3, backbone_lr_mult=0.2, weight_decay=0.02,
        warmup_epochs=1, freeze_epochs=1, patience=5,
    ),
    "vit": TrainRecipe(
        epochs=10, lr=5e-4, backbone_lr_mult=0.06, weight_decay=0.05,
        warmup_epochs=1, freeze_epochs=1, patience=4, max_grad_norm=1.0,
    ),
}


def get_device(prefer_cuda: bool = True):
    """torch.device for training/evaluation (imports torch lazily)."""
    import torch

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
