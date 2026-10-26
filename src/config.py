from pathlib import Path
import torch

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]

# Dataset directory detection
if (ROOT_DIR / "Dataset" / "brain-tumor-mri-dataset").exists():
    DATA_DIR = ROOT_DIR / "Dataset" / "brain-tumor-mri-dataset"
elif (ROOT_DIR / "brain-tumor-mri-dataset").exists():
    DATA_DIR = ROOT_DIR / "brain-tumor-mri-dataset"
else:
    DATA_DIR = ROOT_DIR / "Dataset" / "brain-tumor-mri-dataset"

TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"

# Output directories
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset specifications
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 4
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASS_NAMES)}

# Class display colors & descriptions for UI/Visualizations
CLASS_METADATA = {
    "glioma": {
        "color": "#e63946",
        "description": "Glioma: Tumor originating in the glial cells of the brain or spine.",
        "severity": "High",
    },
    "meningioma": {
        "color": "#f4a261",
        "description": "Meningioma: Tumor arising from the meninges surrounding the brain and spinal cord.",
        "severity": "Moderate",
    },
    "notumor": {
        "color": "#2a9d8f",
        "description": "No Tumor: Normal healthy brain tissue scan without detectable neoplasms.",
        "severity": "Normal",
    },
    "pituitary": {
        "color": "#457b9d",
        "description": "Pituitary Tumor: Abnormal growth in the pituitary gland at the base of the brain.",
        "severity": "Moderate",
    },
}

# Training defaults
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 10
SEED = 42

def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
