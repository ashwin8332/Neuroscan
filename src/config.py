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

# Class display colors & plain-English patient descriptions for UI/Visualizations
CLASS_METADATA = {
    "glioma": {
        "display_name": "Glioma Tumor",
        "color": "#f43f5e",
        "icon": "fa-brain",
        "severity": "High Severity",
        "simple_summary": "A tumor that occurs in the brain and spinal cord, starting in supportive glial cells.",
        "plain_details": "Glial cells surround and nourish neurons. Gliomas can grow into surrounding brain tissue, requiring specialist evaluation.",
        "what_it_means": "The AI detected patterns typical of glial tissue growth. A neurologist or neurosurgeon typically reviews these findings for targeted treatment planning.",
    },
    "meningioma": {
        "display_name": "Meningioma Tumor",
        "color": "#f59e0b",
        "icon": "fa-shield-halved",
        "severity": "Moderate Severity",
        "simple_summary": "Tumors that arise from the meninges (the protective layers/membranes covering the brain).",
        "plain_details": "The meninges are protective outer membranes shielding the brain and spinal cord. Meningiomas are usually non-cancerous (benign) and grow slowly from the outside inward.",
        "what_it_means": "The AI found an outer-membrane mass. Doctors often monitor them with regular imaging or treat them if they compress surrounding brain structures.",
    },
    "notumor": {
        "display_name": "Normal / Healthy Brain",
        "color": "#10b981",
        "icon": "fa-circle-check",
        "severity": "Normal / Clear",
        "simple_summary": "A normal, healthy brain scan with no evidence of tumor or abnormal tissue growth.",
        "plain_details": "Brain tissues, ventricles, and hemispheres appear symmetrical and clear, with no signs of swelling, abnormal growths, or lesions.",
        "what_it_means": "All neural networks found normal, healthy brain anatomy with no detectable tumor signals on this scan.",
    },
    "pituitary": {
        "display_name": "Pituitary Tumor",
        "color": "#3b82f6",
        "icon": "fa-dna",
        "severity": "Moderate Severity",
        "simple_summary": "Tumors located in the pituitary gland (the master hormone gland at the base of the brain).",
        "plain_details": "The pituitary gland produces hormones regulating growth, blood pressure, and metabolism. Most pituitary tumors are benign adenomas that may affect hormone levels or vision.",
        "what_it_means": "The AI detected a growth in the pituitary region. Healthcare providers typically evaluate hormone levels and eyesight.",
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
