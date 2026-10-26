from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .config import CLASS_NAMES, IMAGE_SIZE, MODELS_DIR, get_device
from .data import get_test_transforms
from .models import MODEL_REGISTRY, build_model, load_checkpoint


def prepare_image(image_input: Union[str, Path, Image.Image, np.ndarray]) -> Tuple[torch.Tensor, Image.Image]:
    """Preprocess image input to normalized tensor and PIL Image."""
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(str(image_input)).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")

    transforms_fn = get_test_transforms(IMAGE_SIZE)
    tensor = transforms_fn(pil_img).unsqueeze(0)  # Shape: [1, 3, 224, 224]
    return tensor, pil_img


def predict_single_image(
    model: torch.nn.Module,
    image_input: Union[str, Path, Image.Image, np.ndarray],
    device: torch.device,
    class_names: list = CLASS_NAMES,
) -> Dict[str, Union[str, float, Dict[str, float]]]:
    """
    Generate inference prediction and confidence scores for a single brain MRI image.
    """
    tensor, _ = prepare_image(image_input)
    tensor = tensor.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_idx = int(np.argmax(probs))
    predicted_class = class_names[top_idx]
    confidence = float(probs[top_idx])

    confidence_dict = {
        class_names[i]: float(probs[i]) for i in range(len(class_names))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "confidences": confidence_dict,
    }


def generate_gradcam_overlay(
    model: torch.nn.Module,
    image_input: Union[str, Path, Image.Image, np.ndarray],
    device: torch.device,
    target_category: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Generates Grad-CAM visual explanation overlay.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        tensor, pil_img = prepare_image(image_input)
        tensor = tensor.to(device)

        # Identify target convolutional layer depending on architecture
        target_layers = None
        if hasattr(model, "conv4"):  # Custom CNN
            target_layers = [model.conv4]
        elif hasattr(model, "layer4"):  # ResNet
            target_layers = [model.layer4[-1]]
        elif hasattr(model, "features"):  # MobileNet
            target_layers = [model.features[-1]]

        if target_layers is None:
            return None

        # Resize original image to 224x224 float [0, 1] for overlay
        rgb_img = np.float32(pil_img.resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0

        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        return visualization
    except Exception:
        return None


def get_available_trained_models(
    device: Optional[torch.device] = None,
    models_dir: Path = MODELS_DIR,
) -> Dict[str, torch.nn.Module]:
    """
    Scan models directory and load all available saved checkpoints.
    """
    if device is None:
        device = get_device()

    loaded = {}
    for key, spec in MODEL_REGISTRY.items():
        ckpt_path = models_dir / spec["checkpoint"]
        if ckpt_path.exists():
            model = build_model(key, num_classes=len(CLASS_NAMES), pretrained=False)
            model = load_checkpoint(model, ckpt_path, device)
            loaded[spec["display_name"]] = model

    return loaded
