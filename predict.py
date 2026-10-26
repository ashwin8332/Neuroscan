import argparse
from pathlib import Path
import torch

from src.config import CLASS_NAMES, MODELS_DIR, get_device
from src.inference import predict_single_image
from src.models import MODEL_REGISTRY, build_model, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Predict brain tumor class from an MRI image.")
    parser.add_argument("image", type=str, help="Path to brain MRI image file")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture to use for prediction",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return

    device = torch.device("cpu") if args.cpu else get_device()
    spec = MODEL_REGISTRY[args.model]
    ckpt_path = MODELS_DIR / spec["checkpoint"]

    print(f"\nAnalyzing: {image_path}")
    print(f"Model: {spec['display_name']} ({device})")

    model = build_model(args.model, num_classes=len(CLASS_NAMES), pretrained=not ckpt_path.exists())
    if ckpt_path.exists():
        model = load_checkpoint(model, ckpt_path, device)
        print(f"Loaded checkpoint from: {ckpt_path}")
    else:
        print("Note: Running with base ImageNet weights (no fine-tuned checkpoint found).")

    result = predict_single_image(model, image_path, device)

    print("\n" + "=" * 50)
    print(f"Prediction: {result['predicted_class'].upper()} ({result['confidence']*100:.2f}% confidence)")
    print("=" * 50)
    print("Class Confidence Distribution:")
    for cls_name, prob in result["confidences"].items():
        bar = "#" * int(prob * 30)
        print(f"  {cls_name:<12} : {prob*100:5.2f}% {bar}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
