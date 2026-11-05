"""Classify a single brain-MRI slice from the command line.

    python predict.py sample_scans/glioma/Te-gl_10.jpg            # calibrated ensemble
    python predict.py scan.png --model resnet18                    # one model
    python predict.py scan.png --json > result.json                # machine-readable
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from src.config import MEDICAL_DISCLAIMER, MODELS_DIR, get_device
from src.inference import InferenceService
from src.models import MODEL_REGISTRY


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Predict the brain tumor class of an MRI slice.")
    parser.add_argument("image", type=Path, help="Path to a JPEG/PNG MRI slice")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), help="Use one model instead of the ensemble")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--no-tta", action="store_true", help="Disable flip test-time augmentation")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a report")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args(argv)

    if not args.image.is_file():
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        return 2

    device = torch.device("cpu") if args.cpu else get_device()
    service = InferenceService(models_dir=args.models_dir, device=device, tta=not args.no_tta).load()
    if args.model:
        if args.model not in service.models:
            print(f"Error: {args.model} is unavailable ({service.unavailable.get(args.model, 'not loaded')})", file=sys.stderr)
            return 2
        service.models = {args.model: service.models[args.model]}
    if not service.models:
        print(f"Error: no trained checkpoints found in {args.models_dir}", file=sys.stderr)
        return 2

    result = service.predict(args.image, with_gradcam=False)

    if args.json:
        slim = {k: v for k, v in result.items() if not k.endswith("_b64") and k not in {"gradcam", "categories_guide"}}
        print(json.dumps(slim, indent=2))
        return 0

    width = 64
    print("=" * width)
    print(f" {args.image}")
    print("=" * width)
    print(f" Finding    : {result['consensus_display_name']}")
    print(f" Confidence : {result['confidence_level']} ({result['consensus_confidence']:.1f}% top class, "
          f"{result['agreement_text']})")
    print("-" * width)
    for cls, prob in sorted(result["ensemble_probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {cls:<11} {prob:6.2f}% {'#' * int(round(prob / 100 * 36))}")
    print("-" * width)
    for row in result["model_results"]:
        print(f"  {row['display_name']:<32} {row['predicted_class']:<11} {row['confidence_pct']:6.2f}%  "
              f"(weight {row['weight'] * 100:.0f}%)")
    for missing in result["unavailable_models"]:
        print(f"  {missing['display_name']:<32} not used: {missing['reason']}")
    for warning in result["warnings"]:
        print(f"  ! {warning}")
    print("-" * width)
    print(f" {MEDICAL_DISCLAIMER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
