"""Export the PyTorch checkpoints in models/ to ONNX for serving (Vercel, CPU).

    python scripts/export_onnx.py            # export every checkpoint present, verify parity
    python scripts/export_onnx.py --check    # exit 1 if an export is missing or stale

The web app (server.py) serves models/onnx/*.onnx with ONNX Runtime, so run
this after every training run, then commit models/onnx/. The manifest records
the SHA-256 of each source checkpoint; `--check` (also run by the test-suite
and CI) fails when a .pth changed but was not re-exported.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MODEL_SPECS, MODELS_DIR, ONNX_DIR  # noqa: E402

MAX_LOGIT_DIFF = 1e-3
MAX_CAM_DIFF = 1e-3
MAX_FILE_MB = 95  # GitHub rejects files > 100 MB; the Vercel bundle has a size budget too
# ViT-B/16 exports to ~330 MB: too big for GitHub and for a serverless bundle.
DEFAULT_EXCLUDED = {
    "vit": "excluded from the web deployment: its ~330 MB ONNX file exceeds GitHub's 100 MB "
           "file limit and the serverless bundle budget (use --include-vit when self-hosting)",
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--out-dir", type=Path, default=ONNX_DIR)
    parser.add_argument("--check", action="store_true", help="Only check that exports are present and current")
    parser.add_argument("--no-verify", action="store_true", help="Skip the ONNX-vs-PyTorch parity check")
    parser.add_argument("--include-vit", action="store_true", help="Also export ViT-B/16 (~330 MB, self-hosting only)")
    parser.add_argument("--allow-large", action="store_true", help=f"Allow ONNX files above {MAX_FILE_MB} MB")
    args = parser.parse_args(argv)
    excluded = {} if args.include_vit else dict(DEFAULT_EXCLUDED)

    from src.onnx_export import build_manifest, export_checkpoint, stale_models

    manifest_path = args.out_dir / "manifest.json"
    if args.check:
        if not manifest_path.exists():
            print(f"No manifest at {manifest_path}; run: python scripts/export_onnx.py", file=sys.stderr)
            return 1
        manifest = json.loads(manifest_path.read_text())
        problems = stale_models(manifest, args.models_dir)
        missing = [k for k, e in manifest.get("models", {}).items() if not (args.out_dir / e["file"]).exists()]
        for key in missing:
            problems[key] = f"{manifest['models'][key]['file']} is missing"
        for key, problem in problems.items():
            print(f"  {key}: {problem}", file=sys.stderr)
        if problems:
            print("ONNX exports are out of date; run: python scripts/export_onnx.py", file=sys.stderr)
            return 1
        print(f"ONNX exports are current ({len(manifest.get('models', {}))} models).")
        return 0

    warnings.filterwarnings("ignore", category=UserWarning)
    entries = {}
    for key, spec in MODEL_SPECS.items():
        checkpoint = args.models_dir / spec["checkpoint"]
        if key in excluded:
            print(f"  {key}: {excluded[key]}")
            continue
        if not checkpoint.exists():
            print(f"  {key}: no checkpoint ({checkpoint.name}), skipped")
            continue
        entry = export_checkpoint(key, checkpoint, args.out_dir, verify=not args.no_verify)
        size_mb = (args.out_dir / entry["file"]).stat().st_size / 2**20
        if size_mb > MAX_FILE_MB and not args.allow_large:
            (args.out_dir / entry["file"]).unlink()
            print(f"  {key}: {size_mb:.0f} MB exceeds {MAX_FILE_MB} MB (GitHub/Vercel limits); "
                  "pass --allow-large to keep it", file=sys.stderr)
            return 1
        check = entry.get("verification", {})
        if check.get("max_abs_logit_diff", 0.0) > MAX_LOGIT_DIFF or check.get("max_abs_cam_diff", 0.0) > MAX_CAM_DIFF:
            print(f"  {key}: ONNX output differs from PyTorch: {check}", file=sys.stderr)
            return 1
        entries[key] = entry
        size = (args.out_dir / entry["file"]).stat().st_size / 2**20
        print(f"  {key}: {entry['file']} ({size:.1f} MB, Grad-CAM: {entry['supports_gradcam']}) {check}")

    if not entries:
        print("Nothing exported: no checkpoints found.", file=sys.stderr)
        return 1
    # Remove exports for models whose checkpoint disappeared.
    for stale in args.out_dir.glob("*.onnx"):
        if stale.name not in {e["file"] for e in entries.values()}:
            stale.unlink()
    manifest = build_manifest(entries)
    manifest["excluded"] = excluded
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
