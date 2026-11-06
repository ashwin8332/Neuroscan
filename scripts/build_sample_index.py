"""Write public/samples/index.json - the static list of sample scans used by the UI.

    python scripts/build_sample_index.py           # regenerate
    python scripts/build_sample_index.py --check   # exit 1 if the index is stale

The frontend reads this file from the CDN (no API call), fetches the chosen
image and uploads it like any other scan, so the serverless function does not
need the sample files. Run it after adding or removing files in public/samples/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "public" / "samples"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def build_index(samples_dir: Path = SAMPLES) -> dict:
    samples = []
    for cls in CLASSES:
        folder = samples_dir / cls
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                samples.append({
                    "id": f"{cls}/{path.name}",
                    "class": cls,
                    "filename": path.name,
                    "path": f"/samples/{cls}/{path.name}",
                })
    return {"samples": samples}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    expected = json.dumps(build_index(), indent=2) + "\n"
    target = SAMPLES / "index.json"
    if args.check:
        if not target.exists() or target.read_text() != expected:
            print(f"{target} is stale; run: python scripts/build_sample_index.py", file=sys.stderr)
            return 1
        print(f"{target} is current.")
        return 0
    target.write_text(expected)
    print(f"Wrote {target} ({len(json.loads(expected)['samples'])} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
