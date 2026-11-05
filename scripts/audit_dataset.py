"""Audit a brain-MRI dataset for the problems that inflate validation/test scores.

    python scripts/audit_dataset.py --data-dir Dataset/brain-tumor-mri-dataset

Reports (and writes results/dataset_audit.json):
* class balance and image-size spread per split;
* near-duplicate groups inside Training/ (augmented copies, re-scaled scans,
  adjacent slices) - these leak into a random validation split;
* duplicate groups whose members carry *different* labels (label noise);
* the share of Testing/ images that have a near-copy in Training/.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CLASS_NAMES, DATA_DIR, RESULTS_DIR  # noqa: E402
from src.data import compute_image_hashes, group_near_duplicates, near_duplicate_pairs, scan_image_folder  # noqa: E402


def size_summary(samples):
    sizes = Counter()
    for path, _ in samples:
        with Image.open(path) as img:
            sizes[img.size] += 1
    return {"distinct_sizes": len(sizes), "most_common": [[f"{w}x{h}", n] for (w, h), n in sizes.most_common(5)]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--hash-threshold", type=int, default=24, help="Max differing bits of 256 to call two images near-duplicates")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "dataset_audit.json")
    args = parser.parse_args(argv)

    train_dir, test_dir = args.data_dir / "Training", args.data_dir / "Testing"
    if not train_dir.is_dir():
        print(f"Training folder not found: {train_dir}", file=sys.stderr)
        return 2

    report = {"data_dir": str(args.data_dir), "hash_threshold": args.hash_threshold}
    train = scan_image_folder(train_dir)
    train_labels = np.array([label for _, label in train])
    report["train_counts"] = dict(zip(CLASS_NAMES, np.bincount(train_labels, minlength=len(CLASS_NAMES)).tolist()))
    report["train_sizes"] = size_summary(train)
    print(f"Training: {len(train)} images {report['train_counts']}")

    hashes = compute_image_hashes([p for p, _ in train])
    pairs = near_duplicate_pairs(hashes[0], hashes[1], args.hash_threshold)
    groups = group_near_duplicates(len(train), pairs)
    sizes = np.bincount(groups)
    dup_groups = np.flatnonzero(sizes > 1)
    conflicts = [g for g in dup_groups if len(set(train_labels[groups == g])) > 1]
    report["train_duplicates"] = {
        "images_in_duplicate_groups": int((sizes[groups] > 1).sum()),
        "share": round(float((sizes[groups] > 1).mean()), 4),
        "groups": int(len(dup_groups)),
        "largest_group": int(sizes.max()),
        "label_conflict_groups": len(conflicts),
        "label_conflict_examples": [
            [Path(train[i][0]).relative_to(train_dir).as_posix() for i in np.flatnonzero(groups == g)][:6]
            for g in conflicts[:10]
        ],
    }
    print(f"  near-duplicates: {report['train_duplicates']['images_in_duplicate_groups']} images "
          f"({report['train_duplicates']['share']:.1%}) in {len(dup_groups)} groups; "
          f"{len(conflicts)} groups with conflicting labels")

    if test_dir.is_dir():
        test = scan_image_folder(test_dir)
        test_labels = np.array([label for _, label in test])
        report["test_counts"] = dict(zip(CLASS_NAMES, np.bincount(test_labels, minlength=len(CLASS_NAMES)).tolist()))
        report["test_sizes"] = size_summary(test)
        test_hashes = compute_image_hashes([p for p, _ in test])
        leaked = np.zeros(len(test), dtype=bool)
        cross_label = 0
        for i, j in near_duplicate_pairs(test_hashes[0], test_hashes[1], args.hash_threshold, other=hashes[0]):
            leaked[i] = True
            cross_label += int(test_labels[i] != train_labels[j])
        report["test_leakage"] = {"overall": round(float(leaked.mean()), 4)} | {
            name: round(float(leaked[test_labels == k].mean()), 4)
            for k, name in enumerate(CLASS_NAMES) if (test_labels == k).any()
        }
        report["test_leakage"]["pairs_with_different_labels"] = cross_label
        print(f"Testing: {len(test)} images {report['test_counts']}")
        print(f"  test images with a near-copy in Training/: {report['test_leakage']}")
        if leaked.mean() > 0.05:
            print("  WARNING: test accuracy is optimistic. Report results on the leakage-free subset too:\n"
                  "           python evaluate.py --exclude-leaked")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
