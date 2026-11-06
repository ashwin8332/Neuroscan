import pytest

pytest.importorskip("torch")  # PyTorch tests are skipped in the serving-only (Vercel) environment

from pathlib import Path

import numpy as np

from src.config import CLASS_NAMES
from src.data import (
    audit_leakage,
    build_data,
    compute_image_hashes,
    group_near_duplicates,
    near_duplicate_pairs,
    scan_image_folder,
    stratified_group_split,
)
from tests.conftest import write_dataset


def test_scan_maps_alias_folder_names(tmp_path):
    write_dataset(tmp_path, per_class=2, folder_names=["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"])
    (tmp_path / "README.txt").write_text("not a class")
    samples = scan_image_folder(tmp_path)
    assert len(samples) == 8
    assert sorted({label for _, label in samples}) == [0, 1, 2, 3]


def test_lfs_pointer_files_are_ignored(tmp_path):
    write_dataset(tmp_path, per_class=1)
    (tmp_path / "glioma" / "pointer.jpg").write_text("version https://git-lfs.github.com/spec/v1\n")
    assert len(scan_image_folder(tmp_path)) == 4


def test_exact_and_mirrored_duplicates_share_a_group(synthetic_dataset):
    samples = scan_image_folder(synthetic_dataset / "Training")
    paths = [p for p, _ in samples]
    hashes = compute_image_hashes(paths)
    groups = group_near_duplicates(len(samples), near_duplicate_pairs(hashes[0], hashes[1], 24))
    index = {Path(p).name: i for i, p in enumerate(paths) if Path(p).parent.name == "glioma"}
    assert groups[index["img_000.jpg"]] == groups[index["dup_copy.jpg"]] == groups[index["dup_mirror.jpg"]]
    # different synthetic scans are not merged into one giant group
    assert np.bincount(groups).max() <= 3


def test_grouped_split_never_separates_duplicates(synthetic_dataset):
    samples = scan_image_folder(synthetic_dataset / "Training")
    train_idx, val_idx, report, hashes = stratified_group_split(samples, val_ratio=0.25, seed=0)
    assert set(train_idx).isdisjoint(val_idx)
    assert len(train_idx) + len(val_idx) == len(samples)
    groups = group_near_duplicates(len(samples), near_duplicate_pairs(hashes[0], hashes[1], 24))
    assert not set(groups[train_idx]) & set(groups[val_idx])
    assert report.grouped and report.duplicate_images >= 3
    assert all(count > 0 for count in report.val_class_counts)  # stratified: every class validated


def test_leakage_audit_detects_copied_test_image(synthetic_dataset):
    leak = audit_leakage(scan_image_folder(synthetic_dataset / "Training"), scan_image_folder(synthetic_dataset / "Testing"))
    assert leak["pituitary"] > 0
    assert leak["glioma"] == 0


def test_build_data_shapes(synthetic_dataset):
    bundle = build_data(train_dir=synthetic_dataset / "Training", test_dir=synthetic_dataset / "Testing",
                        batch_size=8, num_workers=0, cache=True, audit_test_leakage=True)
    images, labels = next(iter(bundle.train_loader))
    assert images.shape[1:] == (3, 224, 224) and images.dtype.is_floating_point
    assert labels.max() < len(CLASS_NAMES)
    assert bundle.split_report.test_leakage["overall"] > 0
    val_images, _ = next(iter(bundle.val_loader))
    assert val_images.shape[1:] == (3, 224, 224)
