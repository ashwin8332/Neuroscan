"""Shared fixtures: a tiny synthetic "MRI" dataset so tests never need real data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CLASS_NAMES  # noqa: E402


def make_scan(seed: int, cls_index: int, size=(160, 192), border: int = 20) -> Image.Image:
    """Dark background, bright elliptical 'head' with random low-frequency
    'anatomy' (so different scans hash differently, like real slices) and a
    class-dependent bright blob."""
    rng = np.random.default_rng(seed)
    h, w = size[1], size[0]
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2 + rng.uniform(-5, 5), w / 2 + rng.uniform(-5, 5)
    ry, rx = (h / 2 - border) * rng.uniform(0.9, 1.0), (w / 2 - border) * rng.uniform(0.85, 1.0)
    head = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
    anatomy = np.asarray(Image.fromarray(rng.normal(0, 1, (12, 12)).astype(np.float32), mode="F").resize((w, h), Image.BICUBIC))
    img = np.zeros((h, w), dtype=np.float32)
    img[head] = 120 + 55 * anatomy[head] + rng.normal(0, 6, head.sum())
    # class-specific bright blob at a class-specific position
    by = cy + [-0.3, 0.0, 0.0, 0.35][cls_index] * ry + rng.uniform(-6, 6)
    bx = cx + [-0.3, 0.35, 0.0, 0.0][cls_index] * rx + rng.uniform(-6, 6)
    if CLASS_NAMES[cls_index] != "notumor":
        blob = ((yy - by) ** 2 + (xx - bx) ** 2) <= (10 + rng.uniform(0, 6)) ** 2
        img[blob & head] = 235
    img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(img).convert("RGB")


def write_dataset(root: Path, per_class: int, seed_offset: int = 0, folder_names=None) -> Path:
    folder_names = folder_names or CLASS_NAMES
    for ci, folder in enumerate(folder_names):
        d = root / folder
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            make_scan(seed_offset + ci * 1000 + i, ci).save(d / f"img_{i:03d}.jpg", quality=95)
    return root


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("dataset")
    write_dataset(root / "Training", per_class=24)
    write_dataset(root / "Testing", per_class=6, seed_offset=50_000)
    # exact and mirrored duplicates inside Training/ (must be grouped together)
    src = root / "Training" / "glioma" / "img_000.jpg"
    Image.open(src).save(root / "Training" / "glioma" / "dup_copy.jpg", quality=95)
    Image.open(src).transpose(Image.FLIP_LEFT_RIGHT).save(root / "Training" / "glioma" / "dup_mirror.jpg", quality=95)
    # a test image leaked from Training/
    Image.open(root / "Training" / "pituitary" / "img_005.jpg").save(root / "Testing" / "pituitary" / "leaked.jpg", quality=95)
    return root


@pytest.fixture(scope="session")
def sample_image() -> Image.Image:
    return make_scan(7, 0)
