import pytest

pytest.importorskip("torch")  # PyTorch tests are skipped in the serving-only (Vercel) environment

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.data import get_test_transforms
from src.preprocessing import (
    DEFAULT_PREPROCESS,
    LEGACY_PREPROCESS,
    PreprocessConfig,
    assess_image_quality,
    crop_to_brain,
    find_brain_bbox,
    pad_to_square,
    preprocess_image,
)
from tests.conftest import make_scan


def test_bbox_finds_head_and_crop_removes_border():
    img = make_scan(1, 0, size=(200, 240), border=40)
    left, upper, right, lower = find_brain_bbox(img)
    assert 30 <= left <= 50 and 30 <= upper <= 50
    assert 150 <= right <= 170 and 190 <= lower <= 210
    cropped = crop_to_brain(img)
    assert cropped.width < img.width and cropped.height < img.height


def test_blank_image_falls_back_to_full_frame():
    blank = Image.new("RGB", (128, 128), 0)
    assert find_brain_bbox(blank) is None
    assert crop_to_brain(blank).size == (128, 128)


def test_pad_to_square_keeps_aspect_ratio():
    img = Image.new("RGB", (100, 50), (255, 255, 255))
    squared = pad_to_square(img)
    assert squared.size == (100, 100)
    arr = np.asarray(squared)
    assert arr[0, 0].sum() == 0 and arr[50, 50].sum() == 765  # black padding, content centred


def test_preprocess_output_size_and_mode():
    out = preprocess_image(make_scan(2, 1).convert("L"), DEFAULT_PREPROCESS)
    assert out.size == (224, 224) and out.mode == "RGB"


def test_legacy_pipeline_matches_original_torchvision_resize():
    """Old checkpoints must see exactly the preprocessing they were trained with."""
    img = make_scan(3, 2, size=(180, 210))
    original = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    assert torch.allclose(get_test_transforms(224, LEGACY_PREPROCESS)(img), original)


def test_config_roundtrip_and_legacy_default():
    cfg = PreprocessConfig(image_size=192, crop_brain=True, pad_to_square=False)
    assert PreprocessConfig.from_dict(cfg.to_dict()) == cfg
    assert PreprocessConfig.from_dict(None) == LEGACY_PREPROCESS


def test_quality_check_flags_colour_photos_but_not_scans():
    assert assess_image_quality(make_scan(4, 0))["looks_like_mri"]
    rng = np.random.default_rng(0)
    photo = Image.fromarray(rng.integers(0, 255, (200, 200, 3), dtype=np.uint8))
    report = assess_image_quality(photo)
    assert not report["looks_like_mri"]
    assert any("colour" in w for w in report["warnings"])


# --------------------------------------------------------------------------- #
# The NumPy re-implementations must reproduce scipy.ndimage exactly: models
# were trained with the scipy version, and the deployed app has no SciPy.
# --------------------------------------------------------------------------- #
def _scipy_bbox(img, analysis_size=256, min_area_frac=0.05):
    from scipy import ndimage

    from src.preprocessing import _otsu_threshold

    gray_img = img.convert("L")
    width, height = gray_img.size
    scale = min(1.0, analysis_size / max(width, height))
    if scale < 1.0:
        gray_img = gray_img.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.BILINEAR)
    gray = np.asarray(gray_img, dtype=np.uint8)
    smooth = ndimage.gaussian_filter(gray.astype(np.float32), sigma=1.5)
    threshold = float(np.clip(0.5 * _otsu_threshold(gray), 12.0, 60.0))
    mask = ndimage.binary_opening(smooth > threshold, structure=np.ones((3, 3)), iterations=2)
    labels, num = ndimage.label(mask)
    if num == 0:
        return None
    sizes = np.bincount(labels.ravel())[1:]
    largest = int(np.argmax(sizes)) + 1
    if sizes[largest - 1] < min_area_frac * mask.size:
        return None
    comp = labels == largest
    rows, cols, inv = np.flatnonzero(comp.any(1)), np.flatnonzero(comp.any(0)), 1.0 / scale
    return (int(np.floor(cols[0] * inv)), int(np.floor(rows[0] * inv)),
            min(width, int(np.ceil((cols[-1] + 1) * inv))), min(height, int(np.ceil((rows[-1] + 1) * inv))))


def test_numpy_morphology_matches_scipy():
    ndimage = pytest.importorskip("scipy.ndimage")
    from src.preprocessing import binary_opening, gaussian_blur, largest_component

    rng = np.random.default_rng(0)
    for _ in range(100):
        shape = (int(rng.integers(5, 80)), int(rng.integers(5, 80)))
        arr = rng.integers(0, 256, shape).astype(np.uint8)
        assert np.array_equal(gaussian_blur(arr, 1.5), ndimage.gaussian_filter(arr.astype(np.float32), 1.5))
        mask = rng.random(shape) < rng.uniform(0.2, 0.8)
        assert np.array_equal(binary_opening(mask, 2),
                              ndimage.binary_opening(mask, structure=np.ones((3, 3)), iterations=2))
        labels, num = ndimage.label(mask)
        component, size = largest_component(mask)
        if num:
            sizes = np.bincount(labels.ravel())[1:]
            assert size == sizes.max() and np.array_equal(component, labels == np.argmax(sizes) + 1)
        else:
            assert size == 0 and not component.any()


def test_head_crop_matches_scipy_on_real_and_synthetic_scans():
    pytest.importorskip("scipy")
    from pathlib import Path

    paths = sorted(Path(__file__).resolve().parents[1].glob("public/samples/*/*.jpg"))
    images = [Image.open(p) for p in paths] + [make_scan(s, s % 4, size=(150 + s, 190 - s)) for s in range(20)]
    images += [Image.new("L", (64, 64), 0), Image.new("L", (64, 64), 200)]
    for img in images:
        assert find_brain_bbox(img) == _scipy_bbox(img)
