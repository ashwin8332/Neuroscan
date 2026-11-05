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
