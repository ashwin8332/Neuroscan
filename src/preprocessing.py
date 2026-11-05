"""Deterministic MRI preprocessing shared by training, evaluation and inference.

Why this exists
---------------
The Kaggle "Brain Tumor MRI" dataset is stitched together from three sources
(figshare, SARTAJ, Br35H) and each class comes mostly from one of them. The
sources differ in zoom level, black-border width and resolution, so a network
can reach high *validation* accuracy by learning "how is this image framed"
instead of anatomy - a shortcut that does not survive a new scanner and is one
reason the original models scored much higher on validation than on test.

New checkpoints therefore use: crop to the head -> pad to square (keeps the
aspect ratio, the old pipeline stretched every scan) -> resize. Checkpoints
trained before this module existed keep their original stretch-resize via
``LEGACY_PREPROCESS`` so their accuracy is unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int = 224
    crop_brain: bool = True
    pad_to_square: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PreprocessConfig":
        """Build from checkpoint metadata. Missing metadata means a legacy checkpoint."""
        if not data:
            return LEGACY_PREPROCESS
        fields = {k: data[k] for k in ("image_size", "crop_brain", "pad_to_square") if k in data}
        return cls(**fields)


LEGACY_PREPROCESS = PreprocessConfig(image_size=224, crop_brain=False, pad_to_square=False)
DEFAULT_PREPROCESS = PreprocessConfig()


def _otsu_threshold(gray: np.ndarray) -> float:
    """Otsu's threshold for a uint8 image (numpy only, no OpenCV dependency)."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    levels = np.arange(256, dtype=np.float64)
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    cum_mean = np.cumsum(hist * levels)
    valid = (weight_bg > 0) & (weight_fg > 0)
    between = np.zeros(256)
    mean_bg = cum_mean[valid] / weight_bg[valid]
    mean_fg = (cum_mean[-1] - cum_mean[valid]) / weight_fg[valid]
    between[valid] = weight_bg[valid] * weight_fg[valid] * (mean_bg - mean_fg) ** 2
    return float(np.argmax(between))


def find_brain_bbox(
    img: Image.Image,
    analysis_size: int = 256,
    min_area_frac: float = 0.05,
) -> Optional[BBox]:
    """Locate the head as the largest bright connected region.

    Returns ``(left, upper, right, lower)`` in original pixel coordinates, or
    ``None`` when no plausible head region is found (the caller then keeps the
    full image, so preprocessing never fails).
    """
    gray_img = img.convert("L")
    width, height = gray_img.size
    if width < 8 or height < 8:
        return None
    scale = min(1.0, analysis_size / max(width, height))
    if scale < 1.0:
        gray_img = gray_img.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))), Image.BILINEAR
        )
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

    component = labels == largest
    rows = np.flatnonzero(component.any(axis=1))
    cols = np.flatnonzero(component.any(axis=0))
    inv = 1.0 / scale
    left = int(np.floor(cols[0] * inv))
    upper = int(np.floor(rows[0] * inv))
    right = min(width, int(np.ceil((cols[-1] + 1) * inv)))
    lower = min(height, int(np.ceil((rows[-1] + 1) * inv)))
    return left, upper, right, lower


def crop_to_brain(img: Image.Image, margin_frac: float = 0.03) -> Image.Image:
    bbox = find_brain_bbox(img)
    if bbox is None:
        return img
    left, upper, right, lower = bbox
    margin = int(round(margin_frac * max(right - left, lower - upper)))
    return img.crop(
        (
            max(0, left - margin),
            max(0, upper - margin),
            min(img.width, right + margin),
            min(img.height, lower + margin),
        )
    )


def pad_to_square(img: Image.Image) -> Image.Image:
    width, height = img.size
    if width == height:
        return img
    side = max(width, height)
    canvas = Image.new(img.mode, (side, side), 0)
    canvas.paste(img, ((side - width) // 2, (side - height) // 2))
    return canvas


def preprocess_image(
    img: Image.Image,
    cfg: PreprocessConfig = DEFAULT_PREPROCESS,
    output_size: Optional[int] = None,
) -> Image.Image:
    """Apply the deterministic part of the pipeline and return an RGB PIL image."""
    size = output_size or cfg.image_size
    img = img.convert("RGB")
    if cfg.crop_brain:
        img = crop_to_brain(img)
    if cfg.pad_to_square:
        img = pad_to_square(img)
    # Bilinear PIL resize == torchvision.transforms.Resize on PIL input, which keeps
    # legacy checkpoints bit-for-bit compatible with how they were trained.
    return img.resize((size, size), Image.BILINEAR)


class MRIPreprocess:
    """Picklable ``PIL -> PIL`` transform usable inside ``transforms.Compose``."""

    def __init__(self, cfg: PreprocessConfig = DEFAULT_PREPROCESS, output_size: Optional[int] = None):
        self.cfg = cfg
        self.output_size = output_size or cfg.image_size

    def __call__(self, img: Image.Image) -> Image.Image:
        return preprocess_image(img, self.cfg, self.output_size)

    def __repr__(self) -> str:
        return f"MRIPreprocess(cfg={self.cfg}, output_size={self.output_size})"


# --------------------------------------------------------------------------- #
# Input sanity checks used by the web app before running the models
# --------------------------------------------------------------------------- #
def assess_image_quality(img: Image.Image) -> Dict[str, Any]:
    """Cheap heuristics that flag inputs that are probably not a brain MRI slice.

    The classifiers will happily assign a tumor class to a photo of a cat, so
    the UI must warn when the input is out of distribution.
    """
    warnings: List[str] = []
    width, height = img.size
    rgb = np.asarray(img.convert("RGB").resize((128, 128), Image.BILINEAR), dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    colourfulness = float((np.abs(r - g) + np.abs(g - b) + np.abs(r - b)).mean() / 3.0)
    gray = rgb.mean(axis=2)
    contrast = float(gray.std())

    if min(width, height) < 96:
        warnings.append(f"Very low resolution ({width}x{height}px); results are unreliable.")
    if colourfulness > 12.0:
        warnings.append("The image contains colour. MRI slices are greyscale - is this a photo or screenshot?")
    if contrast < 12.0:
        warnings.append("The image has almost no contrast.")
    if find_brain_bbox(img) is None:
        warnings.append("Could not find a head/brain region in the image.")

    return {
        "width": width,
        "height": height,
        "colourfulness": round(colourfulness, 2),
        "contrast": round(contrast, 2),
        "looks_like_mri": not warnings,
        "warnings": warnings,
    }
