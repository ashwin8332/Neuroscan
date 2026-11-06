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

BBox = Tuple[int, int, int, int]

# 16/32-bit and float modes common in medical exports. Pillow's convert("RGB")
# clips them to 0-255, which turns a 12/16-bit MRI slice almost completely white.
HIGH_BIT_MODES = {"I;16", "I;16B", "I;16L", "I;16N", "I", "F"}


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


def to_rgb(img: Image.Image) -> Image.Image:
    """Convert any Pillow mode to 8-bit RGB without destroying the image.

    * 16/32-bit and float images are min-max scaled to 0-255 instead of clipped;
    * transparent images (RGBA, LA, PA, palette with transparency) are
      composited onto black, the background colour of an MRI slice;
    * everything else (L, RGB, P, CMYK, YCbCr, ...) uses Pillow's conversion,
      which keeps 8-bit inputs bit-identical to the training pipeline.
    """
    if img.mode in HIGH_BIT_MODES:
        arr = np.asarray(img, dtype=np.float64)
        finite = np.isfinite(arr)
        lo = float(arr[finite].min()) if finite.any() else 0.0
        hi = float(arr[finite].max()) if finite.any() else 0.0
        scaled = np.zeros_like(arr) if hi <= lo else (np.where(finite, arr, lo) - lo) * (255.0 / (hi - lo))
        img = Image.fromarray(np.clip(np.rint(scaled), 0, 255).astype(np.uint8))
    if img.mode in ("RGBA", "LA", "PA", "RGBa", "La") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, (0, 0, 0, 255)), rgba)
    return img if img.mode == "RGB" else img.convert("RGB")


# --------------------------------------------------------------------------- #
# NumPy re-implementations of the three scipy.ndimage operations used to find
# the head. They reproduce scipy's results exactly (see tests/test_preprocessing)
# so the serving bundle does not need SciPy (~110 MB) on Vercel.
# --------------------------------------------------------------------------- #
def gaussian_blur(arr: np.ndarray, sigma: float, truncate: float = 4.0) -> np.ndarray:
    """== scipy.ndimage.gaussian_filter(arr.astype(float32), sigma) (mode='reflect')."""
    radius = int(truncate * float(sigma) + 0.5)
    taps = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 / (float(sigma) ** 2) * taps ** 2)
    kernel /= kernel.sum()
    out = np.asarray(arr, dtype=np.float32)
    for axis in (0, 1):
        width = [(0, 0), (0, 0)]
        width[axis] = (radius, radius)
        padded = np.pad(out.astype(np.float64), width, mode="symmetric")
        acc = np.zeros(out.shape, dtype=np.float64)
        length = out.shape[axis]
        for i, weight in enumerate(kernel):
            window = [slice(None), slice(None)]
            window[axis] = slice(i, i + length)
            acc += weight * padded[tuple(window)]
        out = acc.astype(np.float32)  # scipy stores each 1-D pass in the input dtype
    return out


def _neighbourhood(mask: np.ndarray, reduce) -> np.ndarray:
    """3x3 neighbourhood AND/OR with pixels outside the image treated as False."""
    height, width = mask.shape
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    out = padded[1:height + 1, 1:width + 1].copy()
    for dy in range(3):
        for dx in range(3):
            if dy != 1 or dx != 1:
                reduce(out, padded[dy:dy + height, dx:dx + width], out=out)
    return out


def binary_opening(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """== scipy.ndimage.binary_opening(mask, structure=ones((3, 3)), iterations=iterations)."""
    out = np.asarray(mask, dtype=bool)
    for _ in range(iterations):
        out = _neighbourhood(out, np.logical_and)
    for _ in range(iterations):
        out = _neighbourhood(out, np.logical_or)
    return out


def largest_component(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Largest 4-connected component (scipy.ndimage.label's default connectivity).

    Union-find with min-index hooking and pointer jumping; ties go to the
    component that starts first in raster order, exactly like
    ``np.argmax(np.bincount(scipy.ndimage.label(mask)[0])[1:])``.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask), 0
    height, width = mask.shape
    index = np.arange(height * width, dtype=np.int64).reshape(height, width)
    horizontal = mask[:, :-1] & mask[:, 1:]
    vertical = mask[:-1, :] & mask[1:, :]
    a = np.concatenate([index[:, :-1][horizontal], index[:-1, :][vertical]])
    b = np.concatenate([index[:, 1:][horizontal], index[1:, :][vertical]])
    parent = index.ravel().copy()
    while a.size:
        root_a, root_b = parent[a], parent[b]
        differ = root_a != root_b
        if not differ.any():
            break
        np.minimum.at(parent, np.maximum(root_a[differ], root_b[differ]), np.minimum(root_a[differ], root_b[differ]))
        while True:  # pointer jumping until every node points at its root
            grand = parent[parent]
            if np.array_equal(grand, parent):
                break
            parent = grand
    roots = parent[index.ravel()[mask.ravel()]]
    counts = np.bincount(roots, minlength=height * width)
    best = int(np.argmax(counts))  # first maximum = lowest root = first in raster order
    component = (parent.reshape(height, width) == best) & mask
    return component, int(counts[best])


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
    gray_img = to_rgb(img).convert("L")
    width, height = gray_img.size
    if width < 8 or height < 8:
        return None
    scale = min(1.0, analysis_size / max(width, height))
    if scale < 1.0:
        gray_img = gray_img.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))), Image.BILINEAR
        )
    gray = np.asarray(gray_img, dtype=np.uint8)

    smooth = gaussian_blur(gray, sigma=1.5)
    threshold = float(np.clip(0.5 * _otsu_threshold(gray), 12.0, 60.0))
    mask = binary_opening(smooth > threshold, iterations=2)

    component, size = largest_component(mask)
    if size == 0 or size < min_area_frac * mask.size:
        return None

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
    img = to_rgb(img)
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
    rgb = np.asarray(to_rgb(img).resize((128, 128), Image.BILINEAR), dtype=np.float32)
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
