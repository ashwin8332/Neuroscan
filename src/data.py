"""Dataset discovery, leak-free splitting and DataLoader construction.

Key fixes compared with the original pipeline
--------------------------------------------
1. **Duplicate-aware validation split.** The training folder contains many
   near-identical images (augmented copies, adjacent slices, the same scan at
   two resolutions). ``random_split`` scattered those across train/val, so
   validation accuracy measured memorisation, early stopping/model selection
   picked over-fitted checkpoints, and test accuracy came out far lower. We
   group near-duplicates with a perceptual hash and split by group
   (``StratifiedGroupKFold``), keeping the class balance.
2. **Shortcut-resistant preprocessing** (see ``src.preprocessing``).
3. **Stronger, anatomy-preserving augmentation** to fight over-fitting.
4. **In-memory cache** of preprocessed images so CPU training is not
   bottlenecked by JPEG decoding.
"""
from __future__ import annotations

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import BATCH_SIZE, CLASS_ALIASES, CLASS_NAMES, IMAGE_SIZE, SEED, TEST_DIR, TRAIN_DIR
from .preprocessing import DEFAULT_PREPROCESS, MRIPreprocess, PreprocessConfig, preprocess_image

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# Training images are cached slightly larger so RandomResizedCrop has room to zoom.
TRAIN_RESIZE_RATIO = 256 / 224

Sample = Tuple[str, int]


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def is_valid_image(path_str: str) -> bool:
    """Image extension and not a tiny Git-LFS pointer file."""
    try:
        path = Path(path_str)
        return path.suffix.lower() in IMAGE_EXTENSIONS and path.stat().st_size > 500
    except OSError:
        return False


def scan_image_folder(root: Path, class_names: Sequence[str] = CLASS_NAMES) -> List[Sample]:
    """List ``(path, label)`` pairs from ``root/<class>/*``.

    Folder names are mapped through ``CLASS_ALIASES`` so the Kaggle, SARTAJ and
    Br35H layouts (``glioma`` / ``glioma_tumor`` / ``no_tumor`` ...) all work.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    samples: List[Sample] = []
    seen_classes = set()
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        canonical = CLASS_ALIASES.get(child.name.lower())
        if canonical is None or canonical not in class_to_idx:
            logger.warning("Skipping unknown class folder: %s", child)
            continue
        seen_classes.add(canonical)
        for file in sorted(child.rglob("*")):
            if file.is_file() and is_valid_image(str(file)):
                samples.append((str(file), class_to_idx[canonical]))
    missing = set(class_names) - seen_classes
    if missing:
        logger.warning("No folder found for classes %s under %s", sorted(missing), root)
    if not samples:
        raise RuntimeError(f"No images found under {root}")
    return samples


def class_counts(samples: Sequence[Sample], num_classes: int = len(CLASS_NAMES)) -> List[int]:
    return np.bincount([label for _, label in samples], minlength=num_classes).tolist()


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def _normalize() -> List:
    return [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]


def train_resolution(image_size: int = IMAGE_SIZE) -> int:
    return int(round(image_size * TRAIN_RESIZE_RATIO))


def get_train_augmentation(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Augmentation applied to an already preprocessed (cropped, squared) image.

    Anatomy-preserving only: left/right flips are plausible, up/down flips are
    not, and strong colour changes would destroy MRI contrast information.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.70, 1.0), ratio=(0.85, 1.18)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.RandomAffine(degrees=12, translate=(0.04, 0.04), shear=4)], p=0.6
            ),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.8),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.25),
            *_normalize(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.08), value=0),
        ]
    )


def get_eval_tensorize() -> transforms.Compose:
    return transforms.Compose(_normalize())


def get_test_transforms(
    image_size: int = IMAGE_SIZE, preprocess: PreprocessConfig = DEFAULT_PREPROCESS
) -> transforms.Compose:
    """Raw PIL image -> normalised tensor, deterministic."""
    return transforms.Compose([MRIPreprocess(preprocess, image_size), *_normalize()])


def get_train_transforms(
    image_size: int = IMAGE_SIZE, preprocess: PreprocessConfig = DEFAULT_PREPROCESS
) -> transforms.Compose:
    """Raw PIL image -> augmented tensor (uncached path)."""
    return transforms.Compose(
        [MRIPreprocess(preprocess, train_resolution(image_size)), get_train_augmentation(image_size)]
    )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class MRIDataset(Dataset):
    """Brain-MRI dataset with deterministic preprocessing and optional caching.

    With ``cache=True`` every image is decoded and preprocessed once into a
    single contiguous uint8 array. DataLoader workers forked from the main
    process share it copy-on-write, so the cache costs memory only once.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        transform: transforms.Compose,
        preprocess: PreprocessConfig = DEFAULT_PREPROCESS,
        resolution: int = IMAGE_SIZE,
        cache: bool = False,
        class_names: Sequence[str] = CLASS_NAMES,
    ):
        self.samples = list(samples)
        self.targets = [label for _, label in self.samples]
        self.transform = transform
        self.preprocess = preprocess
        self.resolution = resolution
        self.classes = list(class_names)
        self._cache: Optional[np.ndarray] = None
        if cache:
            self.warm_cache()

    def _load(self, index: int) -> Image.Image:
        path = self.samples[index][0]
        with Image.open(path) as img:
            return preprocess_image(img, self.preprocess, self.resolution)

    def warm_cache(self, max_workers: Optional[int] = None) -> None:
        if self._cache is not None:
            return
        cache = np.empty((len(self.samples), self.resolution, self.resolution, 3), dtype=np.uint8)

        def fill(index: int) -> None:
            cache[index] = np.asarray(self._load(index), dtype=np.uint8)

        with ThreadPoolExecutor(max_workers=max_workers or min(8, os.cpu_count() or 4)) as pool:
            list(pool.map(fill, range(len(self.samples))))
        self._cache = cache

    def get_image(self, index: int) -> Image.Image:
        if self._cache is not None:
            return Image.fromarray(self._cache[index])
        return self._load(index)

    def __getitem__(self, index: int):
        return self.transform(self.get_image(index)), self.samples[index][1]

    def __len__(self) -> int:
        return len(self.samples)


# --------------------------------------------------------------------------- #
# Near-duplicate detection & leak-free splitting
# --------------------------------------------------------------------------- #
def _dhash_bits(img: Image.Image, hash_size: int = 16) -> np.ndarray:
    gray = np.asarray(img.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR), dtype=np.int16)
    return np.packbits((gray[:, 1:] > gray[:, :-1]).ravel())


def compute_image_hashes(
    paths: Sequence[str],
    preprocess: PreprocessConfig = DEFAULT_PREPROCESS,
    hash_size: int = 16,
    max_workers: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Difference hashes of each image and of its mirror image (packed uint8)."""

    def one(path: str) -> Tuple[np.ndarray, np.ndarray]:
        with Image.open(path) as img:
            prepared = preprocess_image(img, preprocess, 64)
        return _dhash_bits(prepared, hash_size), _dhash_bits(
            prepared.transpose(Image.FLIP_LEFT_RIGHT), hash_size
        )

    with ThreadPoolExecutor(max_workers=max_workers or min(8, os.cpu_count() or 4)) as pool:
        results = list(pool.map(one, paths))
    return np.stack([r[0] for r in results]), np.stack([r[1] for r in results])


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between packed hash rows: [len(a), len(b)]."""
    xor = a[:, None, :] ^ b[None, :, :]
    return _POPCOUNT[xor].sum(axis=2, dtype=np.int32)


def near_duplicate_pairs(
    hashes: np.ndarray,
    mirror_hashes: np.ndarray,
    threshold: int,
    other: Optional[np.ndarray] = None,
    chunk: int = 256,
) -> List[Tuple[int, int]]:
    """Index pairs whose (flip-invariant) hash distance is <= ``threshold``.

    Without ``other`` the pairs are within ``hashes`` (i < j); with ``other``
    they are (i in hashes, j in other).
    """
    target = hashes if other is None else other
    pairs: List[Tuple[int, int]] = []
    for start in range(0, len(hashes), chunk):
        stop = min(start + chunk, len(hashes))
        dist = np.minimum(_hamming(hashes[start:stop], target), _hamming(mirror_hashes[start:stop], target))
        rows, cols = np.nonzero(dist <= threshold)
        for r, c in zip(rows.tolist(), cols.tolist()):
            i = start + r
            if other is not None or c > i:
                pairs.append((i, c))
    return pairs


def group_near_duplicates(num_items: int, pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Union-find connected components -> group id per item."""
    parent = list(range(num_items))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = [find(i) for i in range(num_items)]
    _, group_ids = np.unique(roots, return_inverse=True)
    return group_ids


@dataclass
class SplitReport:
    num_train: int
    num_val: int
    num_groups: int
    largest_group: int
    duplicate_images: int
    train_class_counts: List[int]
    val_class_counts: List[int]
    hash_threshold: int
    grouped: bool
    test_leakage: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return dict(self.__dict__)


def stratified_group_split(
    samples: Sequence[Sample],
    val_ratio: float = 0.15,
    seed: int = SEED,
    hash_threshold: int = 24,
    grouped: bool = True,
    preprocess: PreprocessConfig = DEFAULT_PREPROCESS,
) -> Tuple[List[int], List[int], SplitReport, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Split indices into train/val so near-duplicates never straddle the split.

    ``hash_threshold`` is in bits of a 256-bit difference hash (~9%); it was
    chosen so augmented copies and re-scaled duplicates match while different
    patients do not (see ``scripts/audit_dataset.py``).
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

    labels = np.array([label for _, label in samples])
    hashes = None
    if grouped:
        hashes = compute_image_hashes([p for p, _ in samples], preprocess)
        pairs = near_duplicate_pairs(hashes[0], hashes[1], hash_threshold)
        groups = group_near_duplicates(len(samples), pairs)
        n_splits = max(2, int(round(1.0 / val_ratio)))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels, groups))
    else:
        groups = np.arange(len(samples))
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))

    group_sizes = np.bincount(groups)
    report = SplitReport(
        num_train=len(train_idx),
        num_val=len(val_idx),
        num_groups=int(len(group_sizes)),
        largest_group=int(group_sizes.max()),
        duplicate_images=int((group_sizes[groups] > 1).sum()),
        train_class_counts=np.bincount(labels[train_idx], minlength=len(CLASS_NAMES)).tolist(),
        val_class_counts=np.bincount(labels[val_idx], minlength=len(CLASS_NAMES)).tolist(),
        hash_threshold=hash_threshold,
        grouped=grouped,
    )
    return sorted(train_idx.tolist()), sorted(val_idx.tolist()), report, hashes


def audit_leakage(
    train_samples: Sequence[Sample],
    test_samples: Sequence[Sample],
    hash_threshold: int = 24,
    preprocess: PreprocessConfig = DEFAULT_PREPROCESS,
    train_hashes: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict[str, float]:
    """Fraction of test images (overall and per class) that have a near-duplicate in train."""
    if train_hashes is None:
        train_hashes = compute_image_hashes([p for p, _ in train_samples], preprocess)
    test_hashes = compute_image_hashes([p for p, _ in test_samples], preprocess)
    pairs = near_duplicate_pairs(test_hashes[0], test_hashes[1], hash_threshold, other=train_hashes[0])
    leaked = np.zeros(len(test_samples), dtype=bool)
    for i, _ in pairs:
        leaked[i] = True
    labels = np.array([label for _, label in test_samples])
    result = {"overall": round(float(leaked.mean()), 4)}
    for idx, name in enumerate(CLASS_NAMES):
        mask = labels == idx
        if mask.any():
            result[name] = round(float(leaked[mask].mean()), 4)
    return result


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def default_num_workers() -> int:
    # Windows uses "spawn": every worker would re-pickle the dataset, which is
    # slower than loading from the in-memory cache in the main process.
    if sys.platform.startswith("win"):
        return 0
    return min(4, os.cpu_count() or 1)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random

    random.seed(worker_seed)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int = SEED) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=generator,
        drop_last=False,
    )


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: Optional[DataLoader]
    class_names: List[str]
    split_report: SplitReport


def build_data(
    train_dir: Path = TRAIN_DIR,
    test_dir: Optional[Path] = TEST_DIR,
    batch_size: int = BATCH_SIZE,
    val_ratio: float = 0.15,
    subset_fraction: float = 1.0,
    num_workers: Optional[int] = None,
    seed: int = SEED,
    preprocess: PreprocessConfig = DEFAULT_PREPROCESS,
    grouped_split: bool = True,
    hash_threshold: int = 24,
    cache: bool = True,
    audit_test_leakage: bool = False,
) -> DataBundle:
    """Build train/val(/test) loaders with a leak-free split."""
    samples = scan_image_folder(Path(train_dir))
    if 0.0 < subset_fraction < 1.0:
        from sklearn.model_selection import train_test_split

        labels = [label for _, label in samples]
        samples, _ = train_test_split(
            samples, train_size=subset_fraction, stratify=labels, random_state=seed
        )
        samples = sorted(samples)

    train_idx, val_idx, report, hashes = stratified_group_split(
        samples, val_ratio=val_ratio, seed=seed, hash_threshold=hash_threshold,
        grouped=grouped_split, preprocess=preprocess,
    )
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    workers = default_num_workers() if num_workers is None else num_workers
    size = preprocess.image_size

    train_ds = MRIDataset(
        train_samples, get_train_augmentation(size), preprocess, train_resolution(size), cache=cache
    )
    val_ds = MRIDataset(val_samples, get_eval_tensorize(), preprocess, size, cache=cache)

    test_loader = None
    if test_dir and Path(test_dir).is_dir():
        test_samples = scan_image_folder(Path(test_dir))
        test_ds = MRIDataset(test_samples, get_eval_tensorize(), preprocess, size, cache=cache)
        test_loader = make_loader(test_ds, batch_size, False, workers, seed)
        if audit_test_leakage:
            full_hashes = None
            if hashes is not None and len(samples) == len(train_samples) + len(val_samples):
                full_hashes = hashes
            report.test_leakage = audit_leakage(
                samples, test_samples, hash_threshold, preprocess, train_hashes=full_hashes
            )

    return DataBundle(
        train_loader=make_loader(train_ds, batch_size, True, workers, seed),
        val_loader=make_loader(val_ds, batch_size, False, workers, seed),
        test_loader=test_loader,
        class_names=list(CLASS_NAMES),
        split_report=report,
    )


# --------------------------------------------------------------------------- #
# Backwards-compatible helpers (used by notebooks / older scripts)
# --------------------------------------------------------------------------- #
def get_image_folder(
    path: Path,
    is_train: bool = False,
    image_size: int = IMAGE_SIZE,
    preprocess: PreprocessConfig = DEFAULT_PREPROCESS,
    cache: bool = False,
) -> MRIDataset:
    samples = scan_image_folder(Path(path))
    if is_train:
        return MRIDataset(samples, get_train_augmentation(image_size), preprocess,
                          train_resolution(image_size), cache=cache)
    return MRIDataset(samples, get_eval_tensorize(), preprocess, image_size, cache=cache)


def get_data_loaders(
    train_dir: Path = TRAIN_DIR,
    test_dir: Optional[Path] = TEST_DIR,
    batch_size: int = BATCH_SIZE,
    val_ratio: float = 0.15,
    subset_fraction: float = 1.0,
    num_workers: Optional[int] = None,
    seed: int = SEED,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], list]:
    bundle = build_data(
        train_dir=train_dir, test_dir=test_dir, batch_size=batch_size, val_ratio=val_ratio,
        subset_fraction=subset_fraction, num_workers=num_workers, seed=seed, **kwargs,
    )
    return bundle.train_loader, bundle.val_loader, bundle.test_loader, bundle.class_names
