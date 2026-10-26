from pathlib import Path
from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from .config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    SEED,
    TEST_DIR,
    TRAIN_DIR,
)

# Standard ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_test_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Deterministic transforms for validation and testing."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Data augmentation transforms for training MRI scans."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def is_valid_image(path_str: str) -> bool:
    """Check if file exists, has valid image extension, and is not a tiny Git-LFS text pointer."""
    try:
        p = Path(path_str)
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            return False
        # Git LFS pointers are text files < 500 bytes
        if p.stat().st_size <= 500:
            return False
        return True
    except Exception:
        return False


def get_image_folder(path: Path, is_train: bool = False, image_size: int = IMAGE_SIZE) -> datasets.ImageFolder:
    """Load an ImageFolder dataset with appropriate transforms, filtering out LFS pointers."""
    transform = get_train_transforms(image_size) if is_train else get_test_transforms(image_size)
    return datasets.ImageFolder(root=str(path), transform=transform, is_valid_file=is_valid_image)


def get_data_loaders(
    train_dir: Path = TRAIN_DIR,
    test_dir: Optional[Path] = TEST_DIR,
    batch_size: int = BATCH_SIZE,
    val_ratio: float = 0.15,
    subset_fraction: float = 1.0,
    num_workers: int = 0,
    seed: int = SEED,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], list]:
    """
    Prepare Train, Validation, and Test DataLoaders.
    
    Args:
        train_dir: Path to training images folder
        test_dir: Path to test images folder (optional)
        batch_size: Mini-batch size
        val_ratio: Proportion of training data reserved for validation
        subset_fraction: Subsample fraction (0.0 < fraction <= 1.0) for fast testing
        num_workers: Number of DataLoader subprocesses (0 for Windows compatibility)
        seed: Random seed for reproducibility
    """
    base_train_dataset = datasets.ImageFolder(root=str(train_dir), is_valid_file=is_valid_image)
    class_names = base_train_dataset.classes
    total_samples = len(base_train_dataset)

    # Optional subset sampling for accelerated testing
    if 0.0 < subset_fraction < 1.0:
        num_keep = max(int(total_samples * subset_fraction), len(class_names) * 4)
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total_samples, generator=generator)[:num_keep].tolist()
        base_dataset = Subset(base_train_dataset, indices)
        total_samples = len(base_dataset)
    else:
        base_dataset = base_train_dataset

    # Split into train and validation
    val_size = int(total_samples * val_ratio)
    train_size = total_samples - val_size

    train_subset, val_subset = random_split(
        base_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # Wrap subsets with dedicated augmentations
    class TransformSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __getitem__(self, idx):
            x, y = self.subset[idx]
            # If underlying is ImageFolder, x is PIL Image; apply transform
            if not isinstance(x, torch.Tensor):
                x = self.transform(x)
            return x, y

        def __len__(self):
            return len(self.subset)

    train_data = TransformSubset(train_subset, get_train_transforms())
    val_data = TransformSubset(val_subset, get_test_transforms())

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = None
    if test_dir and Path(test_dir).exists():
        test_dataset = get_image_folder(Path(test_dir), is_train=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

    return train_loader, val_loader, test_loader, class_names
