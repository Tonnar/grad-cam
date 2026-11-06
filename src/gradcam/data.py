"""Image folders, a leak-free train/validation split, and preprocessing.

Expected layout: one sub-folder per class, ``root/<class>/<image>.jpg``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def open_rgb(path: str | Path) -> Image.Image:
    """Open an image upright: phone photos are often stored sideways with an
    EXIF orientation flag, which PIL (unlike most viewers) ignores by default."""
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def train_transform(size: int = 224) -> transforms.Compose:
    """Random augmentation, applied on the fly to training images only."""
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(size * 256 / 224)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def stratified_split(labels: list[int], val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split sample indices so each class keeps roughly ``val_fraction`` for validation."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        n_val = max(1, int(round(len(idx) * val_fraction))) if len(idx) > 1 else 0
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return np.sort(train_idx), np.sort(val_idx)


class _Folder(datasets.ImageFolder):
    """ImageFolder that can skip some class folders (e.g. personal test photos)."""

    def __init__(self, root, exclude: tuple[str, ...] = (), **kw):
        self._exclude = set(exclude)
        super().__init__(root, loader=open_rgb, **kw)

    def find_classes(self, directory):
        classes = sorted(
            e.name for e in Path(directory).iterdir()
            if e.is_dir() and not e.name.startswith(".") and e.name not in self._exclude
        )
        if not classes:
            raise FileNotFoundError(f"no class folders in {directory}")
        return classes, {c: i for i, c in enumerate(classes)}


def make_loaders(root: str | Path, val_fraction: float = 0.2, batch_size: int = 32, size: int = 224,
                 seed: int = 0, exclude: tuple[str, ...] = (), num_workers: int = 2):
    """Return ``(train_loader, val_loader, class_names)``.

    The split is made on the original files *before* any augmentation, so
    augmented copies of a validation image can never appear in training.
    """
    train_view = _Folder(root, exclude, transform=train_transform(size))
    val_view = _Folder(root, exclude, transform=eval_transform(size))
    train_idx, val_idx = stratified_split(train_view.targets, val_fraction, seed)
    train_loader = DataLoader(Subset(train_view, train_idx), batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(Subset(val_view, val_idx), batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_view.classes


def load_image(path: str | Path, size: int = 224) -> tuple[torch.Tensor, np.ndarray]:
    """Load one image for inference: (1, 3, size, size) tensor and the RGB array
    of the same centre crop (for drawing overlays)."""
    img = open_rgb(path)
    tf = eval_transform(size)
    crop = transforms.Compose(tf.transforms[:2])(img)
    return tf(img).unsqueeze(0), np.asarray(crop)
