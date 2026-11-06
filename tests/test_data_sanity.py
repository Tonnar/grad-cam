import numpy as np
import pytest
import torch
from PIL import Image

from gradcam.data import load_image, make_loaders, open_rgb, stratified_split
from gradcam.sanity import randomization_test, spearman


def test_split_is_disjoint_and_stratified():
    labels = [0] * 10 + [1] * 20 + [2] * 5
    tr, va = stratified_split(labels, 0.2, seed=0)
    assert not set(tr) & set(va) and len(tr) + len(va) == len(labels)
    va_labels = np.asarray(labels)[va]
    assert [int((va_labels == c).sum()) for c in range(3)] == [2, 4, 1]


def test_loaders_split_files_before_augmentation(image_folder):
    train, val, classes = make_loaders(image_folder, 0.2, batch_size=4, size=32, exclude=("my_pets",),
                                       num_workers=0)
    assert classes == ["cat", "corgi", "husky"]
    train_files = {train.dataset.dataset.samples[i][0] for i in train.dataset.indices}
    val_files = {val.dataset.dataset.samples[i][0] for i in val.dataset.indices}
    # No image (and therefore no augmented copy of it) is in both sets.
    assert not train_files & val_files
    assert len(train_files) + len(val_files) == 17
    x, y = next(iter(val))
    assert x.shape[1:] == (3, 32, 32) and y.dtype == torch.long


def test_load_image_returns_tensor_and_matching_crop(image_folder):
    x, rgb = load_image(image_folder / "cat" / "cat0.jpg", size=32)
    assert x.shape == (1, 3, 32, 32)
    assert rgb.shape == (32, 32, 3) and rgb.dtype == np.uint8


def test_spearman():
    a = np.arange(10.0)
    assert spearman(a, a) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)


def test_randomization_test_reports_each_step():
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(8, 8, 3, padding=1), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(8, 3),
    )
    rows = randomization_test(model, "2", torch.randn(2, 3, 16, 16), steps=3)
    assert [r["step"] for r in rows] == [1, 2, 3]
    assert rows[0]["randomized_down_to"] == "6"  # the output layer goes first
    assert all(-1.0 <= r["spearman"] <= 1.0 for r in rows)


def test_exif_orientation_is_respected(tmp_path):
    img = Image.new("RGB", (64, 48), (200, 10, 10))
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90 degrees to display": stored sideways, as phones do
    img.save(tmp_path / "phone.jpg", exif=exif)
    assert open_rgb(tmp_path / "phone.jpg").size == (48, 64)
