import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def image_folder(tmp_path):
    """Tiny class-per-folder dataset of random JPEGs (plus a folder to exclude)."""
    rng = np.random.default_rng(0)
    root = tmp_path / "pets"
    for cls, n in [("cat", 6), ("corgi", 6), ("husky", 5), ("my_pets", 2)]:
        (root / cls).mkdir(parents=True)
        for i in range(n):
            # Give each class a different mean colour so a model can separate them.
            base = {"cat": 40, "corgi": 140, "husky": 220, "my_pets": 100}[cls]
            arr = np.clip(rng.normal(base, 30, (48, 64, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(root / cls / f"{cls}{i}.jpg")
    return root
