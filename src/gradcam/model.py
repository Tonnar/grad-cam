"""Transfer-learning classifiers and checkpoint I/O."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

# Last convolutional block of each supported backbone: the usual Grad-CAM layer.
TARGET_LAYERS = {
    "densenet121": "features.norm5",
    "resnet18": "layer4",
    "resnet50": "layer4",
}


def _head(in_features: int, num_classes: int, hidden: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_features
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, num_classes))
    return nn.Sequential(*layers)


def build_classifier(
    num_classes: int,
    arch: str = "densenet121",
    pretrained: bool = True,
    freeze_backbone: bool = True,
    hidden: tuple[int, ...] = (512,),
) -> nn.Module:
    """An ImageNet backbone with a new MLP head for ``num_classes``.

    With ``freeze_backbone=True`` only the head is trained (feature
    extraction); otherwise the whole network is fine-tuned.
    """
    if arch not in TARGET_LAYERS:
        raise ValueError(f"arch must be one of {list(TARGET_LAYERS)}")
    weights = "DEFAULT" if pretrained else None
    model = getattr(models, arch)(weights=weights)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    if arch.startswith("densenet"):
        model.classifier = _head(model.classifier.in_features, num_classes, hidden)
    else:
        model.fc = _head(model.fc.in_features, num_classes, hidden)
    return model


def save_checkpoint(path: str | Path, model: nn.Module, classes: list[str], arch: str, hidden: tuple[int, ...],
                    **extra) -> None:
    torch.save(
        {"state_dict": model.state_dict(), "classes": classes, "arch": arch, "hidden": list(hidden), **extra},
        path,
    )


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild the model saved by :func:`save_checkpoint` (no download needed)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_classifier(len(ckpt["classes"]), ckpt["arch"], pretrained=False, freeze_backbone=False,
                             hidden=tuple(ckpt["hidden"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
