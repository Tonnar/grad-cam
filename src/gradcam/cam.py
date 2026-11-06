"""Grad-CAM (Selvaraju et al., 2017) for any convolutional PyTorch model.

Grad-CAM explains a prediction by weighting each channel of a convolutional
feature map by the average gradient of the class score with respect to that
channel, summing over channels and keeping the positive part:

    alpha_k = mean_{i,j} d y_c / d A_k[i, j]
    CAM     = ReLU( sum_k alpha_k * A_k )

The implementation captures activations and gradients with hooks on a chosen
layer, so the model runs its own ``forward`` unchanged and the explanation is
always of the prediction the model actually makes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_module(model: nn.Module, name: str) -> nn.Module:
    """Look up a sub-module by dotted name, e.g. ``"features.norm5"``."""
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(f"no module named {name!r}; try one of {list(modules)[-5:]}")
    return modules[name]


class GradCAM:
    """Compute Grad-CAM heatmaps for ``model`` at ``target_layer``.

    Use as a context manager (or call :meth:`remove`) so the hooks are
    detached afterwards::

        with GradCAM(model, "features.norm5") as cam:
            heatmaps, logits = cam(images)            # explain the predicted class
            heatmaps, _ = cam(images, target=3)      # or a chosen class
    """

    def __init__(self, model: nn.Module, target_layer: str | nn.Module):
        self.model = model
        layer = get_module(model, target_layer) if isinstance(target_layer, str) else target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [
            layer.register_forward_hook(self._save_activation),
        ]

    def _save_activation(self, module, inputs, output):
        self._activations = output
        output.register_hook(self._save_gradient)
        # Hand the rest of the network a copy: torchvision's DenseNet applies an
        # in-place ReLU right after `features`, which would otherwise overwrite
        # the saved activations.
        return output.clone()

    def _save_gradient(self, grad):
        self._gradients = grad

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def __call__(
        self, images: torch.Tensor, target: int | torch.Tensor | None = None, upsample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(heatmaps, logits)``.

        Args:
            images: (B, C, H, W) input batch, preprocessed as the model expects.
            target: class index (or (B,) indices) to explain; defaults to each
                image's predicted class.
            upsample: resize heatmaps to (H, W); otherwise keep the feature-map
                resolution.

        Heatmaps are in [0, 1] per image (all zeros if the class has no
        positive evidence at this layer).
        """
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.enable_grad():
                images = images.detach().requires_grad_(False)
                logits = self.model(images)
                if target is None:
                    target_idx = logits.argmax(dim=1)
                elif isinstance(target, int):
                    target_idx = torch.full((len(images),), target, device=logits.device, dtype=torch.long)
                else:
                    target_idx = torch.as_tensor(target, device=logits.device, dtype=torch.long)
                score = logits.gather(1, target_idx.view(-1, 1)).sum()
                self.model.zero_grad(set_to_none=True)
                score.backward()
        finally:
            self.model.train(was_training)

        acts, grads = self._activations.detach(), self._gradients.detach()
        weights = grads.mean(dim=(2, 3), keepdim=True)  # one weight per channel, all channels
        cam = F.relu((weights * acts).sum(dim=1))  # (B, h, w)
        if upsample:
            cam = F.interpolate(cam.unsqueeze(1), size=images.shape[-2:], mode="bilinear", align_corners=False)
            cam = cam.squeeze(1)
        peak = cam.flatten(1).max(dim=1).values.clamp_min(1e-12).view(-1, 1, 1)
        return cam / peak, logits.detach()


def overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4, cmap: str = "jet") -> np.ndarray:
    """Blend a [0, 1] heatmap onto an RGB uint8 image (resizing the heatmap)."""
    from matplotlib import colormaps
    from PIL import Image

    h, w = image.shape[:2]
    hm = np.asarray(Image.fromarray(np.uint8(255 * np.clip(heatmap, 0, 1))).resize((w, h), Image.BILINEAR))
    colored = colormaps[cmap](hm / 255.0)[..., :3] * 255.0
    out = (1 - alpha) * image.astype(np.float32) + alpha * colored
    return np.clip(out, 0, 255).astype(np.uint8)
