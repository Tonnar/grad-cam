"""Model-randomisation sanity check for saliency maps (Adebayo et al., 2018).

A saliency method that is actually explaining the model should produce very
different maps once the model's learned weights are destroyed. If the maps
barely change, the method is mostly reflecting the input (edges, colour
contrast) or the architecture, not what the model learned.

:func:`randomization_test` re-initialises the model's layers from the output
backwards ("cascading randomisation") and reports, after each step, the rank
correlation between the original heatmap and the new one.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from gradcam.cam import GradCAM


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two arrays (ties broken by order)."""
    ra = np.argsort(np.argsort(a.ravel())).astype(np.float64)
    rb = np.argsort(np.argsort(b.ravel())).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def _parametrised_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [
        (name, m) for name, m in model.named_modules()
        if isinstance(m, (nn.Conv2d, nn.Linear)) and m.weight is not None
    ]


@torch.no_grad()
def _reinit(module: nn.Module, generator: torch.Generator) -> None:
    w = module.weight
    fan_in = w[0].numel()
    w.copy_(torch.randn(w.shape, generator=generator) * (2.0 / fan_in) ** 0.5)
    if getattr(module, "bias", None) is not None:
        module.bias.zero_()


def randomization_test(model: nn.Module, target_layer: str, images: torch.Tensor, steps: int = 5,
                       target: int | None = None, seed: int = 0) -> list[dict]:
    """Cascading randomisation of the top ``steps`` groups of layers.

    The layers with weights are split into ``steps`` equal groups, ordered
    from the output backwards; group k is randomised on top of groups
    1..k-1. Returns one row per step with the mean Spearman correlation
    between original and randomised heatmaps (over ``images``).
    The explained class is fixed to the original model's prediction (or
    ``target``) so the comparison is like for like.
    """
    with GradCAM(model, target_layer) as cam:
        base, logits = cam(images, target)
    target_idx = logits.argmax(dim=1) if target is None else target

    layers = _parametrised_layers(model)[::-1]
    groups = np.array_split(np.arange(len(layers)), steps)
    randomized = copy.deepcopy(model)
    rand_layers = dict(_parametrised_layers(randomized))
    gen = torch.Generator().manual_seed(seed)
    rows = []
    for k, group in enumerate(groups, start=1):
        names = [layers[i][0] for i in group]
        for name in names:
            _reinit(rand_layers[name], gen)
        with GradCAM(randomized, target_layer) as cam:
            maps, _ = cam(images, target_idx)
        corr = np.mean([spearman(b.numpy(), m.numpy()) for b, m in zip(base.cpu(), maps.cpu())])
        rows.append({"step": k, "randomized_down_to": names[-1], "spearman": float(corr)})
    return rows
