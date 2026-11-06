import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from gradcam.cam import GradCAM, overlay
from gradcam.model import TARGET_LAYERS, build_classifier


def _toy_model():
    """logit_0 = mean(x), logit_1 = mean(-x): the evidence is the input itself."""
    conv = nn.Conv2d(1, 2, kernel_size=1, bias=False)
    with torch.no_grad():
        conv.weight[:] = torch.tensor([1.0, -1.0]).view(2, 1, 1, 1)
    return nn.Sequential(conv, nn.AdaptiveAvgPool2d(1), nn.Flatten())


def _two_patch_image():
    x = torch.zeros(1, 1, 8, 8)
    x[..., :3, :3] = 1.0   # positive patch, top-left
    x[..., 5:, 5:] = -2.0  # negative patch, bottom-right
    return x


def test_heatmap_highlights_the_evidence_for_the_requested_class():
    model, x = _toy_model(), _two_patch_image()
    with GradCAM(model, "0") as cam:
        h0, _ = cam(x, target=0)
        h1, _ = cam(x, target=1)
    assert h0.shape == (1, 8, 8) and h0.max() == pytest.approx(1.0)
    assert torch.all(h0[0, :3, :3] == 1) and torch.all(h0[0, 5:, 5:] == 0)
    assert torch.all(h1[0, 5:, 5:] == 1) and torch.all(h1[0, :3, :3] == 0)


def test_default_target_is_the_predicted_class():
    model, x = _toy_model(), _two_patch_image()  # mean(x) < 0, so class 1 wins
    with GradCAM(model, "0") as cam:
        h_default, logits = cam(x)
        h1, _ = cam(x, target=1)
    assert logits.argmax(1).item() == 1
    assert torch.equal(h_default, h1)


def test_densenet_explanation_matches_a_manual_computation():
    torch.manual_seed(0)
    model = build_classifier(4, "densenet121", pretrained=False, freeze_backbone=False).eval()
    x = torch.randn(2, 3, 64, 64)

    with GradCAM(model, TARGET_LAYERS["densenet121"]) as cam:
        heat, logits = cam(x, target=2, upsample=False)

    # The model's own predictions are unchanged by the hooks (the original
    # notebook re-implemented forward() and dropped DenseNet's final ReLU).
    with torch.no_grad():
        assert torch.allclose(logits, model(x), atol=1e-5)

    # Manual Grad-CAM over all 1024 channels.
    feats = model.features(x).detach().requires_grad_(True)
    out = model.classifier(torch.flatten(F.adaptive_avg_pool2d(F.relu(feats), 1), 1))
    (grad,) = torch.autograd.grad(out[:, 2].sum(), feats)
    manual = F.relu((grad.mean(dim=(2, 3), keepdim=True) * feats).sum(1)).detach()
    manual = manual / manual.flatten(1).max(1).values.clamp_min(1e-12).view(-1, 1, 1)
    assert feats.shape[1] == 1024
    assert torch.allclose(heat, manual, atol=1e-5)


def test_hooks_are_removed_and_training_mode_restored():
    model = _toy_model().train()
    layer = model[0]
    with GradCAM(model, "0") as cam:
        cam(_two_patch_image())
        assert len(layer._forward_hooks) == 1
    assert len(layer._forward_hooks) == 0
    assert model.training


def test_unknown_layer_name_raises():
    with pytest.raises(KeyError):
        GradCAM(_toy_model(), "nope")


def test_overlay_blends_into_uint8_rgb():
    img = np.full((30, 40, 3), 100, dtype=np.uint8)
    out = overlay(img, np.random.rand(7, 7), alpha=0.5)
    assert out.shape == (30, 40, 3) and out.dtype == np.uint8
    assert np.array_equal(overlay(img, np.zeros((7, 7)), alpha=0.0), img)
