"""Tests for the U-Net, FLOP estimator, training loop and classical baseline."""

import numpy as np
import pytest
import torch
from torch import nn

from treeid import baseline
from treeid.model import UNet, count_params, estimate_flops
from treeid.train import TrainConfig, fit, predict_proba, soft_dice_loss


def test_unet_output_shape():
    m = UNet(5, base=4, depth=2)
    assert m(torch.zeros(2, 5, 32, 32)).shape == (2, 1, 32, 32)


def test_flops_hand_count_single_conv():
    """2 * H*W*Cout*k*k*Cin for one 3x3 conv, no padding effects on the count."""
    conv = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1, bias=False))
    est = estimate_flops(conv, (3, 16, 16))
    assert est["flops"] == 2 * 16 * 16 * 8 * 3 * 3 * 3
    assert est["params"] == count_params(conv) == 8 * 3 * 3 * 3


def test_flops_scale_with_pixels():
    m = UNet(3, base=4, depth=2)
    f32 = estimate_flops(m, (3, 32, 32))["flops"]
    f64 = estimate_flops(m, (3, 64, 64))["flops"]
    assert f64 == pytest.approx(4 * f32)


def test_soft_dice_perfect_and_worst():
    t = torch.zeros(1, 1, 8, 8); t[..., 2:5, 2:5] = 1
    good = (t * 2 - 1) * 50
    assert soft_dice_loss(good, t) < 1e-3
    assert soft_dice_loss(-good, t) > 0.9


def test_fit_learns_trivial_task_and_is_repeatable():
    """Overfit a tiny synthetic task: branch = bright pixels. Same seed -> same model."""
    rng = np.random.default_rng(0)
    y = np.zeros((16, 32, 32), bool); y[:, :, 14:18] = True
    x = rng.normal(0, 0.1, (16, 3, 32, 32)).astype(np.float32)
    x[:, :, y[0]] += 1.0
    cfg = TrainConfig(base=4, depth=2, max_epochs=25, patience=25, batch_size=8, seed=3)
    r1 = fit(x, y, x, y, cfg, log=None)
    r2 = fit(x, y, x, y, cfg, log=None)
    assert r1.history[-1]["val_cl_f1"] > 0.9
    p1, p2 = predict_proba(r1.model, x), predict_proba(r2.model, x)
    assert np.allclose(p1, p2, atol=1e-5)


def test_fit_rejects_bad_shapes():
    with pytest.raises(ValueError):
        fit(np.zeros((2, 3, 8, 8), np.float32), np.zeros((3, 8, 8), bool),
            np.zeros((1, 3, 8, 8), np.float32), np.zeros((1, 8, 8), bool), log=None)


def test_baseline_finds_near_dark_ridge_and_ignores_leaves():
    rgb = np.full((64, 64, 3), 0.8, np.float32)
    rgb[:, 30:34] = [0.35, 0.3, 0.28]          # dark bark-coloured vertical strip
    rgb[:, 50:54] = [0.1, 0.6, 0.1]            # green strip (leaf) at same depth
    near = np.zeros((64, 64), np.float32); near[:, 28:56] = 1.0
    valid = np.ones((64, 64), bool)
    p = baseline.predict(rgb, near, valid, baseline.BaselineParams(0.5, 0.5, 0.4, 0))
    assert p[:, 30:34].mean() > 0.8
    assert p[:, 50:54].mean() < 0.05
