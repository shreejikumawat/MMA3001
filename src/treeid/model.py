"""Compact U-Net for binary branch segmentation, with a FLOP estimator.

The architecture follows Ronneberger et al. (2015) with three down-sampling
levels and a configurable base width. It is deliberately small (~0.5 M
parameters at ``base=16``) because the dataset has only a few hundred labelled
images and the model must run on an embedded robot computer.
"""

from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Sequential):
    """Two 3x3 convolutions, each followed by batch normalisation and ReLU."""

    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """U-Net with ``depth`` pooling stages.

    Args:
        in_ch: Number of input channels (3 for RGB, 5 for RGB + depth + valid).
        base: Channels in the first stage; doubled at every down-sampling.
        depth: Number of pooling stages. Input size must be divisible by ``2**depth``.

    The forward pass returns *logits* of shape (B, 1, H, W); apply ``sigmoid`` for
    probabilities.
    """

    def __init__(self, in_ch: int = 3, base: int = 16, depth: int = 3):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList()
        c = in_ch
        for ch in chs:
            self.down.append(DoubleConv(c, ch))
            c = ch
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for ch in reversed(chs[:-1]):
            self.up.append(nn.ConvTranspose2d(c, ch, 2, stride=2))
            self.dec.append(DoubleConv(2 * ch, ch))
            c = ch
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, blk in enumerate(self.down):
            x = blk(x)
            if i < len(self.down) - 1:
                skips.append(x)
                x = self.pool(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), s], dim=1))
        return self.head(x)


def count_params(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model: nn.Module, in_shape: tuple[int, int, int]) -> dict[str, float]:
    """Estimate forward-pass FLOPs and memory traffic for one image.

    Counts 2 FLOPs per multiply-accumulate in every ``Conv2d`` and
    ``ConvTranspose2d`` (other layers are < 2% of the total). Memory traffic is
    approximated as weights + input activations + output activations read or
    written once in float32, which gives a *lower bound* on bytes moved and hence
    an *upper bound* on arithmetic intensity.

    Args:
        model: Network to analyse.
        in_shape: ``(C, H, W)`` of a single input.

    Returns:
        Dict with ``flops``, ``bytes``, ``arith_intensity`` (FLOP/byte) and ``params``.
    """
    flops, nbytes = 0.0, 0.0
    hooks = []

    def hook(mod, inp, out):
        nonlocal flops, nbytes
        x = inp[0]
        if isinstance(mod, nn.Conv2d):
            k = mod.kernel_size[0] * mod.kernel_size[1] * mod.in_channels // mod.groups
            macs = out.numel() * k
        else:  # ConvTranspose2d: every input element scatters into a k*k*cout patch
            macs = x.numel() * mod.kernel_size[0] * mod.kernel_size[1] * mod.out_channels
        flops += 2 * macs
        nbytes += 4 * (x.numel() + out.numel() + mod.weight.numel())

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(m.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *in_shape))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return {"flops": flops, "bytes": nbytes, "arith_intensity": flops / nbytes,
            "params": count_params(model)}
