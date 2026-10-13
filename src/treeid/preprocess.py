"""Geometric and photometric preprocessing.

The RGB and depth images are 640x480 while the ground-truth masks are 256x256.
Following Chen et al. (2021) we take the central 480x480 square of each image
(columns 80-559) and resample it to the mask resolution. The crop window is a
parameter so that the alignment study in ``scripts/run_audit.py`` can test the
assumption instead of trusting it.

Interpolation choice (Week 3 content): colour and depth are resampled with
area averaging (``cv2.INTER_AREA``), which is an anti-aliasing low-pass filter for
down-sampling; binary masks are resampled with nearest-neighbour so that labels
are never invented by interpolation.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Default crop window ``(y0, y1, x0, x1)`` applied to 640x480 inputs.
DEFAULT_CROP = (0, 480, 80, 560)


def crop_resize(img: np.ndarray, size: int = 256, crop=DEFAULT_CROP,
                is_mask: bool = False) -> np.ndarray:
    """Crop a 640x480 image to the mask field of view and resize to ``size`` x ``size``.

    Args:
        img: (H, W) or (H, W, C) array.
        size: Output edge length in pixels.
        crop: ``(y0, y1, x0, x1)`` window in input pixel coordinates.
        is_mask: Use nearest-neighbour interpolation (labels) instead of area
            averaging (intensities).

    Returns:
        Array of shape (size, size[, C]) with the input dtype.

    Raises:
        ValueError: If the crop window lies outside the image.
    """
    y0, y1, x0, x1 = crop
    h, w = img.shape[:2]
    if not (0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w):
        raise ValueError(f"crop {crop} outside image of size {(h, w)}")
    sub = img[y0:y1, x0:x1]
    if is_mask:
        out = cv2.resize(sub.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
        return out.astype(img.dtype)
    return cv2.resize(sub, (size, size), interpolation=cv2.INTER_AREA)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize of a square boolean mask to ``size`` x ``size``."""
    if mask.shape[0] == size:
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8), (size, size),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def excess_green(rgb: np.ndarray) -> np.ndarray:
    """Excess-green vegetation index ``ExG = 2g - r - b`` on chromaticity coordinates.

    ``r, g, b`` are each channel divided by ``R+G+B`` so the index is insensitive
    to overall brightness (Woebbecke et al., 1995). Range is [-1, 2]; foliage is
    typically above ~0.05, bark and soil below.
    """
    x = rgb.astype(np.float32) + 1e-6
    s = x.sum(axis=-1)
    r, g, b = x[..., 0] / s, x[..., 1] / s, x[..., 2] / s
    return 2 * g - r - b


def foliage_mask(rgb: np.ndarray, thresh: float = 0.05) -> np.ndarray:
    """Boolean map of pixels that look like leaves (``ExG > thresh``).

    Used to define *occluded* branch pixels: a ground-truth branch pixel that is
    covered by foliage in the summer image cannot be seen directly and must be
    inferred, which is the hard part of the task.
    """
    return excess_green(rgb) > thresh


def depth_foreground(depth: np.ndarray, near_is_bright: bool = True) -> np.ndarray:
    """Soft foreground map in [0, 1] from an 8-bit depth image.

    Zero-valued pixels are treated as *missing* (no depth return) and set to 0.
    The remaining values are rank-normalised so the nearest objects map to 1.
    Rank normalisation makes the map independent of the unknown, per-image 8-bit
    scaling documented in the data audit.

    Args:
        depth: (H, W) uint8 depth image.
        near_is_bright: Whether larger pixel values mean closer surfaces. Set by
            the data audit; see ``results/audit.json``.
    """
    d = depth.astype(np.float32)
    valid = d > 0
    out = np.zeros_like(d)
    if valid.any():
        v = d[valid] if near_is_bright else -d[valid]
        ranks = v.argsort().argsort().astype(np.float32)
        out[valid] = ranks / max(len(ranks) - 1, 1)
    return out


def make_input(rgb: np.ndarray, depth: np.ndarray, size: int = 256,
               use_depth: bool = True, crop=DEFAULT_CROP,
               near_is_bright: bool = True) -> np.ndarray:
    """Build the network input tensor for one tree.

    Channels are RGB scaled to [0, 1] and, optionally, the rank-normalised depth
    foreground plus a binary depth-valid flag (so the network can tell "far" from
    "missing").

    Returns:
        float32 array of shape (C, size, size) with C = 3 or 5.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must be (H, W, 3), got {rgb.shape}")
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"depth {depth.shape} and rgb {rgb.shape[:2]} sizes differ")
    x = crop_resize(rgb, size, crop).astype(np.float32) / 255.0
    chans = [x.transpose(2, 0, 1)]
    if use_depth:
        dcrop = crop_resize(depth, size, crop, is_mask=True)  # keep zeros as zeros
        chans.append(depth_foreground(dcrop, near_is_bright)[None])
        chans.append((dcrop > 0).astype(np.float32)[None])
    return np.concatenate(chans, axis=0)
