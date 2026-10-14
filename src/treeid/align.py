"""Registration of the ground-truth masks to the image frame.

The data audit found the summer ground-truth (GT) masks offset horizontally from
the visible trunk by a tree-dependent amount. If uncorrected, this offset is
*label noise*: a perfect detector would be scored as wrong. This module
estimates the offset of a mask by maximising its cross-correlation with a
reference image of where the tree is. Two references were tried (report,
Section 6): a label-free bark map from colour and ridge cues (too ambiguous to
use automatically) and the prediction of a trained network
(``scripts/refine_labels.py``, via :func:`treeid.metrics.shift_compensated_f1`).

Numerical method
----------------
For two zero-mean images ``a`` and ``b`` of size N x M the circular
cross-correlation is ``c = IFFT( FFT(a) * conj(FFT(b)) )``. Computing it this way
costs O(NM log NM) instead of O(N^2 M^2) for direct summation. Images are
zero-padded to 2N x 2M so circular wrap-around does not produce spurious peaks.
The integer peak is refined to sub-pixel precision by fitting a parabola through
the peak and its two neighbours (three-point quadratic interpolation).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def xcorr2_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full linear 2-D cross-correlation of ``a`` with ``b`` via the FFT.

    Args:
        a: Reference image, shape (N, M).
        b: Moving image, same shape as ``a``.

    Returns:
        Array of shape (2N-1, 2M-1). Index ``(N-1, M-1)`` corresponds to zero
        shift; element ``(N-1+dy, M-1+dx)`` is ``sum a[y, x] * b[y-dy, x-dx]``,
        i.e. the score for moving ``b`` by ``(dy, dx)``.

    Raises:
        ValueError: If shapes differ.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    n, m = a.shape
    shape = (2 * n - 1, 2 * m - 1)
    fa = np.fft.rfft2(a, shape)
    fb = np.fft.rfft2(b, shape)
    c = np.fft.irfft2(fa * np.conj(fb), shape)
    # Re-order so zero shift sits at the centre index (n-1, m-1).
    return np.roll(c, (n - 1, m - 1), axis=(0, 1))


def _parabolic_offset(cm: float, c0: float, cp: float) -> float:
    """Vertex of the parabola through (-1, cm), (0, c0), (+1, cp), clipped to [-0.5, 0.5]."""
    den = cm - 2 * c0 + cp
    if den >= 0:  # not a maximum (flat or convex): no refinement
        return 0.0
    return float(np.clip(0.5 * (cm - cp) / den, -0.5, 0.5))


def estimate_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    max_shift: tuple[int, int] = (10, 60),
    subpixel: bool = True,
) -> tuple[float, float, float]:
    """Shift ``(dy, dx)`` that best aligns ``moving`` onto ``reference``.

    Both images are mean-subtracted before correlation so that large uniform
    regions do not dominate. Only shifts within ``max_shift`` are considered,
    which encodes the physical prior that the masks are only slightly mis-placed.

    Args:
        reference: Fixed image (e.g. depth foreground probability).
        moving: Image to be shifted (e.g. GT mask).
        max_shift: Maximum absolute ``(dy, dx)`` searched, in pixels.
        subpixel: Refine the integer peak with three-point parabolic interpolation.

    Returns:
        ``(dy, dx, peak_ratio)``. ``peak_ratio`` is the best correlation divided by
        the second-best correlation outside a 3-pixel neighbourhood; values close
        to 1 mean the alignment is ambiguous and should not be trusted.
    """
    ref = np.asarray(reference, float)
    mov = np.asarray(moving, float)
    ref = ref - ref.mean()
    mov = mov - mov.mean()
    c = xcorr2_fft(ref, mov)
    n, m = ref.shape
    cy, cx = n - 1, m - 1
    sy, sx = max_shift
    win = c[cy - sy: cy + sy + 1, cx - sx: cx + sx + 1]
    iy, ix = np.unravel_index(np.argmax(win), win.shape)
    best = win[iy, ix]
    dy, dx = float(iy - sy), float(ix - sx)
    if subpixel:
        if 0 < iy < win.shape[0] - 1:
            dy += _parabolic_offset(win[iy - 1, ix], best, win[iy + 1, ix])
        if 0 < ix < win.shape[1] - 1:
            dx += _parabolic_offset(win[iy, ix - 1], best, win[iy, ix + 1])
    masked = win.copy()
    masked[max(0, iy - 3): iy + 4, max(0, ix - 3): ix + 4] = -np.inf
    second = masked.max()
    peak_ratio = float(best / second) if second > 0 else float("inf")
    return dy, dx, peak_ratio


def apply_shift(mask: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Translate a binary mask by ``(dy, dx)`` pixels, filling exposed areas with False.

    Sub-pixel shifts use bilinear interpolation followed by a 0.5 threshold, so
    the area of the mask is preserved to within one pixel per boundary.
    """
    m = np.asarray(mask, float)
    out = ndimage.shift(m, (dy, dx), order=1, mode="constant", cval=0.0)
    return out >= 0.5
