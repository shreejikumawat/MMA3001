"""Manufactured-solution tests for FFT cross-correlation registration.

A synthetic 'tree' is translated by a known amount; the estimator must recover it.
"""

import numpy as np
import pytest
from scipy import ndimage

from treeid.align import apply_shift, estimate_shift, xcorr2_fft


def synthetic_tree(n=128):
    m = np.zeros((n, n), bool)
    m[70:, 60:68] = True                                   # trunk
    for k in range(70):                                     # two leaders
        m[70 - k, 60 - k // 2: 64 - k // 2] = True
        m[70 - k, 64 + k // 2: 68 + k // 2] = True
    return m


def test_xcorr_matches_direct_sum():
    rng = np.random.default_rng(0)
    a, b = rng.random((6, 7)), rng.random((6, 7))
    c = xcorr2_fft(a, b)
    for dy, dx in [(0, 0), (1, -2), (-3, 4)]:
        direct = sum(a[y, x] * b[y - dy, x - dx]
                     for y in range(6) for x in range(7)
                     if 0 <= y - dy < 6 and 0 <= x - dx < 7)
        assert c[5 + dy, 6 + dx] == pytest.approx(direct)


@pytest.mark.parametrize("dy,dx", [(0, 0), (0, 17), (0, -23), (4, -9)])
def test_recovers_integer_shift(dy, dx):
    ref = synthetic_tree().astype(float)
    moving = ndimage.shift(ref, (-dy, -dx), order=0)
    edy, edx, ratio = estimate_shift(ref, moving, max_shift=(10, 40), subpixel=False)
    assert (edy, edx) == (dy, dx)
    assert ratio > 1.0


def test_subpixel_refinement_improves_estimate():
    ref = ndimage.gaussian_filter(synthetic_tree().astype(float), 2)
    moving = ndimage.shift(ref, (0, -5.4), order=3)
    _, dx_int, _ = estimate_shift(ref, moving, (2, 20), subpixel=False)
    _, dx_sub, _ = estimate_shift(ref, moving, (2, 20), subpixel=True)
    assert abs(dx_sub - 5.4) < abs(dx_int - 5.4)
    assert abs(dx_sub - 5.4) < 0.2


def test_apply_shift_roundtrip_preserves_area():
    m = synthetic_tree()
    back = apply_shift(apply_shift(m, 0, 11), 0, -11)
    assert abs(int(back.sum()) - int(m.sum())) <= 0.02 * m.sum()


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        xcorr2_fft(np.zeros((3, 3)), np.zeros((3, 4)))
