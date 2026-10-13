"""Verification of the metric implementations against hand-computable cases."""

import numpy as np
import pytest

from treeid.metrics import (boundary_f1, dice, iou, occluded_recall, score_all,
                            tolerant_precision_recall)


def vline(shape=(64, 64), col=32, width=1, rows=slice(5, 60)):
    m = np.zeros(shape, bool)
    m[rows, col: col + width] = True
    return m


def test_iou_dice_known_values():
    a = np.zeros((10, 10), bool); a[:, :5] = True     # 50 px
    b = np.zeros((10, 10), bool); b[:, 2:7] = True    # 50 px, overlap 30
    assert iou(a, b) == pytest.approx(30 / 70)
    assert dice(a, b) == pytest.approx(60 / 100)


def test_empty_conventions():
    z = np.zeros((8, 8), bool)
    assert iou(z, z) == 1.0 and dice(z, z) == 1.0
    p, r, f = tolerant_precision_recall(z, z)
    assert (p, r) == (1.0, 1.0)


def test_identical_masks_are_perfect():
    m = vline(width=5)
    s = score_all(m, m, occluded=np.ones_like(m))
    for k in ("iou", "dice", "bf1", "cl_precision", "cl_recall", "cl_f1", "occ_recall"):
        assert s[k] == pytest.approx(1.0), k


@pytest.mark.parametrize("shift,tol,expect", [(2, 2, 1.0), (3, 2, 0.0), (3, 3, 1.0), (10, 4, 0.0)])
def test_tolerance_is_a_hard_distance_threshold(shift, tol, expect):
    """A one-pixel line shifted by `shift` px is matched iff shift <= tol."""
    g = vline()
    p = vline(col=32 + shift)
    _, r, _ = tolerant_precision_recall(p, g, tol=tol, skeleton=False)
    assert r == pytest.approx(expect)


def test_iou_collapses_for_thin_offset_but_tolerant_score_does_not():
    """Motivates the centreline metric: 2-px offset of a 1-px line -> IoU 0, clF1 1."""
    g, p = vline(), vline(col=34)
    assert iou(p, g) == 0.0
    assert tolerant_precision_recall(p, g, tol=2)[2] == pytest.approx(1.0)


def test_half_recall():
    g = vline(rows=slice(0, 60))
    p = vline(rows=slice(0, 30))
    _, r, _ = tolerant_precision_recall(p, g, tol=0, skeleton=False)
    assert r == pytest.approx(0.5)


def test_occluded_recall_nan_when_nothing_occluded():
    g = vline(width=5)
    assert np.isnan(occluded_recall(g, g, np.zeros_like(g)))


def test_boundary_f1_symmetric():
    a = vline(width=7); b = vline(width=7, col=34)
    assert boundary_f1(a, b) == pytest.approx(boundary_f1(b, a))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        iou(np.zeros((4, 4)), np.zeros((5, 5)))


def test_negative_tolerance_raises():
    with pytest.raises(ValueError):
        tolerant_precision_recall(vline(), vline(), tol=-1)


def test_shift_compensated_recovers_known_offset():
    from treeid.metrics import shift_compensated_f1
    g = vline(width=5)
    p = vline(width=5, col=32 + 11)
    strict = tolerant_precision_recall(p, g, tol=2)[2]
    f, k = shift_compensated_f1(p, g, tol=2, max_shift=20)
    assert strict == 0.0
    assert f == pytest.approx(1.0) and k == 11


def test_shift_compensated_never_below_strict():
    from treeid.metrics import shift_compensated_f1
    rng = np.random.default_rng(2)
    for _ in range(5):
        g = rng.random((40, 40)) > 0.97
        p = rng.random((40, 40)) > 0.97
        assert shift_compensated_f1(p, g, tol=2, max_shift=5)[0] >= \
            tolerant_precision_recall(p, g, tol=2)[2] - 1e-12
