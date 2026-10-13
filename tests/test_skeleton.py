"""Manufactured-solution tests for the structural measurements."""

import numpy as np
import pytest

from treeid.skeleton import backproject, fork_row, largest_component, leader_angle, runs_per_row


def v_tree(n=200, half_slope=0.5, fork=120, w=6):
    """Synthetic open-V tree: vertical trunk below `fork`, two straight leaders above.

    Each leader moves `half_slope` columns per row, so the exact opening angle is
    2*atan(half_slope).
    """
    m = np.zeros((n, n), bool)
    c = n // 2
    m[fork:, c - w // 2: c + w // 2] = True
    for k in range(fork):
        y = fork - k
        dx = int(round(half_slope * k))
        m[y, c - dx - w // 2: c - dx + w // 2] = True
        m[y, c + dx - w // 2: c + dx + w // 2] = True
    return m


@pytest.mark.parametrize("half_slope", [0.25, 0.5, 0.7])
def test_leader_angle_matches_analytical(half_slope):
    exact = np.degrees(2 * np.arctan(half_slope))
    ang, info = leader_angle(v_tree(half_slope=half_slope))
    assert ang == pytest.approx(exact, abs=1.5)


def test_fork_row_close_to_construction():
    fy = fork_row(v_tree(fork=120))
    assert fy is not None and abs(fy - 120) <= 15


def test_runs_per_row():
    m = np.zeros((2, 10), bool); m[0, [1, 2, 5, 8, 9]] = True
    assert list(runs_per_row(m)) == [3, 0]


def test_largest_component_drops_specks():
    m = v_tree(); m[5, 5] = True
    assert not largest_component(m)[5, 5]


def test_no_fork_returns_none():
    m = np.zeros((100, 100), bool); m[:, 48:52] = True
    ang, info = leader_angle(m)
    assert ang is None and info["fork_row"] is None


def test_backproject_planar_scene():
    """Constant depth Z -> all points at Z, X proportional to (u - cx)."""
    m = np.zeros((50, 50), bool); m[10:40, 30] = True
    depth = np.full((50, 50), 2.0)
    pts = backproject(m, depth, fx=100, fy=100, cx=25, cy=25)
    assert np.allclose(pts[:, 2], 2.0)
    assert np.allclose(pts[:, 0], (30 - 25) * 2.0 / 100)


def test_backproject_skips_missing_depth():
    m = np.zeros((50, 50), bool); m[10:40, 30] = True
    assert len(backproject(m, np.zeros((50, 50)), 100, 100, 25, 25)) == 0
