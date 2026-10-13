"""From a branch mask to engineering quantities a pruning / thinning robot can use.

The orchard uses a *2D open-V* trellis: one trunk that forks into two leaders.
Two geometric quantities summarise that structure and are what a robot needs to
plan an approach path: the **fork height** (where the trunk splits) and the
**leader angle** (the opening of the V). Both are estimated here from a mask so
that predicted masks can be compared with ground truth in engineering units
rather than only by pixel overlap.

Leader directions are obtained by linear least-squares regression of column on
row for the skeleton pixels of each leader (Week 5.1, linear regression), which
is well conditioned because leaders are close to vertical.

:func:`backproject` lifts skeleton pixels to 3-D with the pinhole camera model.
Because the supplied depth is an 8-bit re-scaled image with unknown scale, the
result is in *relative* depth units unless a calibration is supplied.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 8-connected component of a boolean mask."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n <= 1:
        return mask.astype(bool)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def runs_per_row(mask: np.ndarray) -> np.ndarray:
    """Number of separate horizontal runs of ``True`` in each image row."""
    m = mask.astype(np.int8)
    starts = (np.diff(m, axis=1, prepend=0) == 1).sum(axis=1)
    return starts


def fork_row(mask: np.ndarray, persist: int = 8) -> int | None:
    """Image row at which the trunk splits into two leaders.

    Scanning upward from the bottom, the fork is the first row above which at
    least two runs persist for ``persist`` consecutive rows (filters noise).

    Returns:
        Row index (0 = top) or ``None`` if no fork is found.
    """
    r = runs_per_row(largest_component(mask))
    h = len(r)
    for y in range(h - 1, persist - 1, -1):
        if r[y] >= 1 and (r[y - persist: y] >= 2).all():
            return y - persist
    return None


def leader_angle(mask: np.ndarray, persist: int = 8) -> tuple[float, dict] | tuple[None, dict]:
    """Opening angle (degrees) between the two leaders above the fork.

    Skeleton pixels above the fork are split at the fork's column into left and
    right leaders; each is fitted with ``x = a*y + b`` by least squares and the
    angle between the two direction vectors is returned.

    Returns:
        ``(angle_deg, info)`` where ``info`` holds the fork row, slopes and fit
        residuals; ``(None, info)`` if a leader has fewer than 10 skeleton pixels.
    """
    m = largest_component(mask)
    fy = fork_row(m, persist)
    info: dict = {"fork_row": fy}
    if fy is None:
        return None, info
    sk = skeletonize(m)
    ys, xs = np.nonzero(sk[:fy])
    if len(xs) < 20:
        return None, info
    # Column of the fork: mean column of the mask in the fork row.
    fx = float(np.nonzero(m[fy])[0].mean()) if m[fy].any() else float(xs.mean())
    slopes = []
    for side in (xs < fx, xs >= fx):
        if side.sum() < 10:
            return None, info
        A = np.c_[ys[side], np.ones(side.sum())]
        coef, res, *_ = np.linalg.lstsq(A, xs[side], rcond=None)
        slopes.append(coef[0])
        info.setdefault("rms_px", []).append(float(np.sqrt(res[0] / side.sum())) if len(res) else 0.0)
    # Direction vectors pointing upward (dy = -1): (dx, dy) = (-a, -1).
    v1 = np.array([-slopes[0], -1.0]); v2 = np.array([-slopes[1], -1.0])
    cosang = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))
    info["slopes"] = [float(s) for s in slopes]
    info["fork_col"] = fx
    return float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))), info


def backproject(mask: np.ndarray, depth: np.ndarray, fx: float, fy: float,
                cx: float, cy: float, depth_scale: float = 1.0, win: int = 7) -> np.ndarray:
    """3-D points of the mask skeleton with the pinhole model ``X = (u-cx) Z / fx``.

    Depth at each skeleton pixel is the median of valid (non-zero) depth values
    in a ``win`` x ``win`` window, which bridges small holes. Pixels with no valid
    depth in the window are dropped.

    Args:
        mask: Branch mask in the same pixel grid as ``depth``.
        depth: Depth image (0 = missing).
        fx, fy, cx, cy: Intrinsics in pixels for this grid.
        depth_scale: Multiplier converting depth values to metres (1.0 leaves them
            in raw units, which is the case for the 8-bit TreeID depth).
        win: Odd window size for the median.

    Returns:
        (K, 3) float array of ``(X, Y, Z)``.
    """
    sk = skeletonize(largest_component(mask))
    vs, us = np.nonzero(sk)
    h, w = depth.shape
    r = win // 2
    pts = []
    for u, v in zip(us, vs):
        patch = depth[max(0, v - r): v + r + 1, max(0, u - r): u + r + 1]
        good = patch[patch > 0]
        if good.size == 0:
            continue
        z = float(np.median(good)) * depth_scale
        pts.append(((u - cx) * z / fx, (v - cy) * z / fy, z))
    return np.asarray(pts, float).reshape(-1, 3)
