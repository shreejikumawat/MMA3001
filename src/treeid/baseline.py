"""Classical (non-learning) branch detector used as the baseline.

The detector encodes three physical assumptions about a trellised apple tree
photographed from the inter-row:

1. **Near**: the tree being imaged is the closest object, so branch pixels have
   high depth-foreground rank (:func:`treeid.preprocess.depth_foreground`).
2. **Not leaf**: bark is brown/grey, so branch pixels have low excess-green.
3. **Thin and elongated**: branches are tube-like, which the Sato ridge filter
   (Hessian eigen-analysis, a second-order finite-difference operator) responds to.

A pixel is labelled branch when a weighted score of these cues exceeds a
threshold; small connected components are then removed. The four parameters are
tuned by grid search on the *training rows only*.

The baseline cannot, by construction, recover branches hidden behind leaves:
any such recovery must come from learned context, which is what the comparison
with the U-Net measures.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from skimage.filters import sato
from skimage.morphology import remove_small_objects

from .preprocess import excess_green


@dataclass(frozen=True)
class BaselineParams:
    """Tunable parameters of :func:`predict`.

    Attributes:
        w_ridge: Weight of the normalised ridge response.
        w_near: Weight of the depth-foreground rank.
        thresh: Decision threshold on the combined score.
        min_size: Connected components smaller than this (pixels) are removed.
    """

    w_ridge: float = 0.5
    w_near: float = 0.5
    thresh: float = 0.5
    min_size: int = 30


def score_map(rgb01: np.ndarray, near: np.ndarray, valid: np.ndarray,
              exg_leaf: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Compute the two normalised cue maps used by the baseline.

    Args:
        rgb01: (H, W, 3) float RGB in [0, 1] already in mask frame.
        near: (H, W) depth-foreground rank in [0, 1].
        valid: (H, W) bool, depth available.
        exg_leaf: Excess-green above which a pixel is treated as foliage.

    Returns:
        ``(ridge, near_gated)`` each in [0, 1]; foliage pixels are zeroed in both.
    """
    gray = rgb01.mean(axis=-1)
    # Dark ridges (branches are darker than sky / background canopy).
    ridge = sato(gray, sigmas=(1, 2, 3), black_ridges=True)
    ridge = ridge / (np.percentile(ridge, 99.5) + 1e-9)
    ridge = np.clip(ridge, 0, 1)
    not_leaf = excess_green(rgb01 * 255) <= exg_leaf
    near_g = np.where(valid, near, 0.0)
    return ridge * not_leaf, near_g * not_leaf


def predict(rgb01: np.ndarray, near: np.ndarray, valid: np.ndarray,
            params: BaselineParams = BaselineParams(), cues=None) -> np.ndarray:
    """Binary branch mask from the classical cues.

    Args:
        rgb01, near, valid: See :func:`score_map`.
        params: Detector parameters.
        cues: Optional pre-computed output of :func:`score_map` (for fast grid search).

    Returns:
        (H, W) bool mask.
    """
    ridge, near_g = cues if cues is not None else score_map(rgb01, near, valid)
    s = (params.w_ridge * ridge + params.w_near * near_g) / (params.w_ridge + params.w_near)
    mask = s > params.thresh
    if params.min_size > 0:
        mask = remove_small_objects(mask, min_size=params.min_size)
    return mask


def grid_search(samples, metric, grid=None) -> tuple[BaselineParams, list[dict]]:
    """Exhaustive search of baseline parameters on training samples.

    Args:
        samples: Iterable of ``(rgb01, near, valid, gt)`` tuples (training rows only).
        metric: Callable ``metric(pred, gt) -> float`` to maximise (mean over samples).
        grid: Optional dict of candidate values per :class:`BaselineParams` field.

    Returns:
        ``(best_params, table)`` where ``table`` lists every candidate and its score.
    """
    grid = grid or {
        "w_ridge": [0.0, 0.25, 0.5, 0.75, 1.0],
        "thresh": [0.3, 0.4, 0.5, 0.6, 0.7],
        "min_size": [0, 30, 100],
    }
    cached = [(score_map(r, n, v), g) for r, n, v, g in samples]
    table = []
    best, best_s = None, -np.inf
    for wr, th, ms in itertools.product(grid["w_ridge"], grid["thresh"], grid["min_size"]):
        p = BaselineParams(w_ridge=wr, w_near=1 - wr, thresh=th, min_size=ms)
        if p.w_ridge + p.w_near == 0:
            continue
        s = float(np.mean([metric(predict(None, None, None, p, c), g) for c, g in cached]))
        table.append({**p.__dict__, "score": s})
        if s > best_s:
            best, best_s = p, s
    return best, table
