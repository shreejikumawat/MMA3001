"""Segmentation metrics for thin branch structures.

Pixel-overlap scores (IoU, Dice) under-reward thin structures because a one-pixel
offset along a branch only a few pixels wide destroys most of the overlap. The
module also has tolerance-based scores: a true centreline pixel counts as found
if a predicted centreline pixel is within ``tol`` pixels of it, and the same the
other way round for precision. This is closer to what matters for a pruning
robot than exact pixel overlap.

All functions take boolean arrays of identical shape and return Python floats.
Empty-set conventions are documented per function so that results are never
``nan`` silently.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def _check(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and cast a prediction / ground-truth pair to bool.

    Raises:
        ValueError: If the shapes differ.
    """
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    return pred, gt


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection over union (Jaccard index).

    Returns 1.0 when both masks are empty (nothing to find, nothing predicted).
    """
    pred, gt = _check(pred, gt)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice / F1 overlap, ``2|A∩B| / (|A| + |B|)``. 1.0 when both are empty."""
    pred, gt = _check(pred, gt)
    s = pred.sum() + gt.sum()
    if s == 0:
        return 1.0
    return float(2 * np.logical_and(pred, gt).sum() / s)


def _dist_to(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance (pixels) from every pixel to the nearest ``True`` of ``mask``.

    If ``mask`` is empty every distance is ``inf``.
    """
    if not mask.any():
        return np.full(mask.shape, np.inf)
    return ndimage.distance_transform_edt(~mask)


def tolerant_precision_recall(
    pred: np.ndarray, gt: np.ndarray, tol: float = 2.0, skeleton: bool = True
) -> tuple[float, float, float]:
    """Precision, recall and F1 with a spatial tolerance.

    A true branch pixel counts as *recalled* if any predicted pixel lies within
    ``tol`` pixels of it; a predicted pixel counts as *precise* if any true pixel
    lies within ``tol``. With ``skeleton=True`` both masks are first reduced to
    one-pixel-wide centrelines so that thick and thin branches are weighted by
    length rather than area, which is the quantity a pruning robot needs.

    Args:
        pred: Predicted branch mask.
        gt: Ground-truth branch mask.
        tol: Matching tolerance in pixels (>= 0).
        skeleton: Compare centrelines instead of full masks.

    Returns:
        ``(precision, recall, f1)``. Precision is 1.0 when nothing is predicted;
        recall is 1.0 when the ground truth is empty.

    Raises:
        ValueError: If ``tol`` is negative or shapes differ.
    """
    if tol < 0:
        raise ValueError("tol must be non-negative")
    pred, gt = _check(pred, gt)
    if skeleton:
        pred = skeletonize(pred) if pred.any() else pred
        gt = skeletonize(gt) if gt.any() else gt
    d_to_gt = _dist_to(gt)
    d_to_pred = _dist_to(pred)
    precision = float((d_to_gt[pred] <= tol).mean()) if pred.any() else 1.0
    recall = float((d_to_pred[gt] <= tol).mean()) if gt.any() else 1.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tol: float = 2.0) -> float:
    """Boundary F1 score (Csurka et al., 2013) used by Chen et al. (2021).

    Boundaries are the mask pixels removed by a 3x3 erosion. The score is the
    tolerant F1 of the two boundary maps (no skeletonisation).
    """
    pred, gt = _check(pred, gt)
    pb = pred & ~ndimage.binary_erosion(pred)
    gb = gt & ~ndimage.binary_erosion(gt)
    return tolerant_precision_recall(pb, gb, tol=tol, skeleton=False)[2]


def occluded_recall(
    pred: np.ndarray, gt: np.ndarray, occluded: np.ndarray, tol: float = 2.0
) -> float:
    """Recall restricted to ground-truth centreline pixels hidden by foliage.

    Args:
        pred: Predicted branch mask.
        gt: Ground-truth branch mask.
        occluded: Boolean map of pixels where the branch is not visible in the
            input image (e.g. leaf-covered). See
            :func:`treeid.preprocess.foliage_mask`.
        tol: Matching tolerance in pixels.

    Returns:
        Fraction of occluded centreline pixels recovered, or ``nan`` when the tree
        has no occluded branch pixels (callers should use ``np.nanmean``).
    """
    pred, gt = _check(pred, gt)
    occ = np.asarray(occluded, bool)
    g = skeletonize(gt) if gt.any() else gt
    g_occ = g & occ
    if not g_occ.any():
        return float("nan")
    return float((_dist_to(pred)[g_occ] <= tol).mean())


def shift_compensated_f1(pred: np.ndarray, gt: np.ndarray, tol: float = 4.0,
                         max_shift: int = 24, step: int = 1) -> tuple[float, int]:
    """Centreline F1 after the best horizontal re-registration of the label.

    The ground truth is translated horizontally by every integer offset in
    ``[-max_shift, max_shift]`` and the best tolerant F1 is kept. This removes the
    effect of the label mis-registration found in the data audit and therefore
    measures whether the *shape* of the predicted structure is right. It is an
    optimistic score and is always reported next to the strict score.

    Returns:
        ``(best_f1, best_shift_px)``; when several shifts tie, the median of the
        tied shifts is returned.
    """
    pred, gt = _check(pred, gt)
    ps = skeletonize(pred) if pred.any() else pred
    gs = skeletonize(gt) if gt.any() else gt
    d_to_pred = _dist_to(ps)
    scores = {}
    w = gt.shape[1]
    for k in range(-max_shift, max_shift + 1, step):
        g = np.zeros_like(gs)
        if k >= 0:
            g[:, k:] = gs[:, : w - k]
        else:
            g[:, :k] = gs[:, -k:]
        if not g.any() or not ps.any():
            f = 1.0 if (not g.any() and not ps.any()) else 0.0
        else:
            prec = float((_dist_to(g)[ps] <= tol).mean())
            rec = float((d_to_pred[g] <= tol).mean())
            f = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        scores[k] = f
    best = max(scores.values())
    # Several shifts within the tolerance can tie; report the centre of the plateau.
    ties = [k for k, f in scores.items() if f >= best - 1e-9]
    return best, int(np.median(ties))


def score_all(pred: np.ndarray, gt: np.ndarray, occluded: np.ndarray | None = None,
              tol: float = 2.0) -> dict[str, float]:
    """Compute every metric used in the report for one image.

    Returns:
        Dict with keys ``iou, dice, bf1, cl_precision, cl_recall, cl_f1`` and,
        if ``occluded`` is given, ``occ_recall``.
    """
    p, r, f = tolerant_precision_recall(pred, gt, tol=tol, skeleton=True)
    out = {
        "iou": iou(pred, gt),
        "dice": dice(pred, gt),
        "bf1": boundary_f1(pred, gt, tol=tol),
        "cl_precision": p,
        "cl_recall": r,
        "cl_f1": f,
    }
    if occluded is not None:
        out["occ_recall"] = occluded_recall(pred, gt, occluded, tol=tol)
    return out
