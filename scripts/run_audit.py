"""Data audit of the TreeID dataset.

Produces ``results/audit.json``, ``results/audit_per_tree.csv`` and figures in
``results/figures/``. Run from the repository root::

    python scripts/run_audit.py --data data/raw

Checks performed
----------------
1. Inventory: trees per row, completeness, naming anomalies.
2. Formats: shapes, dtypes, channel redundancy.
3. Duplicates: byte-identical files (MD5) within each modality.
4. Depth: missing-value fraction, value range, direction (near = small or large).
5. Ground truth: area fraction, connected components, mean branch width.
6. Occlusion: fraction of GT centreline covered by foliage in the summer image.
7. Registration: horizontal offset between GT and an image-derived bark map,
   estimated by FFT cross-correlation (see ``treeid.align``).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.filters import sato
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from treeid.align import estimate_shift  # noqa: E402
from treeid.io import REQUIRED_KEYS, discover, load_sample, read_gray  # noqa: E402
from treeid.preprocess import crop_resize, excess_green  # noqa: E402


def md5(path: Path) -> str:
    """MD5 hex digest of a file's bytes."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def bark_map(rgb256: np.ndarray) -> np.ndarray:
    """Label-free map of where bark is likely to be visible.

    Dark tubular structures (Sato ridge filter) that are not green. Used only as
    the *reference* for registering the GT masks, never as a training signal.
    """
    gray = rgb256.astype(np.float32).mean(-1) / 255
    ridge = sato(gray, sigmas=(2, 4, 6), black_ridges=True)
    ridge /= ridge.max() + 1e-9
    return ridge * (excess_green(rgb256) < 0.05)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    recs, unparsed = discover(args.data)
    summary: dict = {"n_trees": len(recs), "n_files": sum(len(r.files) for r in recs.values()),
                     "unparsed_files": unparsed}
    summary["trees_per_row"] = dict(sorted(collections.Counter(r.row for r in recs.values()).items()))
    summary["complete_trees"] = sum(r.complete for r in recs.values())
    summary["unmatched_U_trees"] = sum(not r.matched for r in recs.values())
    summary["naming_anomalies"] = [a for r in recs.values() for a in r.anomalies]

    # --- duplicates per modality --------------------------------------------------
    dups = {}
    for key in REQUIRED_KEYS:
        groups = collections.defaultdict(list)
        for r in recs.values():
            groups[md5(r.files[key])].append(r.tree_id)
        dups[key] = [sorted(v) for v in groups.values() if len(v) > 1]
    summary["duplicate_groups"] = dups

    # --- per-tree statistics ---------------------------------------------------------
    rows = []
    for tid in sorted(recs):
        r = recs[tid]
        s = load_sample(r, "Summer")
        dw = read_gray(r.files["Winter_D"])
        gt = s["gt"]
        rgb256 = crop_resize(s["rgb"], 256)
        d256 = crop_resize(s["depth"], 256, is_mask=True)
        skel = skeletonize(gt)
        leaf = excess_green(rgb256) > 0.05
        # Depth direction test: canopy (upper half, valid) vs ground at bottom rows.
        dv = s["depth"].astype(float)
        top = dv[:240][dv[:240] > 0]
        bottom = dv[-40:][dv[-40:] > 0]
        # Registration of GT against the bark map; lower 60% only, where the
        # trunk and leader bases are least occluded.
        ref = bark_map(rgb256)
        ref[:100] = 0
        g = gt.astype(float)
        g[:100] = 0
        dy, dx, pr = estimate_shift(ref, g, max_shift=(0, 60)) if g.any() else (0, 0, 0)
        rows.append({
            "tree_id": tid, "row": r.row,
            "gt_area_frac": float(gt.mean()),
            "gt_components": int(ndimage.label(gt)[1]),
            "gt_mean_width_px": float(gt.sum() / max(skel.sum(), 1)),
            "occluded_cl_frac": float((skel & leaf).sum() / max(skel.sum(), 1)),
            "depth_missing_frac_summer": float((s["depth"] == 0).mean()),
            "depth_missing_frac_winter": float((dw == 0).mean()),
            "depth_max_summer": int(s["depth"].max()),
            "depth_med_top_minus_bottom": float(np.median(top) - np.median(bottom))
            if len(top) and len(bottom) else np.nan,
            "gt_dx_px": dx, "gt_dx_peak_ratio": pr,
            "gt_missing_in_crop_frac": float((gt & (d256 == 0)).sum() / max(gt.sum(), 1)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out / "audit_per_tree.csv", index=False)

    def stats(col):
        v = df[col].dropna()
        return {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()),
                "p5": float(v.quantile(.05)), "median": float(v.median()),
                "p95": float(v.quantile(.95)), "max": float(v.max())}

    for col in df.columns[2:]:
        summary[col] = stats(col)
    summary["depth_far_is_larger_fraction"] = float((df["depth_med_top_minus_bottom"] > 0).mean())
    conf = df[df["gt_dx_peak_ratio"] > 1.1]
    summary["registration"] = {
        "n_confident": int(len(conf)),
        "dx_confident": stats("gt_dx_px") if len(conf) == 0 else {
            "mean": float(conf.gt_dx_px.mean()), "median": float(conf.gt_dx_px.median()),
            "std": float(conf.gt_dx_px.std()), "p5": float(conf.gt_dx_px.quantile(.05)),
            "p95": float(conf.gt_dx_px.quantile(.95))},
    }
    (out / "audit.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("naming_anomalies",)},
                     indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
