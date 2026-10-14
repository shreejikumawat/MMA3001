"""Compare hand labels and model predictions with manually marked trunk positions.

Input: ``results/trunk_annotations.csv`` written by ``doc/trunk_annotator.html``.
For each annotated test tree the trunk centre of a mask is the mean column of
the mask pixels on row 230 (largest run). The script reports, against the
manual mark, the absolute horizontal error of:

* the supplied ground-truth mask (how far the labels are from the visible trunk),
* each model's prediction.

If a model is closer to the visible trunk than the label is, the strict scores in
the report under-state that model's accuracy.

    python scripts/trunk_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from treeid.splits import split_by_row  # noqa: E402

Y_LINE = 230
RUNS = ["rgbd_s0", "v2r_rgbd_s0", "rgbd_64", "v2r_rgbd_64"]


def trunk_x(mask: np.ndarray, y: int = Y_LINE) -> float:
    """Centre column of the widest run of ``mask`` on row ``y`` (nan if empty)."""
    row = mask[y].astype(np.int8)
    if not row.any():
        return float("nan")
    edges = np.flatnonzero(np.diff(np.r_[0, row, 0]))
    starts, ends = edges[::2], edges[1::2]
    j = int(np.argmax(ends - starts))
    return (starts[j] + ends[j] - 1) / 2


def main():
    ann = pd.read_csv("results/trunk_annotations.csv")
    ann = ann[ann.visible == 1]
    d = {k: v for k, v in np.load("data/cache_256.npz").items() if k != "x"}
    _, _, te = split_by_row(d["row"])
    te_idx = np.flatnonzero(te)
    pos = {str(d["tree"][i]): k for k, i in enumerate(te_idx)}
    rows = []
    preds = {}
    for n in RUNS:
        p = Path("results/runs") / n
        if (p / "test_prob.npz").exists():
            thr = json.loads((p / "summary.json").read_text())["threshold"]
            preds[n] = np.load(p / "test_prob.npz")["prob"].astype(np.float32), thr
    for _, a in ann.iterrows():
        k = pos[a.tree_id]
        r = {"tree_id": a.tree_id, "manual_x": a.x_px_256,
             "gt_err": abs(trunk_x(d["y"][te_idx[k]]) - a.x_px_256)}
        for n, (pr, thr) in preds.items():
            m = cv2.resize(pr[k], (256, 256), interpolation=cv2.INTER_LINEAR) > thr
            r[f"{n}_err"] = abs(trunk_x(m) - a.x_px_256)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv("results/trunk_check.csv", index=False)
    summ = {c: {"median_px": float(df[c].median()), "mean_px": float(df[c].mean()),
                "frac_within_5px": float((df[c] <= 5).mean())}
            for c in df.columns if c.endswith("_err")}
    summ["n_annotated_visible"] = int(len(df))
    Path("results/trunk_check.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
