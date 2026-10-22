"""How much can label mis-registration alone cost? (No model involved.)

The data audit showed ground-truth masks offset horizontally from the visible
trunk. This script measures the score a *perfect* detector would receive if the
labels were displaced by ``k`` pixels: each test mask is shifted by ``k`` and
scored against its unshifted self. The resulting curve converts the audit's
offset estimate into an upper bound on any model's measured score, which is
needed to interpret the test results honestly.

Outputs ``results/label_noise.csv`` and ``results/figures/label_noise.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from treeid.align import apply_shift  # noqa: E402
from treeid.metrics import iou, tolerant_precision_recall  # noqa: E402
from treeid.splits import split_by_row  # noqa: E402

SHIFTS = [0, 2, 4, 6, 8, 12, 16, 20, 24]
TOLS = [2, 4, 8]


def main():
    d = {k: v for k, v in np.load("data/cache_256.npz").items() if k != "x"}
    _, _, te = split_by_row(d["row"])
    rows = []
    for k in SHIFTS:
        for i in np.flatnonzero(te):
            g = d["y"][i]
            p = apply_shift(g, 0, k)
            r = {"shift_px": k, "tree_id": str(d["tree"][i]), "iou": iou(p, g)}
            for t in TOLS:
                r[f"cl_f1_tol{t}"] = tolerant_precision_recall(p, g, tol=t)[2]
            rows.append(r)
    df = pd.DataFrame(rows)
    agg = df.groupby("shift_px").mean(numeric_only=True)
    agg.to_csv("results/label_noise.csv")
    print(agg.round(3))
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(agg.index, agg["iou"], "o-", label="IoU")
    for t in TOLS:
        ax.plot(agg.index, agg[f"cl_f1_tol{t}"], "o-", label=f"centreline F1, tol {t} px")
    ax.set_xlabel("horizontal label offset (px at 256×256)")
    ax.set_ylabel("score of a perfect detector")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("results/figures/label_noise.png", dpi=200)


if __name__ == "__main__":
    main()
