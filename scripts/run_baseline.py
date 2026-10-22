"""Tune the classical baseline on training rows and score it on the test rows.

Outputs ``results/runs/baseline/{grid.csv,test_metrics.csv,summary.json}``.

    python scripts/run_baseline.py
"""

from __future__ import annotations

import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from treeid import baseline  # noqa: E402
from treeid.metrics import score_all, tolerant_precision_recall  # noqa: E402
from treeid.splits import split_by_row  # noqa: E402

EVAL_TOL = 4.0


def unpack(xi):
    """Cache entry (5, H, W) -> (rgb01 HxWx3, near HxW, valid HxW bool)."""
    xi = xi.astype(np.float32)
    return xi[:3].transpose(1, 2, 0), xi[3], xi[4] > 0.5


def main():
    out = Path("results/runs/baseline")
    out.mkdir(parents=True, exist_ok=True)
    # Materialise arrays once: indexing an NpzFile re-decompresses the whole array.
    d = dict(np.load("data/cache_256.npz"))
    tr, va, te = split_by_row(d["row"])
    # Tune on every second training tree (deterministic) to bound runtime; the
    # validation rows are not needed because the baseline has no early stopping.
    idx = np.flatnonzero(tr)[::2]
    samples = [(*unpack(d["x"][i]), d["y"][i]) for i in idx]
    metric = lambda p, g: tolerant_precision_recall(p, g, tol=EVAL_TOL)[2]  # noqa: E731
    t0 = time.perf_counter()
    best, table = baseline.grid_search(samples, metric)
    tune_s = time.perf_counter() - t0
    pd.DataFrame(table).sort_values("score", ascending=False).to_csv(out / "grid.csv", index=False)

    rows, times = [], []
    for i in np.flatnonzero(te):
        rgb, near, valid = unpack(d["x"][i])
        t1 = time.perf_counter()
        pred = baseline.predict(rgb, near, valid, best)
        times.append(time.perf_counter() - t1)
        rows.append({"tree_id": str(d["tree"][i]), "row": int(d["row"][i]),
                     **score_all(pred, d["y"][i], d["occ"][i], tol=EVAL_TOL)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "test_metrics.csv", index=False)
    summ = {"params": best.__dict__, "tune_seconds": tune_s, "n_tune": len(idx),
            "infer_seconds_per_image": float(np.mean(times)),
            "test_mean": df.drop(columns=["tree_id", "row"]).mean(numeric_only=True).to_dict()}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
