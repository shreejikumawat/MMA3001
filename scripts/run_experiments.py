"""Train and evaluate U-Net variants.

Every run writes ``results/runs/<name>/`` containing ``history.json``,
``test_metrics.csv`` (one row per test tree), ``summary.json`` and ``model.pt``.

Examples::

    python scripts/run_experiments.py --name rgbd_s0 --size 128 --depth-input --seed 0
    python scripts/run_experiments.py --name rgb_s0  --size 128 --seed 0
    python scripts/run_experiments.py --name rgbd_frac25 --size 128 --depth-input --train-frac 0.25
    python scripts/run_experiments.py --name rgbd_refined --depth-input --labels data/refined_128.npz

Model selection (early stopping and the decision threshold) uses validation rows
only; the test rows are scored once, after training finishes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

#: Primary matching tolerance (pixels at 256x256). Ground-truth branches are
#: ~11 px wide (audit), so 4 px accepts a centreline that stays inside the branch.
EVAL_TOL = 4.0

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from treeid.metrics import score_all, tolerant_precision_recall  # noqa: E402
from treeid.model import estimate_flops  # noqa: E402
from treeid.splits import split_by_row  # noqa: E402
from treeid.train import TrainConfig, fit, predict_proba  # noqa: E402


def choose_threshold(prob: np.ndarray, y: np.ndarray, grid=np.arange(0.2, 0.81, 0.05)) -> float:
    """Decision threshold maximising mean validation centreline-F1."""
    scores = [np.mean([tolerant_precision_recall(p > t, g)[2] for p, g in zip(prob, y)])
              for t in grid]
    return float(grid[int(np.argmax(scores))])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--depth-input", action="store_true", help="use RGB-D (5 ch) instead of RGB")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--labels", default=None, help="npz with key y replacing training labels")
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--base", type=int, default=16)
    a = ap.parse_args(argv)

    out = Path("results/runs") / a.name
    out.mkdir(parents=True, exist_ok=True)
    d = dict(np.load(f"data/cache_{a.size}.npz"))  # materialise once
    x = d["x"].astype(np.float32)
    if not a.depth_input:
        x = x[:, :3]
    y_raw = d["y"]
    y_train_labels = np.load(a.labels)["y"] if a.labels else y_raw
    tr, va, te = split_by_row(d["row"])
    idx_tr = np.flatnonzero(tr)
    if a.train_frac < 1:
        rng = np.random.default_rng(1234)  # fixed: same subset for every seed
        idx_tr = np.sort(rng.choice(idx_tr, int(round(a.train_frac * len(idx_tr))), replace=False))
    cfg = TrainConfig(base=a.base, seed=a.seed, max_epochs=a.max_epochs, patience=a.patience)
    log_f = open(out / "train.log", "w")

    def log(msg):
        log_f.write(msg + "\n")
        log_f.flush()

    res = fit(x[idx_tr], y_train_labels[idx_tr], x[va], y_train_labels[va], cfg, log=log)
    prob_va = predict_proba(res.model, x[va])
    thr = choose_threshold(prob_va, y_train_labels[va])

    # ---- single, final evaluation on the held-out test rows ----------------------
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    prob_te = predict_proba(res.model, x[te], batch_size=1)
    infer_s = (time.perf_counter() - t0) / te.sum()
    # Score every model at the native 256x256 mask resolution so that runs trained
    # at different input sizes are compared on the same footing.
    d256 = {k: v for k, v in np.load("data/cache_256.npz").items() if k != "x"}
    rows = []
    for i, p in zip(np.flatnonzero(te), prob_te):
        p256 = cv2.resize(p.astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR)
        m = score_all(p256 > thr, d256["y"][i], d256["occ"][i], tol=EVAL_TOL)
        rows.append({"tree_id": str(d["tree"][i]), "row": int(d["row"][i]), **m})
    df = pd.DataFrame(rows)
    df.to_csv(out / "test_metrics.csv", index=False)
    np.savez_compressed(out / "test_prob.npz", prob=prob_te.astype(np.float16),
                        tree=d["tree"][te])
    torch.save(res.model.state_dict(), out / "model.pt")
    fl = estimate_flops(res.model, x.shape[1:])
    summ = {"args": vars(a), "n_train": int(len(idx_tr)), "n_val": int(va.sum()),
            "n_test": int(te.sum()), "threshold": thr, "best_epoch": res.best_epoch,
            "epochs_run": len(res.history), "train_seconds": res.train_seconds,
            "infer_seconds_per_image_1thread": infer_s, **fl,
            "test_mean": df.drop(columns=["tree_id", "row"]).mean(numeric_only=True).to_dict()}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    (out / "history.json").write_text(json.dumps(res.history, indent=1))
    print(json.dumps({k: summ[k] for k in ("threshold", "best_epoch", "train_seconds", "test_mean")},
                     indent=1))


if __name__ == "__main__":
    main()
