"""Model-based re-registration of the training labels (label refinement).

Hypothesis (from the audit and Chen et al., 2021): the GT masks were hand-drawn on
a different capture of each tree, so they are displaced horizontally from the
summer image by a tree-specific amount. A network trained on such labels learns
the *average* displacement and blurs the leader position.

Procedure (one round of expectation-maximisation-style label refinement):

1. Predict every **training and validation** tree with a model trained on the raw
   labels (``--model``).
2. For each tree, find the horizontal shift of its GT (within ±24 px at 256x256)
   that maximises the centreline F1 against the prediction
   (:func:`treeid.metrics.shift_compensated_f1`).
3. Accept the shift only if it improves the F1 by at least ``--min-gain``;
   otherwise keep the original label (guards against fitting noise).
4. Save the shifted labels at 128x128 to ``data/refined_128.npz``.

Test labels are **never** modified, so the test set remains independent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from treeid.align import apply_shift  # noqa: E402
from treeid.metrics import shift_compensated_f1, tolerant_precision_recall  # noqa: E402
from treeid.model import UNet  # noqa: E402
from treeid.preprocess import resize_mask  # noqa: E402
from treeid.splits import split_by_row  # noqa: E402
from treeid.train import predict_proba  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/runs/rgbd_s0/model.pt")
    ap.add_argument("--min-gain", type=float, default=0.05)
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    d128 = dict(np.load("data/cache_128.npz"))
    y256 = np.load("data/cache_256.npz")["y"]
    thr = json.loads(Path(a.model).with_name("summary.json").read_text())["threshold"]
    tr, va, te = split_by_row(d128["row"])
    model = UNet(5, 16, 3)
    model.load_state_dict(torch.load(a.model, map_location="cpu"))
    y_new = d128["y"].copy()
    log = []
    for i in np.flatnonzero(tr | va):
        p = predict_proba(model, d128["x"][i: i + 1].astype(np.float32))[0]
        p256 = cv2.resize(p, (256, 256), interpolation=cv2.INTER_LINEAR) > thr
        g = y256[i]
        before = tolerant_precision_recall(p256, g, tol=4)[2]
        after, k = shift_compensated_f1(p256, g, tol=4, max_shift=24)
        accepted = (after - before) >= a.min_gain and k != 0
        if accepted:
            y_new[i] = resize_mask(apply_shift(g, 0, k), 128)
        log.append({"tree": str(d128["tree"][i]), "shift_px": int(k), "f1_before": before,
                    "f1_after": after, "accepted": bool(accepted)})
    np.savez_compressed("data/refined_128.npz", y=y_new)
    acc = [r for r in log if r["accepted"]]
    summ = {"n_considered": len(log), "n_accepted": len(acc),
            "median_abs_shift_accepted": float(np.median([abs(r["shift_px"]) for r in acc])) if acc else 0,
            "shift_hist": np.histogram([r["shift_px"] for r in acc], bins=range(-25, 26, 5))[0].tolist()}
    Path("results/label_refinement.json").write_text(json.dumps({"summary": summ, "per_tree": log}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
