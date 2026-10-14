"""Would the validation rows alone have chosen the same model? (Selection audit.)

The final configuration was identified while looking at test-row results, which
risks optimistic bias. This script re-scores every saved model on the
**validation rows only** (3 and 13), at 256x256 with the 4 px tolerance used for
test reporting, so that the model choice can be justified without the test set.

    python scripts/validation_selection.py   # -> results/validation_selection.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from treeid.metrics import shift_compensated_f1, tolerant_precision_recall  # noqa: E402
from treeid.model import UNet  # noqa: E402
from treeid.splits import split_by_row  # noqa: E402
from treeid.train import predict_proba  # noqa: E402

FAMILIES = {
    "v1 RGB-D 64 px": ["rgbd_64", "rgbd_64_s1", "rgbd_64_s2"],
    "v1 RGB 64 px": ["rgb_64_s0", "rgb_64_s1", "rgb_64_s2"],
    "v1 RGB-D 128 px": ["rgbd_s0", "rgbd_s1", "rgbd_s2"],
    "v1 RGB 128 px": ["rgb_s0", "rgb_s1", "rgb_s2"],
    "v2 RGB-D 128 px": ["v2_rgbd_s0", "v2_rgbd_s1", "v2_rgbd_s2"],
    "v2r RGB-D 128 px": ["v2r_rgbd_s0", "v2r_rgbd_s1", "v2r_rgbd_s2"],
    "v1 RGB-D 256 px": ["rgbd_256"],
}


def main():
    torch.set_num_threads(2)
    y256 = np.load("data/cache_256.npz")["y"]
    caches = {}
    rows = []
    for fam, runs in FAMILIES.items():
        for n in runs:
            p = Path("results/runs") / n
            if not (p / "model.pt").exists():
                continue
            s = json.loads((p / "summary.json").read_text())
            size, depth = s["args"]["size"], s["args"]["depth_input"]
            if size not in caches:
                caches[size] = dict(np.load(f"data/cache_{size}.npz"))
            d = caches[size]
            _, va, _ = split_by_row(d["row"])
            x = d["x"][va].astype(np.float32)
            x = x if depth else x[:, :3]
            m = UNet(x.shape[1], s["args"]["base"], 3)
            m.load_state_dict(torch.load(p / "model.pt", map_location="cpu"))
            prob = predict_proba(m, x)
            strict, comp = [], []
            for pr, i in zip(prob, np.flatnonzero(va)):
                mask = cv2.resize(pr, (256, 256), interpolation=cv2.INTER_LINEAR) > s["threshold"]
                strict.append(tolerant_precision_recall(mask, y256[i], tol=4)[2])
                comp.append(shift_compensated_f1(mask, y256[i], tol=4)[0])
            rows.append({"family": fam, "run": n, "val_strict_cl_f1": float(np.mean(strict)),
                         "val_comp_cl_f1": float(np.mean(comp))})
            print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv("results/validation_selection.csv", index=False)
    agg = df.groupby("family")[["val_strict_cl_f1", "val_comp_cl_f1"]].agg(["mean", "std", "count"])
    print(agg.round(3).sort_values(("val_strict_cl_f1", "mean"), ascending=False))
    agg.round(4).to_csv("results/validation_selection_by_family.csv")


if __name__ == "__main__":
    main()
