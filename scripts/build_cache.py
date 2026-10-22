"""Pre-process every tree once and store network-ready arrays.

Creates ``data/cache_<size>.npz`` with:

* ``x``     float16 (N, 5, S, S): RGB in [0,1], depth-near rank, depth-valid flag
* ``y``     bool    (N, S, S): ground-truth branch mask (as supplied)
* ``occ``   bool    (N, S, S): foliage map (excess green > 0.05) used for occluded recall
* ``tree``  str     (N,): tree identifiers;  ``row`` int (N,)

Depth direction: the audit (``results/audit.json``, key
``depth_far_is_larger_fraction`` = 0.996) shows larger 8-bit values are *farther*,
so ``near_is_bright=False``.

Usage::

    python scripts/build_cache.py --size 128
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from treeid.io import discover, load_sample  # noqa: E402
from treeid.preprocess import crop_resize, foliage_mask, make_input, resize_mask  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--out", default="data")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    recs, _ = discover(a.data)
    ids = sorted(recs)
    xs, ys, occ, rows = [], [], [], []
    for t in ids:
        s = load_sample(recs[t])
        xs.append(make_input(s["rgb"], s["depth"], a.size, use_depth=True, near_is_bright=False))
        ys.append(resize_mask(s["gt"], a.size))
        occ.append(foliage_mask(crop_resize(s["rgb"], a.size)))
        rows.append(recs[t].row)
    out = Path(a.out) / f"cache_{a.size}.npz"
    np.savez_compressed(out, x=np.stack(xs).astype(np.float16), y=np.stack(ys),
                        occ=np.stack(occ), tree=np.array(ids), row=np.array(rows))
    print(f"wrote {out} ({len(ids)} trees) in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
