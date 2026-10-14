"""Profiling, timing and roofline analysis of the end-to-end pipeline.

Answers three questions for the report's performance section:

1. Where does the time go for one image? (``cProfile`` of load -> preprocess ->
   predict -> structure extraction.)
2. How do runtime, FLOPs and arithmetic intensity scale with input resolution,
   and how close is the network to the machine's attainable FLOP rate (roofline,
   Week 2.1)?
3. How does the classical baseline compare in time and memory?

Run *after* training, with nothing else using the CPU::

    python scripts/profile_pipeline.py --model results/runs/rgbd_64/model.pt --size 64

Outputs ``results/profile.json``, ``results/profile_cprofile.txt`` and
``results/figures/roofline.png``.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import platform
import pstats
import sys
import time
import tracemalloc
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from treeid import baseline  # noqa: E402
from treeid.io import discover, load_sample  # noqa: E402
from treeid.model import UNet, estimate_flops  # noqa: E402
from treeid.preprocess import make_input  # noqa: E402
from treeid.skeleton import leader_angle  # noqa: E402


def timeit(fn, reps=20, warmup=3):
    """Median wall time of ``fn()`` in seconds over ``reps`` repetitions."""
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return float(np.median(t))


def attainable_gflops(threads: int) -> float:
    """Empirical compute ceiling: large float32 matrix multiply (GFLOP/s)."""
    torch.set_num_threads(threads)
    n = 1024
    a, b = torch.randn(n, n), torch.randn(n, n)
    s = timeit(lambda: a @ b, reps=10)
    return 2 * n ** 3 / s / 1e9


def attainable_bandwidth() -> float:
    """Empirical memory ceiling: large array copy (GB/s, read + write)."""
    x = np.ones(50_000_000, np.float32)
    y = np.empty_like(x)
    s = timeit(lambda: np.copyto(y, x), reps=5)
    return 2 * x.nbytes / s / 1e9


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/runs/rgbd_64/model.pt")
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--tree", default="R10N05")
    ap.add_argument("--size", type=int, default=64, help="input size of the profiled model")
    a = ap.parse_args(argv)
    out: dict = {"platform": platform.platform(), "processor": platform.processor(),
                 "torch": torch.__version__, "cpu_threads_available": torch.get_num_threads()}

    recs, _ = discover(a.data)
    rec = recs[a.tree]
    model = UNet(5, 16, 3)
    model.load_state_dict(torch.load(a.model, map_location="cpu"))
    model.eval()

    # ---- 1. cProfile of the full single-image pipeline ---------------------------
    torch.set_num_threads(1)

    def pipeline():
        s = load_sample(rec)
        x = make_input(s["rgb"], s["depth"], a.size, near_is_bright=False)
        with torch.no_grad():
            p = torch.sigmoid(model(torch.from_numpy(x)[None]))[0, 0].numpy()
        return leader_angle(p > 0.5)

    pipeline()
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(10):
        pipeline()
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(25)
    Path("results/profile_cprofile.txt").write_text(buf.getvalue())

    s = load_sample(rec)
    stages = {
        "load_png": timeit(lambda: load_sample(rec), reps=10),
        f"preprocess_{a.size}": timeit(lambda: make_input(s["rgb"], s["depth"], a.size, near_is_bright=False)),
    }
    x128 = make_input(s["rgb"], s["depth"], a.size, near_is_bright=False)
    with torch.no_grad():
        stages[f"unet_forward_{a.size}_1thread"] = timeit(lambda: model(torch.from_numpy(x128)[None]))
        p = torch.sigmoid(model(torch.from_numpy(x128)[None]))[0, 0].numpy()
    stages["structure_extraction"] = timeit(lambda: leader_angle(p > 0.5), reps=10)
    out["stage_seconds"] = stages

    # ---- 2. resolution / thread scaling and roofline ----------------------------
    peak = {t: attainable_gflops(t) for t in (1, 2)}
    bw = attainable_bandwidth()
    out["attainable_gflops"] = peak
    out["attainable_bandwidth_GBs"] = bw
    scale = []
    for size in (64, 128, 256):
        m = UNet(5, 16, 3).eval()
        fl = estimate_flops(m, (5, size, size))
        xs = torch.zeros(1, 5, size, size)
        for th in (1, 2):
            torch.set_num_threads(th)
            with torch.no_grad():
                sec = timeit(lambda: m(xs), reps=10)
            scale.append({"size": size, "threads": th, "seconds": sec,
                          "gflops_achieved": fl["flops"] / sec / 1e9,
                          "frac_of_peak": fl["flops"] / sec / 1e9 / peak[th], **fl})
    out["scaling"] = scale
    torch.set_num_threads(1)

    # ---- 3. baseline vs U-Net time and memory ------------------------------------
    x256 = make_input(s["rgb"], s["depth"], 256, near_is_bright=False)
    rgb01, near, valid = x256[:3].transpose(1, 2, 0), x256[3], x256[4] > 0.5
    params = baseline.BaselineParams()
    try:
        params = baseline.BaselineParams(**json.loads(
            Path("results/runs/baseline/summary.json").read_text())["params"])
    except FileNotFoundError:
        pass
    out["baseline_seconds_256"] = timeit(lambda: baseline.predict(rgb01, near, valid, params), reps=10)
    tracemalloc.start()
    baseline.predict(rgb01, near, valid, params)
    out["baseline_peak_python_MB"] = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    out["unet_weights_MB"] = sum(p.numel() * 4 for p in model.parameters()) / 1e6
    # Activation memory: sum of all intermediate conv outputs for one 128 image.
    acts = []
    hooks = [mm.register_forward_hook(lambda mod, i, o: acts.append(o.numel() * 4))
             for mm in model.modules() if isinstance(mm, torch.nn.Conv2d)]
    with torch.no_grad():
        model(torch.from_numpy(x128)[None])
    for h in hooks:
        h.remove()
    out[f"unet_activation_MB_{a.size}"] = sum(acts) / 1e6
    Path("results/profile.json").write_text(json.dumps(out, indent=2))

    # ---- roofline figure ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ai = np.logspace(-1, 3, 200)
    for th, ls in ((1, "--"), (2, "-")):
        ax.loglog(ai, np.minimum(peak[th], bw * ai), "k", ls=ls, lw=1,
                  label=f"roofline, {th} thread{'s' if th > 1 else ''}")
    for r in scale:
        ax.loglog(r["arith_intensity"], r["gflops_achieved"], "o" if r["threads"] == 1 else "s",
                  label=f"U-Net {r['size']}px, {r['threads']}t")
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("GFLOP/s")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig("results/figures/roofline.png", dpi=200)
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
