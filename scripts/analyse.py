"""Aggregate all runs into the tables and figures used in the report.

For every run in ``results/runs`` with saved test probabilities this script
re-scores the test trees at 256x256 with

* strict metrics (labels as supplied),
* the shift-compensated centreline F1 (label offset removed, see metrics.py),
* structural errors: leader-angle error (deg) and fork-height error (px),
* a tolerance sweep (2, 4, 8 px),

then reports mean, 95% bootstrap confidence intervals over test trees, the
seed-to-seed spread, and paired comparisons (Wilcoxon signed-rank) between
RGB-D, RGB and the baseline.

    python scripts/analyse.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from treeid import baseline  # noqa: E402
from treeid.metrics import iou, shift_compensated_f1, tolerant_precision_recall  # noqa: E402
from treeid.skeleton import leader_angle  # noqa: E402
from treeid.splits import split_by_row  # noqa: E402

RUNS = Path("results/runs")
FIG = Path("results/figures")
TOL = 4.0


def boot_ci(v, n=2000, seed=0):
    """95% percentile bootstrap confidence interval of the mean."""
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    means = rng.choice(v, (n, len(v))).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def structure(mask):
    a, info = leader_angle(mask)
    return (np.nan if a is None else a), (np.nan if info["fork_row"] is None else info["fork_row"])


def score_masks(preds, d256, idx):
    """Per-tree metric table for a list of 256x256 predicted masks."""
    rows = []
    for p, i in zip(preds, idx):
        g = d256["y"][i]
        ga, gf = structure(g)
        pa, pf = structure(p)
        comp, k = shift_compensated_f1(p, g, tol=TOL, max_shift=24)
        r = {"tree_id": str(d256["tree"][i]), "iou": iou(p, g), "comp_cl_f1": comp,
             "comp_shift_px": k, "angle_err_deg": abs(pa - ga), "fork_err_px": abs(pf - gf),
             "structure_found": float(not np.isnan(pa))}
        for t in (2, 4, 8):
            r[f"cl_f1_tol{t}"] = tolerant_precision_recall(p, g, tol=t)[2]
        occ = d256["occ"][i]
        from treeid.metrics import occluded_recall
        r["occ_recall"] = occluded_recall(p, g, occ, tol=TOL)
        rows.append(r)
    return pd.DataFrame(rows)


def run_masks(name, d256, te_idx):
    s = json.loads((RUNS / name / "summary.json").read_text())
    pr = np.load(RUNS / name / "test_prob.npz")["prob"].astype(np.float32)
    thr = s["threshold"]
    return [cv2.resize(p, (256, 256), interpolation=cv2.INTER_LINEAR) > thr for p in pr], s


def baseline_masks(d256, te_idx):
    s = json.loads((RUNS / "baseline" / "summary.json").read_text())
    params = baseline.BaselineParams(**s["params"])
    out = []
    for i in te_idx:
        x = d256["x"][i].astype(np.float32)
        out.append(baseline.predict(x[:3].transpose(1, 2, 0), x[3], x[4] > 0.5, params))
    return out, s


def main():
    d256 = dict(np.load("data/cache_256.npz"))  # materialise once (NpzFile re-decompresses)
    _, _, te = split_by_row(d256["row"])
    te_idx = np.flatnonzero(te)
    tables, summaries, masks = {}, {}, {}
    names = ["baseline"] + sorted(p.name for p in RUNS.iterdir()
                                  if (p / "test_prob.npz").exists())
    for n in names:
        if n == "baseline":
            if not (RUNS / "baseline" / "summary.json").exists():
                continue
            m, s = baseline_masks(d256, te_idx)
        else:
            m, s = run_masks(n, d256, te_idx)
        masks[n], summaries[n] = m, s
        tables[n] = score_masks(m, d256, te_idx)
        tables[n].to_csv(RUNS / n / "test_metrics_full.csv", index=False)
        print("scored", n)

    metrics = ["cl_f1_tol4", "comp_cl_f1", "iou", "occ_recall", "angle_err_deg",
               "fork_err_px", "cl_f1_tol2", "cl_f1_tol8", "structure_found"]
    rows = []
    for n, df in tables.items():
        r = {"run": n}
        for m in metrics:
            v = df[m].to_numpy(float)
            r[m] = float(np.nanmean(v))
            lo, hi = boot_ci(v)
            r[m + "_ci"] = f"[{lo:.3f}, {hi:.3f}]"
        s = summaries[n]
        r["train_s"] = s.get("train_seconds", s.get("tune_seconds"))
        r["infer_s_per_img"] = s.get("infer_seconds_per_image_1thread", s.get("infer_seconds_per_image"))
        r["n_train"] = s.get("n_train")
        r["gflops"] = s.get("flops", np.nan) / 1e9 if "flops" in s else np.nan
        rows.append(r)
    table = pd.DataFrame(rows)
    table.to_csv("results/summary_table.csv", index=False)

    # ---- seed aggregation and paired tests --------------------------------------
    agg = {}
    # Seed families: v1 (class-weighted loss, raw labels), v2 (unweighted), v2r (+ refined labels).
    families = {f: [n for n in tables if n.startswith(f + "_s")]
                for f in ("rgbd", "rgb", "v2_rgbd", "v2r_rgbd", "v2r_rgb", "rgb_64")}
    families["rgbd_64"] = [n for n in ("rgbd_64", "rgbd_64_s1", "rgbd_64_s2") if n in tables]
    for fam, seeds in families.items():
        if seeds:
            per_seed = table.set_index("run").loc[seeds, ["cl_f1_tol4", "comp_cl_f1", "iou",
                                                         "angle_err_deg", "fork_err_px"]]
            agg[fam] = {"seeds": seeds, "mean": per_seed.mean().to_dict(),
                        "std_over_seeds": per_seed.std(ddof=1).to_dict() if len(seeds) > 1 else None}
    tests = {}

    def paired(a, b, m):
        x = tables[a][m].to_numpy(float); y = tables[b][m].to_numpy(float)
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < 5:
            return None
        w = stats.wilcoxon(x[ok], y[ok])
        return {"mean_diff": float(np.mean(x[ok] - y[ok])), "p": float(w.pvalue), "n": int(ok.sum())}

    pairs = (("rgbd_s0", "rgb_s0"), ("rgbd_s0", "baseline"), ("v2_rgbd_s0", "rgbd_s0"),
             ("v2r_rgbd_s0", "v2_rgbd_s0"), ("v2r_rgbd_s0", "rgbd_s0"),
             ("v2r_rgbd_s0", "v2r_rgb_s0"), ("v2r_rgbd_s0", "baseline"),
             ("rgbd_64", "rgbd_s0"), ("rgbd_64", "rgb_64_s0"), ("rgbd_64", "baseline"))
    for a, b in pairs:
        if a in tables and b in tables:
            tests[f"{a}_vs_{b}"] = {m: paired(a, b, m) for m in
                                    ("cl_f1_tol4", "comp_cl_f1", "angle_err_deg", "occ_recall")}
    best = "rgbd_64" if "rgbd_64" in tables else "rgbd_s0"
    shifts = tables.get(best, next(iter(tables.values())))["comp_shift_px"]
    report = {"seed_aggregate": agg, "paired_tests": tests,
              "final_model": best,
              "best_shift_px_final_model": {"mean": float(shifts.mean()), "median": float(shifts.median()),
                                        "abs_median": float(shifts.abs().median()),
                                        "frac_abs_gt_4": float((shifts.abs() > 4).mean())}}
    Path("results/analysis.json").write_text(json.dumps(report, indent=2))
    print(table[["run"] + metrics[:6]].round(3).to_string(index=False))
    print(json.dumps(report, indent=1)[:4000])

    # ---- figures -------------------------------------------------------------------
    FIG.mkdir(parents=True, exist_ok=True)
    main_runs = [n for n in ("baseline", "rgbd_s0", "v2r_rgbd_s0", "rgbd_64") if n in masks]
    picks = te_idx[[0, 20, 45, 70, 90]] if len(te_idx) > 90 else te_idx[:5]
    fig, ax = plt.subplots(len(picks), 1 + len(main_runs), figsize=(2.6 * (1 + len(main_runs)), 2.6 * len(picks)))
    for r, i in enumerate(picks):
        rgb = d256["x"][i][:3].astype(np.float32).transpose(1, 2, 0)
        ax[r, 0].imshow(rgb); ax[r, 0].contour(d256["y"][i], [0.5], colors="r", linewidths=0.8)
        ax[r, 0].set_title(f"{d256['tree'][i]} GT", fontsize=8)
        k = list(te_idx).index(i)
        for c, n in enumerate(main_runs, 1):
            ax[r, c].imshow(rgb); ax[r, c].imshow(np.ma.masked_where(~masks[n][k], masks[n][k]),
                                                  cmap="cool", alpha=0.6)
            ax[r, c].contour(d256["y"][i], [0.5], colors="r", linewidths=0.6)
            f = tables[n].iloc[k]
            ax[r, c].set_title(f"{n}  clF1 {f.cl_f1_tol4:.2f} / comp {f.comp_cl_f1:.2f}", fontsize=7)
        for a_ in ax[r]:
            a_.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "qualitative.png", dpi=150); plt.close(fig)

    # 128-px runs only: validation clF1 uses a tolerance in training pixels, so curves at
    # different input sizes are not comparable.
    hist_runs = [n for n in ("rgbd_s0", "v2_rgbd_s0", "v2r_rgbd_s0") if (RUNS / n / "history.json").exists()]
    if hist_runs:
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        for n in hist_runs:
            h = pd.DataFrame(json.loads((RUNS / n / "history.json").read_text()))
            ax[0].plot(h.epoch, h.train_loss, label=n); ax[1].plot(h.epoch, h.val_cl_f1, label=n)
        ax[0].set(xlabel="epoch", ylabel="training loss"); ax[1].set(xlabel="epoch", ylabel="validation clF1")
        for a_ in ax:
            a_.grid(alpha=.3); a_.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(FIG / "training_curves.png", dpi=200); plt.close(fig)

    sens = [(n, summaries[n]) for n in tables if n.startswith("rgbd_frac") or n == "rgbd_s0"]
    if len(sens) > 1:
        xs = [s["n_train"] for _, s in sens]
        fig, ax = plt.subplots(figsize=(4.5, 3))
        for m, lab in (("cl_f1_tol4", "strict clF1"), ("comp_cl_f1", "shift-comp. clF1")):
            ys = [tables[n][m].mean() for n, _ in sens]
            o = np.argsort(xs); ax.plot(np.array(xs)[o], np.array(ys)[o], "o-", label=lab)
        ax.set(xlabel="training trees", ylabel="test score"); ax.grid(alpha=.3); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(FIG / "data_size.png", dpi=200); plt.close(fig)

    tol_runs = [n for n in ("baseline", "rgbd_s0", "rgbd_64") if n in tables]
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for n in tol_runs:
        ax.plot([2, 4, 8], [tables[n][f"cl_f1_tol{t}"].mean() for t in (2, 4, 8)], "o-", label=n)
    ax.set(xlabel="matching tolerance (px)", ylabel="centreline F1"); ax.grid(alpha=.3); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "tolerance.png", dpi=200); plt.close(fig)

    # Resolution study: mean over seeds with min-max bars (seed families from above).
    res_fams = [(64, families.get("rgbd_64", [])), (128, families.get("rgbd", [])),
                (256, [n for n in ("rgbd_256",) if n in tables])]
    res_fams = [(r, f) for r, f in res_fams if f]
    if len(res_fams) > 1:
        fig, ax = plt.subplots(figsize=(4.5, 3))
        for m, lab in (("cl_f1_tol4", "strict clF1"), ("comp_cl_f1", "shift-comp. clF1")):
            mu = [np.mean([tables[n][m].mean() for n in f]) for _, f in res_fams]
            lo = [mu[i] - min(tables[n][m].mean() for n in f) for i, (_, f) in enumerate(res_fams)]
            hi = [max(tables[n][m].mean() for n in f) - mu[i] for i, (_, f) in enumerate(res_fams)]
            ax.errorbar([r for r, _ in res_fams], mu, yerr=[lo, hi], fmt="o-", capsize=3, label=lab)
        ax.set_xscale("log", base=2); ax.set_xticks([r for r, _ in res_fams])
        ax.set_xticklabels([f"{r}\n(n={len(f)})" for r, f in res_fams])
        ax.set(xlabel="network input size (px), n = seeds", ylabel="test score (at 256 px)")
        ax.grid(alpha=.3); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(FIG / "resolution.png", dpi=200); plt.close(fig)

if __name__ == "__main__":
    main()
