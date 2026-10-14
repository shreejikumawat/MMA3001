#!/usr/bin/env bash
# v2 experiment queue (report Section 6). Runs sequentially; ~10-15 min per 128-px run
# on 2 CPU cores. v1 runs (scripts/queue.sh) used the default pos_weight (~12) and raw labels.
#
#   v2   : pos_weight = 1, raw labels           -> isolates the loss-weighting change
#   v2r  : pos_weight = 1, refined train labels -> adds label refinement (scripts/refine_labels.py)
#   then sensitivity studies on the v2r RGB-D configuration.
set -u
cd "$(dirname "$0")/.."
# Wait for any run still in progress from the v1 queue.
while pgrep -f "scripts/run_experiments.py" > /dev/null; do sleep 20; done
run() { n=$1; shift; echo "=== $n $(date +%T)"
        python3 scripts/run_experiments.py --name "$n" "$@" > /dev/null 2> "results/runs/err_$n.txt" || echo "FAILED $n"; }
REF="--labels data/refined_128.npz"
for s in 0 1 2; do run v2_rgbd_s$s  --size 128 --depth-input --seed $s --pos-weight 1; done
for s in 0 1 2; do run v2r_rgbd_s$s --size 128 --depth-input --seed $s --pos-weight 1 $REF; done
for s in 0 1 2; do run v2r_rgb_s$s  --size 128 --seed $s --pos-weight 1 $REF; done
run v2r_rgbd_frac25 --size 128 --depth-input --seed 0 --pos-weight 1 $REF --train-frac 0.25
run v2r_rgbd_frac50 --size 128 --depth-input --seed 0 --pos-weight 1 $REF --train-frac 0.5
run v2r_rgbd_64     --size 64  --depth-input --seed 0 --pos-weight 1 --labels data/refined_64.npz
echo "=== ALL DONE $(date +%T)"
