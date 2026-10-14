#!/usr/bin/env bash
# Confirmation runs (report Section 6): the single 64-px run scored best, so repeat it
# with more seeds, test RGB vs RGB-D at 64 px, and complete the resolution curve at 256 px.
set -u
cd "$(dirname "$0")/.."
run() { n=$1; shift; echo "=== $n $(date +%T)"
        python3 scripts/run_experiments.py --name "$n" "$@" > /dev/null 2> "results/runs/err_$n.txt" || echo "FAILED $n"; }
for s in 1 2; do run rgbd_64_s$s --size 64 --depth-input --seed $s; done
for s in 0 1 2; do run rgb_64_s$s --size 64 --seed $s; done
run rgbd_frac50 --size 128 --depth-input --seed 0 --train-frac 0.5
run rgbd_256 --size 256 --depth-input --seed 0
echo "=== ALL DONE $(date +%T)"
