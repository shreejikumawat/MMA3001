#!/usr/bin/env bash
# Experiment queue (runs sequentially; ~15 min per 128-px run on 2 CPU cores).
set -u
cd "$(dirname "$0")/.."
run() { n=$1; shift; echo "=== $n $(date +%T)"; python3 scripts/run_experiments.py "$@" > /dev/null 2>results/runs/err_$n.txt || echo "FAILED $n"; }
mkdir -p results/runs
run rgbd_s0 --name rgbd_s0 --size 128 --depth-input --seed 0
run rgb_s0  --name rgb_s0  --size 128 --seed 0
run rgbd_s1 --name rgbd_s1 --size 128 --depth-input --seed 1
run rgb_s1  --name rgb_s1  --size 128 --seed 1
run rgbd_s2 --name rgbd_s2 --size 128 --depth-input --seed 2
run rgb_s2  --name rgb_s2  --size 128 --seed 2
run rgbd_64 --name rgbd_64 --size 64 --depth-input --seed 0
run rgbd_frac25 --name rgbd_frac25 --size 128 --depth-input --seed 0 --train-frac 0.25
run rgbd_frac50 --name rgbd_frac50 --size 128 --depth-input --seed 0 --train-frac 0.5
run rgbd_256 --name rgbd_256 --size 256 --depth-input --seed 0
echo "=== ALL DONE $(date +%T)"
