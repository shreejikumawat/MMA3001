#!/usr/bin/env bash
# Second queue: label-refinement experiment (needs results/runs/rgbd_s0 and data/refined_128.npz).
set -u
cd "$(dirname "$0")/.."
while ! grep -q "ALL DONE" results/queue.log; do sleep 30; done
for s in 0 1; do
  echo "=== rgbd_refined_s$s $(date +%T)"
  python3 scripts/run_experiments.py --name rgbd_refined_s$s --size 128 --depth-input --seed $s \
      --labels data/refined_128.npz > /dev/null 2> results/runs/err_rgbd_refined_s$s.txt || echo FAILED
done
echo "=== QUEUE2 DONE $(date +%T)"
