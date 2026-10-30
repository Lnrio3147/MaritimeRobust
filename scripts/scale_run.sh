#!/usr/bin/env bash
# Capacity axis: does the severity-5 compound excess shrink as the backbone grows?
#
# Protocol held fixed (40 epochs, batch 32, combined augmentation, matched-marginal
# calibration). Only the backbone changes: 28M -> 50M -> 89M -> 198M parameters.
# The cheap decisive read is the 47-condition overlap decomposition, which gives
# delta for the paper's own haze+glare and haze+cloud compounds; the fifteen-pair
# composition study is only run afterwards if the answer is not already in.
set -u
cd /home/uestc/mount_2T/uestc/lnrio/MaritimeRobust || exit 1
PY=.venv/bin/python

MODELS=${MODELS:-"convnext_small convnext_base convnext_large"}

for M in $MODELS; do
  echo "=== $M train $(date +%H:%M:%S)"
  for f in 0 1 2; do
    for s in 17 29; do
      echo "--- $M fold$f s$s $(date +%H:%M:%S)"
      $PY scripts/explore_train.py --fold "$f" --seed "$s" --method none \
          --model "$M" --epochs 40 --suffix _e40 || { echo "TRAIN FAIL $M $f $s"; exit 1; }
    done
  done

  echo "=== $M overlap grid $(date +%H:%M:%S)"
  for f in 0 1 2; do
    for s in 17 29; do
      $PY scripts/overlap_main_grid.py --fold "$f" --seed "$s" --model "$M" \
          --out-dir "reports/overlap_$M" > "reports/overlap_${M}_f${f}_s${s}.log" 2>&1 \
          || { echo "OVERLAP FAIL $M $f $s"; exit 1; }
    done
  done
  echo "=== $M overlap done $(date +%H:%M:%S)"
done

echo "=== SCALE DONE $(date +%H:%M:%S)"
