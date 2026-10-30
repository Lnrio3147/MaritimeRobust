#!/usr/bin/env bash
# Coverage study: same augmentation budget, nested pair pools k = 3 -> 6 -> 12.
#
# Every arm applies a compound degradation to exactly half its samples and a
# clean view to the other half, so total compound exposure is identical across
# arms. Only the number of distinct pairs differs. k=3 covers all six families
# once; k=12 is the full lattice minus the three held-out pairs.
#
# Two evaluations per checkpoint:
#   lattice_eval      -- the 15 synthetic pairs, held-out ones flagged
#   overlap_main_grid -- the paper's haze+glare / haze+cloud transfer target
set -u
cd /home/uestc/mount_2T/uestc/lnrio/MaritimeRobust || exit 1
PY=.venv/bin/python
KS=${KS:-"3 6 12"}

for K in $KS; do
  echo "=== c$K train $(date +%H:%M:%S)"
  for f in 0 1 2; do
    for s in 17 29; do
      echo "--- c$K fold$f s$s $(date +%H:%M:%S)"
      $PY scripts/train_coverage.py --fold "$f" --seed "$s" --k "$K" --epochs 40 \
          || { echo "TRAIN FAIL c$K $f $s"; exit 1; }
    done
  done
  echo "=== c$K eval $(date +%H:%M:%S)"
  for f in 0 1 2; do
    for s in 17 29; do
      $PY scripts/lattice_eval.py --fold "$f" --seed "$s" \
          --run-dir "runs/fgsc23_civ8_fold${f}/convnext_tiny_none_c${K}_e40_s${s}" \
          --out-dir "reports/lattice_c${K}" >> "reports/lattice_c${K}_f${f}_s${s}.log" 2>&1 \
          || { echo "LATTICE EVAL FAIL c$K $f $s"; exit 1; }
      $PY scripts/overlap_main_grid.py --fold "$f" --seed "$s" \
          --run-dir "runs/fgsc23_civ8_fold${f}/convnext_tiny_none_c${K}_e40_s${s}" \
          --out-dir "reports/overlap_c${K}" > "reports/overlap_c${K}_f${f}_s${s}.log" 2>&1 \
          || { echo "GRID EVAL FAIL c$K $f $s"; exit 1; }
    done
  done
done
echo "=== COVERAGE DONE $(date +%H:%M:%S)"
