#!/usr/bin/env bash
# s6b: combined recipe kept, plus the six families as singles in the other half.
#
# This is the control that makes the coverage claim interpretable. c12b keeps the
# recipe and adds pairs; s6b keeps the recipe and adds the same families without
# pairing. baseline -> s6b is "more families, recipe intact"; s6b -> c12b is
# "pairing, with everything else held fixed".
set -u
cd /home/uestc/mount_2T/uestc/lnrio/MaritimeRobust || exit 1
PY=.venv/bin/python

echo "=== s6b train $(date +%H:%M:%S)"
for f in 0 1 2; do
  for s in 17 29; do
    echo "--- s6b fold$f s$s $(date +%H:%M:%S)"
    $PY scripts/train_coverage.py --fold "$f" --seed "$s" --k 12 --singles --mix-base \
        --epochs 40 || { echo "TRAIN FAIL s6b $f $s"; exit 1; }
  done
done
echo "=== s6b eval $(date +%H:%M:%S)"
for f in 0 1 2; do
  for s in 17 29; do
    $PY scripts/lattice_eval.py --fold "$f" --seed "$s" \
        --run-dir "runs/fgsc23_civ8_fold${f}/convnext_tiny_none_s6b_e40_s${s}" \
        --out-dir "reports/lattice_s6b" >> "reports/lattice_s6b_f${f}_s${s}.log" 2>&1 \
        || { echo "LATTICE EVAL FAIL s6b $f $s"; exit 1; }
    $PY scripts/overlap_main_grid.py --fold "$f" --seed "$s" \
        --run-dir "runs/fgsc23_civ8_fold${f}/convnext_tiny_none_s6b_e40_s${s}" \
        --out-dir "reports/overlap_s6b" > "reports/overlap_s6b_f${f}_s${s}.log" 2>&1 \
        || { echo "GRID EVAL FAIL s6b $f $s"; exit 1; }
  done
done
echo "=== S6B DONE $(date +%H:%M:%S)"
