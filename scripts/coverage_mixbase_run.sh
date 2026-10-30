#!/usr/bin/env bash
# c12b: the project's combined recipe (haze/glare/cloud) kept in the mix, plus the
# full 12-pair lattice in the other half of the draws.
#
# The reason this arm exists: the pure-lattice arms replace the combined recipe,
# which removes haze and glare from training entirely and drops haze+glare_5
# accuracy from 59 to 35. That makes the transfer comparison meaningless. Here
# both haze+glare and the lattice are trained, so a change in the haze+glare
# excess is attributable to the lattice and nothing else.
set -u
cd /home/uestc/mount_2T/uestc/lnrio/MaritimeRobust || exit 1
PY=.venv/bin/python

echo "=== c12b train $(date +%H:%M:%S)"
for f in 0 1 2; do
  for s in 17 29; do
    echo "--- c12b fold$f s$s $(date +%H:%M:%S)"
    $PY scripts/train_coverage.py --fold "$f" --seed "$s" --k 12 --mix-base --epochs 40 \
        || { echo "TRAIN FAIL c12b $f $s"; exit 1; }
  done
done
echo "=== c12b eval $(date +%H:%M:%S)"
for f in 0 1 2; do
  for s in 17 29; do
    $PY scripts/lattice_eval.py --fold "$f" --seed "$s" \
        --run-dir "runs/fgsc23_civ8_fold${f}/convnext_tiny_none_c12b_e40_s${s}" \
        --out-dir "reports/lattice_c12b" >> "reports/lattice_c12b_f${f}_s${s}.log" 2>&1 \
        || { echo "LATTICE EVAL FAIL c12b $f $s"; exit 1; }
    $PY scripts/overlap_main_grid.py --fold "$f" --seed "$s" \
        --run-dir "runs/fgsc23_civ8_fold${f}/convnext_tiny_none_c12b_e40_s${s}" \
        --out-dir "reports/overlap_c12b" > "reports/overlap_c12b_f${f}_s${s}.log" 2>&1 \
        || { echo "GRID EVAL FAIL c12b $f $s"; exit 1; }
  done
done
echo "=== C12B DONE $(date +%H:%M:%S)"
