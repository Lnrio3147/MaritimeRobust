#!/usr/bin/env bash
# s6 control: the six families, one per draw, same compound/clean rate.
#
# Runs after coverage_run.sh because both take the GPU lock. The comparison this
# exists for is s6 vs c12: identical family exposure, identical corruption rate,
# identical budget -- only pairing removed. Anything c12 wins over s6 is the
# contribution of covering *pairs*; anything s6 already wins is just new
# augmentation families, which is a known effect and not the claim.
set -u
cd /home/uestc/mount_2T/uestc/lnrio/MaritimeRobust || exit 1
PY=.venv/bin/python

echo "=== s6 train $(date +%H:%M:%S)"
for f in 0 1 2; do
  for s in 17 29; do
    echo "--- s6 fold$f s$s $(date +%H:%M:%S)"
    if [ -f "runs/fgsc23_civ8_fold$f/convnext_tiny_none_s6_e40_s$s/complete.json" ]; then
      echo "    (already trained, skipping)"; continue
    fi
    $PY scripts/train_coverage.py --fold "$f" --seed "$s" --k 12 --singles --epochs 40 \
        || { echo "TRAIN FAIL s6 $f $s"; exit 1; }
  done
done
echo "=== s6 eval $(date +%H:%M:%S)"
for f in 0 1 2; do
  for s in 17 29; do
    [ -f "reports/lattice_s6/fold${f}_s${s}.json" ] && { echo "    (already evaluated)"; continue; }
    $PY scripts/lattice_eval.py --fold "$f" --seed "$s" \
        --run-dir "runs/fgsc23_civ8_fold${f}/convnext_tiny_none_s6_e40_s${s}" \
        --out-dir "reports/lattice_s6" >> "reports/lattice_s6_f${f}_s${s}.log" 2>&1 \
        || { echo "LATTICE EVAL FAIL s6 $f $s"; exit 1; }
    $PY scripts/overlap_main_grid.py --fold "$f" --seed "$s" \
        --run-dir "runs/fgsc23_civ8_fold${f}/convnext_tiny_none_s6_e40_s${s}" \
        --out-dir "reports/overlap_s6" > "reports/overlap_s6_f${f}_s${s}.log" 2>&1 \
        || { echo "GRID EVAL FAIL s6 $f $s"; exit 1; }
  done
done
echo "=== S6 DONE $(date +%H:%M:%S)"
