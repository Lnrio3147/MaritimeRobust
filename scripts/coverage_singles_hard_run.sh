#!/usr/bin/env bash
# s6h: the six families as singles at *double* damage, one per draw.
#
# The confound this removes: every pair arm applies two families per image, so a
# compound training image carries roughly twice the damage of a single-family one.
# Any pair-arm gain over the s6 control can therefore be read as "harder
# augmentation", not "pairing". s6h matches the pair arms on damage per image by
# moving the level ladder up instead of stacking a second family, so the only
# difference left between c3 and s6h is whether the two degradations co-occur on
# the same image.
#
# Levels come from lattice_calibrate.py at the same target damage a pair carries
# (~40 pp, twice the matched-20pp component level).
set -u
cd /home/uestc/mount_2T/uestc/lnrio/MaritimeRobust || exit 1
PY=.venv/bin/python

if [ ! -f reports/lattice_levels_40.json ]; then
  echo "missing reports/lattice_levels_40.json -- run lattice_calibrate.py first"; exit 1
fi

echo "=== s6h train $(date +%H:%M:%S)"
for f in 0 1 2; do
  for s in 17 29; do
    echo "--- s6h fold$f s$s $(date +%H:%M:%S)"
    if [ -f "runs/fgsc23_civ8_fold$f/convnext_tiny_none_s6h_e40_s$s/complete.json" ]; then
      echo "    (already trained, skipping)"; continue
    fi
    $PY scripts/train_coverage.py --fold "$f" --seed "$s" --k 12 --singles \
        --levels reports/lattice_levels_40.json --suffix _s6h_e40 --epochs 40 \
        || { echo "TRAIN FAIL s6h $f $s"; exit 1; }
  done
done

echo "=== s6h eval $(date +%H:%M:%S)"
for f in 0 1 2; do
  for s in 17 29; do
    [ -f "reports/lattice_s6h/fold${f}_s${s}.json" ] && { echo "    (already evaluated)"; continue; }
    $PY scripts/lattice_eval.py --fold "$f" --seed "$s" \
        --run-dir "runs/fgsc23_civ8_fold${f}/convnext_tiny_none_s6h_e40_s${s}" \
        --out-dir "reports/lattice_s6h" >> "reports/lattice_s6h_f${f}_s${s}.log" 2>&1 \
        || { echo "LATTICE EVAL FAIL s6h $f $s"; exit 1; }
  done
done
echo "=== S6H DONE $(date +%H:%M:%S)"
