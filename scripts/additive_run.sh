#!/usr/bin/env bash
# Additivity-constrained training (B1), queued after the capacity study.
#
# a0 = control (same data, same compound CE, no residual term)
# a1 = a0 + lambda * ||phi(a+b) - phi(a) - phi(b) + phi(clean)||^2 / scale^2
#
# Both arms are evaluated on the full 47-condition grid, so the comparison is
# in compound accuracy and in delta, not only in clean accuracy.
set -u
cd /home/uestc/mount_2T/uestc/lnrio/MaritimeRobust || exit 1
PY=.venv/bin/python
LAM=${LAM:-0.1}
ARMS=${ARMS:-"a0 a1"}

for A in $ARMS; do
  echo "=== $A train $(date +%H:%M:%S)"
  for f in 0 1 2; do
    for s in 17 29; do
      echo "--- $A fold$f s$s $(date +%H:%M:%S)"
      $PY scripts/train_additive.py --fold "$f" --seed "$s" --arm "$A" \
          --epochs 40 --lam "$LAM" --suffix _e40 \
          || { echo "TRAIN FAIL $A $f $s"; exit 1; }
    done
  done
  echo "=== $A eval $(date +%H:%M:%S)"
  for f in 0 1 2; do
    for s in 17 29; do
      $PY scripts/overlap_main_grid.py --fold "$f" --seed "$s" \
          --run-dir "runs/fgsc23_civ8_fold${f}/convnext_tiny_${A}_e40_s${s}" \
          --out-dir "reports/overlap_$A" > "reports/overlap_${A}_f${f}_s${s}.log" 2>&1 \
          || { echo "EVAL FAIL $A $f $s"; exit 1; }
    done
  done
done
echo "=== ADDITIVE DONE $(date +%H:%M:%S)"
