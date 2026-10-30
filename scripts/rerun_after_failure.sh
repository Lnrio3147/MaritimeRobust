#!/usr/bin/env bash
# Re-runs everything the failed chain did not do, then the CIFAR reference run.
#
# The previous chain died because train_coverage.py was edited while it was
# running; fold0/s17 of s6 had already completed and is skipped here, c12b and
# s6b never started. Order is maritime arms first (minutes), then the double-damage
# control, then the remaining tiny probes, then the long CIFAR run.
set -u
cd /home/uestc/mount_2T/uestc/lnrio/MaritimeRobust || exit 1
PY=.venv/bin/python

until ! pgrep -f "scripts/linear_probe_grid.py" > /dev/null; do sleep 10; done
echo "=== rerun start $(date +%H:%M:%S)"

bash scripts/coverage_singles_run.sh > reports/s6_run.log 2>&1 || echo "S6 FAIL"
bash scripts/coverage_mixbase_run.sh > reports/c12b_run.log 2>&1 || echo "C12B FAIL"
bash scripts/coverage_singles_mixbase_run.sh > reports/s6b_run.log 2>&1 || echo "S6B FAIL"

echo "=== calibrate double damage $(date +%H:%M:%S)"
if [ ! -f reports/lattice_levels_40.json ]; then
  $PY scripts/lattice_calibrate.py --fold 0 --seed 17 --target 40 \
      --out reports/lattice_levels_40.json > reports/lattice_calibrate_40.log 2>&1 \
      || echo "CALIBRATE FAIL"
fi
bash scripts/coverage_singles_hard_run.sh > reports/s6h_run.log 2>&1 || echo "S6H FAIL"

echo "=== tiny probe $(date +%H:%M:%S)"
for f in 0 1 2; do
  for s in 17 29; do
    [ -f "reports/probe_convnext_tiny/fold${f}_s${s}.json" ] && continue
    $PY scripts/linear_probe_grid.py --fold "$f" --seed "$s" --model convnext_tiny \
        --out-dir reports/probe_convnext_tiny \
        > "reports/probe_convnext_tiny_f${f}_s${s}.log" 2>&1 \
        || echo "PROBE FAIL $f $s"
  done
done
echo "=== PROBE DONE $(date +%H:%M:%S)"

bash scripts/cifar_run.sh > reports/cifar_run.log 2>&1 || echo "CIFAR FAIL"
echo "=== RERUN DONE $(date +%H:%M:%S)"
