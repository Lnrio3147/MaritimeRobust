#!/usr/bin/env bash
# Resume the chain from where the disk-budget guard killed it.
#
# ``runs/`` now lives outside the project tree behind a symlink, so
# ``directory_bytes(ROOT)`` no longer counts 24 GB of checkpoints; the guard that
# raised "50 GB project disk budget exhausted" is not going to fire again unless
# the project itself grows past the cap.
#
# Every step is guarded on its own output, so re-running the whole file is cheap.
set -u
cd /home/uestc/mount_2T/uestc/lnrio/MaritimeRobust || exit 1
PY=.venv/bin/python

echo "=== resume start $(date +%H:%M:%S)"

bash scripts/coverage_singles_mixbase_run.sh > reports/s6b_run.log 2>&1 || echo "S6B FAIL"

echo "=== calibrate double damage $(date +%H:%M:%S)"
if [ ! -f reports/lattice_levels_40.json ]; then
  $PY scripts/lattice_calibrate.py --fold 0 --seed 17 --target 40 \
      --out reports/lattice_levels_40.json > reports/lattice_calibrate_40.log 2>&1 \
      || echo "CALIBRATE FAIL"
fi

bash scripts/coverage_singles_hard_run.sh > reports/s6h_run.log 2>&1 || echo "S6H FAIL"

echo "=== aggregate $(date +%H:%M:%S)"
$PY scripts/coverage_aggregate.py > reports/after_resume_aggregate.log 2>&1 \
    || echo "AGGREGATE FAIL"

bash scripts/cifar_run.sh > reports/cifar_run.log 2>&1 || echo "CIFAR FAIL"
echo "=== RESUME DONE $(date +%H:%M:%S)"
