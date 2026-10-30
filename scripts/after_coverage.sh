#!/usr/bin/env bash
# Everything queued behind the running coverage_run.sh (c3/c6/c12), in order.
#
# Each sub-script takes the exclusive GPU lock through Budget, so this is a
# sequential chain by construction, not a parallel fan-out.
set -u
cd /home/uestc/mount_2T/uestc/lnrio/MaritimeRobust || exit 1
PY=.venv/bin/python

until ! pgrep -f "scripts/coverage_run.sh" > /dev/null; do sleep 10; done
echo "=== chain start $(date +%H:%M:%S)"

bash scripts/coverage_singles_run.sh > reports/s6_run.log 2>&1 || echo "S6 FAIL"
bash scripts/coverage_mixbase_run.sh > reports/c12b_run.log 2>&1 || echo "C12B FAIL"
bash scripts/coverage_singles_mixbase_run.sh > reports/s6b_run.log 2>&1 || echo "S6B FAIL"

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
echo "=== CHAIN DONE $(date +%H:%M:%S)"
