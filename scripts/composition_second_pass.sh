#!/usr/bin/env bash
# Second pass over the composition study, run after replication.sh finishes:
#   * interaction decomposition at the 10 pp matched marginals (cleaner: far
#     from the accuracy floor)
#   * mechanism predictors at 20 pp for every fold/seed
set -u
cd /home/uestc/mount_2T/uestc/lnrio/MaritimeRobust || exit 1
PY=.venv/bin/python

while ! grep -q "=== DONE" reports/composition/replication.log 2>/dev/null; do
  sleep 20
done
echo "=== second pass $(date +%H:%M:%S)"

for fs in "0 17" "0 29" "1 17" "1 29" "2 17" "2 29"; do
  set -- $fs
  echo "--- fold$1 s$2 $(date +%H:%M:%S)"
  $PY scripts/composition_interaction.py --fold "$1" --seed "$2" --target 10
  $PY scripts/composition_predictors.py --fold "$1" --seed "$2" --target 20
done
echo "=== SECOND PASS DONE $(date +%H:%M:%S)"
