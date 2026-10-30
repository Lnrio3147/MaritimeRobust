#!/usr/bin/env bash
# Rebuild the target-tagged interaction/predictor outputs for all six cells.
# Stale untagged files (written before the filename carried the target) are
# removed first so nothing can be read out under the wrong label.
set -u
cd /home/uestc/mount_2T/uestc/lnrio/MaritimeRobust || exit 1
PY=.venv/bin/python

rm -f reports/composition/interaction_fold*_s*.json \
      reports/composition/predictors_fold*_s*.json
rm -f reports/composition_r50/interaction_fold*_s*.json \
      reports/composition_r50/predictors_fold*_s*.json

echo "=== rebuild $(date +%H:%M:%S)"
for f in 0 1 2; do
  for s in 17 29; do
    for t in 20 10; do
      echo "--- fold$f s$s target$t $(date +%H:%M:%S)"
      $PY scripts/composition_interaction.py --fold "$f" --seed "$s" --target "$t" &&
      $PY scripts/composition_predictors.py --fold "$f" --seed "$s" --target "$t"
    done
  done
done
echo "=== REBUILD DONE $(date +%H:%M:%S)"
