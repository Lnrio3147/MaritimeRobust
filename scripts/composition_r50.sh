#!/usr/bin/env bash
# Second backbone: train ResNet-50 clean baselines, then repeat the whole
# composition study on them. Results go to a separate directory so nothing
# overwrites the ConvNeXt study. Queued after the ConvNeXt second pass.
set -u
cd /home/uestc/mount_2T/uestc/lnrio/MaritimeRobust || exit 1
PY=.venv/bin/python
OUT=reports/composition_r50

while ! grep -q "REBUILD DONE" reports/composition/rebuild.log 2>/dev/null; do
  sleep 30
done
echo "=== r50 training $(date +%H:%M:%S)"

for f in 0 1 2; do
  for s in 17 29; do
    echo "--- train fold$f s$s $(date +%H:%M:%S)"
    $PY scripts/explore_train.py --fold "$f" --seed "$s" --method none \
        --model resnet50 --epochs 40 --suffix _e40 || exit 1
  done
done

echo "=== r50 composition $(date +%H:%M:%S)"
for f in 0 1 2; do
  for s in 17 29; do
    echo "--- compose fold$f s$s $(date +%H:%M:%S)"
    $PY scripts/composition_calibrate.py --fold "$f" --seed "$s" --model resnet50 --out-dir $OUT &&
    $PY scripts/composition_pairs.py --fold "$f" --seed "$s" --model resnet50 --out-dir $OUT &&
    $PY scripts/composition_interaction.py --fold "$f" --seed "$s" --target 20 --out-dir $OUT &&
    $PY scripts/composition_predictors.py --fold "$f" --seed "$s" --target 20 --out-dir $OUT
  done
done
echo "=== R50 DONE $(date +%H:%M:%S)"
