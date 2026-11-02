#!/usr/bin/env bash
# The disjoint-family control for the coverage claim (pre-registration O1-O3).
#
# The coverage arms train on the corruption families they are tested on, so their
# advantage over AugMix could be strength rather than family identity. This arm
# keeps the rate, the budget and the number of augmentations per image, and makes
# the families disjoint from CIFAR-10-C's fifteen. If strength explains the gain
# this arm closes the gap; if family identity explains it, this arm stays near
# AugMix.
set -u
cd /home/uestc/mount_2T/uestc/lnrio/MaritimeRobust || exit 1
PY=.venv/bin/python
TAG=cifar10_resnet18_cifar_otheraug_s17

echo "=== $TAG train $(date +%H:%M:%S)"
if [ ! -f "runs_cifar/$TAG/complete.json" ]; then
  $PY scripts/cifar_train.py --arch resnet18 --seed 17 --epochs 200 --workers 8 --aug otheraug \
      > "reports/${TAG}_train.log" 2>&1 || { echo "TRAIN FAIL $TAG"; exit 1; }
fi
echo "=== $TAG eval $(date +%H:%M:%S)"
if [ ! -f "reports/cifar_grid/$TAG/summary.json" ]; then
  $PY scripts/cifar_eval.py --run-dir "runs_cifar/$TAG" --which all \
      > "reports/${TAG}_eval.log" 2>&1 || { echo "EVAL FAIL $TAG"; exit 1; }
fi
echo "=== reports $(date +%H:%M:%S)"
bash scripts/cifar_report.sh || echo "REPORT FAIL"
echo "=== CIFAR OTHERAUG DONE $(date +%H:%M:%S)"
