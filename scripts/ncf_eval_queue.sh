#!/usr/bin/env bash
# Evaluate the two arms the queue trained but never scored. The queue's eval step
# looked for a run directory with a doubled seed suffix ("_s17_s17") and failed in
# four seconds; the runs themselves are complete, so this only runs inference.
set -u
cd /home/uestc/mount_2T/uestc/lnrio/MaritimeRobust || exit 1
PY=.venv/bin/python

for TAG in cifar10_resnet18_cifar_ncf_s_ab_r2_s17 cifar10_resnet18_cifar_ncf_c_ab_r2_s29; do
  echo "=== $TAG eval $(date +%H:%M:%S)"
  $PY scripts/cifar_eval.py --run-dir "runs_cifar/$TAG" --which all \
      > "reports/${TAG}_eval.log" 2>&1 || { echo "EVAL FAIL $TAG"; exit 1; }
done
echo "=== NCF EVAL QUEUE DONE $(date +%H:%M:%S)"
