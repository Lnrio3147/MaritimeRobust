#!/usr/bin/env bash
# Stage B must not share the GPU with stage A. Wait for the stage-A chain to
# finish (or fail), then run the coverage arms.
set -u
cd /home/uestc/mount_2T/uestc/lnrio/MaritimeRobust || exit 1
until grep -qE "CIFAR A DONE|CIFAR (TRAIN|EVAL|AGGREGATE|VALIDATE) FAIL" reports/cifar_run.log; do
  sleep 30
done
sleep 20
tail -3 reports/cifar_run.log
bash scripts/cifar_stage_b.sh > reports/cifar_stage_b.log 2>&1
echo "=== STAGE B FINISHED $(date +%H:%M:%S)" >> reports/cifar_stage_b.log
