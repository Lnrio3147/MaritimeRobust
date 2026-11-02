#!/usr/bin/env bash
# Chain the remaining NCF arms on the single GPU, after the running tier-B job.
#
#   F2 ablation   ncf_s_ab_r2_s17   one field per sample (does composition matter?)
#   stability     ncf_c_ab_r2_s29   second seed of the headline arm
#
# Both reuse artifacts/fields_ab.npz, which already exists, so no bank is rebuilt.
set -u
cd /home/uestc/mount_2T/uestc/lnrio/MaritimeRobust || exit 1

CUR="cifar_eval.py --run-dir runs_cifar/cifar10_resnet18_cifar_ncf_c_bb_r2_s17"
echo "=== queue waiting for tier-B eval $(date +%H:%M:%S)"
while pgrep -f "$CUR" > /dev/null; do sleep 30; done
echo "=== queue started $(date +%H:%M:%S)"

bash scripts/cifar_ncf.sh ab s 2 artifacts/field_estimator_blur2.pt 17
echo "=== ablation done $(date +%H:%M:%S)"

bash scripts/cifar_ncf.sh ab c 2 artifacts/field_estimator_blur2.pt 29
echo "=== NCF QUEUE DONE $(date +%H:%M:%S)"
