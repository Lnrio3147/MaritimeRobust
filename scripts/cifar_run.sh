#!/usr/bin/env bash
# Stage A of the CIFAR-10-C replication: one clean ResNet-18, the full single+pair
# grid, then the union decomposition.
#
# Why this exists: every result so far is on one maritime dataset with
# hand-written corruption families. If the excess structure reproduces on the
# published 15-corruption grid with a standard architecture and recipe, the
# claim stops being a property of this dataset. Nothing is trained here except
# the reference model, so the only thing this can fail to reproduce is the
# *measurement*, which is exactly what needs checking.
set -u
cd /home/uestc/mount_2T/uestc/lnrio/MaritimeRobust || exit 1
PY=.venv/bin/python
PYTHONUNBUFFERED=1

echo "=== cifar train clean $(date +%H:%M:%S)"
$PY scripts/cifar_train.py --arch resnet18 --seed 17 --epochs 100 --workers 8 \
    > reports/cifar_train_s17.log 2>&1 || { echo "CIFAR TRAIN FAIL"; exit 1; }

echo "=== cifar eval clean $(date +%H:%M:%S)"
$PY scripts/cifar_eval.py --run-dir runs_cifar/cifar10_resnet18_clean_s17 --which all \
    > reports/cifar_eval_s17.log 2>&1 || { echo "CIFAR EVAL FAIL"; exit 1; }

echo "=== cifar aggregate $(date +%H:%M:%S)"
$PY scripts/cifar_aggregate.py \
    --runs "resnet18_s17=reports/cifar_grid/cifar10_resnet18_clean_s17" \
    > reports/cifar_aggregate.log 2>&1 || { echo "CIFAR AGGREGATE FAIL"; exit 1; }
echo "=== CIFAR A DONE $(date +%H:%M:%S)"
