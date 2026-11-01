#!/usr/bin/env bash
# The matched robustness baseline for the CIFAR-10-C arms.
#
# Every arm so far augments with the corruption families this study wrote. That
# answers "is coverage removable", not "is coverage competitive". AugMix
# (Hendrycks et al. 2020) is the standard data-augmentation answer to CIFAR-10-C
# and needs no corruption model, so it is the right fixed cost of comparison:
# same architecture, same 200 epochs, same optimizer, same 601-condition grid,
# with its Jensen-Shannon term in place.
set -u
cd /home/uestc/mount_2T/uestc/lnrio/MaritimeRobust || exit 1
PY=.venv/bin/python
TAG=cifar10_resnet18_cifar_augmix_s17

echo "=== $TAG train $(date +%H:%M:%S)"
if [ ! -f "runs_cifar/$TAG/complete.json" ]; then
  $PY scripts/cifar_train.py --arch resnet18 --seed 17 --epochs 200 --workers 8 --aug augmix \
      > "reports/${TAG}_train.log" 2>&1 || { echo "TRAIN FAIL $TAG"; exit 1; }
fi
echo "=== $TAG eval $(date +%H:%M:%S)"
if [ ! -f "reports/cifar_grid/$TAG/summary.json" ]; then
  $PY scripts/cifar_eval.py --run-dir "runs_cifar/$TAG" --which all \
      > "reports/${TAG}_eval.log" 2>&1 || { echo "EVAL FAIL $TAG"; exit 1; }
fi

ARMS=("clean_s17=reports/cifar_grid/cifar10_resnet18_cifar_clean_s17"
      "lattice_k15_s17=reports/cifar_grid/cifar10_resnet18_cifar_lattice_k15_s17"
      "lattice_k40_s17=reports/cifar_grid/cifar10_resnet18_cifar_lattice_k40_s17"
      "singles_s17=reports/cifar_grid/cifar10_resnet18_cifar_singles_s17"
      "lattice_k15_s29=reports/cifar_grid/cifar10_resnet18_cifar_lattice_k15_s29"
      "singles_s29=reports/cifar_grid/cifar10_resnet18_cifar_singles_s29"
      "augmix_s17=reports/cifar_grid/$TAG")
echo "=== aggregate $(date +%H:%M:%S)"
$PY scripts/cifar_aggregate.py --runs "${ARMS[@]}" > reports/cifar_aggregate_augmix.log 2>&1 || echo "AGGREGATE FAIL"
$PY scripts/cifar_accuracy.py --runs "${ARMS[@]}" > reports/cifar_accuracy.log 2>&1 || echo "ACCURACY FAIL"
echo "=== held-out pairs $(date +%H:%M:%S)"
$PY scripts/cifar_heldout.py > reports/cifar_heldout.log 2>&1 || echo "HELDOUT FAIL"
echo "=== CIFAR AUGMIX DONE $(date +%H:%M:%S)"
