#!/usr/bin/env bash
# Stage B of the CIFAR-10-C replication: does coverage remove the excess here too?
#
# Three arms, all ResNet-18 / 100 epochs / seed 17, differing only in the
# augmentation support:
#   lattice k=15   a pool of 15 pairs drawn from the 15 canonical corruptions,
#                  with 5 pairs held out of every pool (cifar_c.HELD_OUT)
#   lattice k=40   a wider pool, to test the dose-response claim B1
#   singles        every corruption used alone at the same rate, so that family
#                  exposure is identical and only co-occurrence is removed (B2)
#
# Each arm is evaluated on all 601 conditions and decomposed by cifar_aggregate.py;
# the pre-registration is reports/PREREG_CIFAR.md, B1--B3.
set -u
cd /home/uestc/mount_2T/uestc/lnrio/MaritimeRobust || exit 1
PY=.venv/bin/python

run_arm () {          # $1 = tag suffix, rest = extra training flags
  local tag="cifar10_resnet18_cifar_$1_s17"; shift
  echo "=== $tag train $(date +%H:%M:%S)"
  if [ ! -f "runs_cifar/$tag/complete.json" ]; then
    $PY scripts/cifar_train.py --arch resnet18 --seed 17 --epochs 200 --workers 8 "$@" \
        > "reports/${tag}_train.log" 2>&1 || { echo "TRAIN FAIL $tag"; return 1; }
  fi
  echo "=== $tag eval $(date +%H:%M:%S)"
  if [ ! -f "reports/cifar_grid/$tag/summary.json" ]; then
    $PY scripts/cifar_eval.py --run-dir "runs_cifar/$tag" --which all \
        > "reports/${tag}_eval.log" 2>&1 || { echo "EVAL FAIL $tag"; return 1; }
  fi
}

run_arm lattice_k15 --aug lattice --pool-k 15 || echo "ARM FAIL k15"
run_arm lattice_k40 --aug lattice --pool-k 40 || echo "ARM FAIL k40"
run_arm singles --aug singles || echo "ARM FAIL singles"

echo "=== aggregate $(date +%H:%M:%S)"
$PY scripts/cifar_aggregate.py \
    --runs "clean_s17=reports/cifar_grid/cifar10_resnet18_cifar_clean_s17" \
           "lattice_k15=reports/cifar_grid/cifar10_resnet18_cifar_lattice_k15_s17" \
           "lattice_k40=reports/cifar_grid/cifar10_resnet18_cifar_lattice_k40_s17" \
           "singles=reports/cifar_grid/cifar10_resnet18_cifar_singles_s17" \
    > reports/cifar_aggregate_b.log 2>&1 || echo "AGGREGATE FAIL"
echo "=== held-out pairs $(date +%H:%M:%S)"
$PY scripts/cifar_heldout.py > reports/cifar_heldout.log 2>&1 || echo "HELDOUT FAIL"
echo "=== CIFAR B DONE $(date +%H:%M:%S)"
