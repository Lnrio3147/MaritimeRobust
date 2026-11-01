#!/usr/bin/env bash
# Second seed for the CIFAR pairing contrast (B2).
#
# B2 compares the paired arm against the singles arm at identical family
# exposure. At seed 17 the difference is -0.49 pp, below the -1 pp threshold,
# but a single seed carries no spread. This reruns the two arms that the
# contrast needs at seed 29, then regenerates the grid and held-out summaries
# with all six arms so the contrast can be reported per seed.
#
# k40 is not rerun: B1's dose-response question is already answered at seed 17.
set -u
cd /home/uestc/mount_2T/uestc/lnrio/MaritimeRobust || exit 1
PY=.venv/bin/python

run_arm () {          # $1 = tag suffix, rest = extra training flags
  local tag="cifar10_resnet18_cifar_$1_s29"; shift
  echo "=== $tag train $(date +%H:%M:%S)"
  if [ ! -f "runs_cifar/$tag/complete.json" ]; then
    $PY scripts/cifar_train.py --arch resnet18 --seed 29 --epochs 200 --workers 8 "$@" \
        > "reports/${tag}_train.log" 2>&1 || { echo "TRAIN FAIL $tag"; return 1; }
  fi
  echo "=== $tag eval $(date +%H:%M:%S)"
  if [ ! -f "reports/cifar_grid/$tag/summary.json" ]; then
    $PY scripts/cifar_eval.py --run-dir "runs_cifar/$tag" --which all \
        > "reports/${tag}_eval.log" 2>&1 || { echo "EVAL FAIL $tag"; return 1; }
  fi
  echo "=== $tag done $(date +%H:%M:%S)"
}

run_arm lattice_k15 --aug lattice --pool-k 15 || echo "ARM FAIL k15 s29"
run_arm singles --aug singles || echo "ARM FAIL singles s29"

echo "=== aggregate $(date +%H:%M:%S)"
$PY scripts/cifar_aggregate.py \
    --runs "clean_s17=reports/cifar_grid/cifar10_resnet18_cifar_clean_s17" \
           "lattice_k15_s17=reports/cifar_grid/cifar10_resnet18_cifar_lattice_k15_s17" \
           "lattice_k40_s17=reports/cifar_grid/cifar10_resnet18_cifar_lattice_k40_s17" \
           "singles_s17=reports/cifar_grid/cifar10_resnet18_cifar_singles_s17" \
           "lattice_k15_s29=reports/cifar_grid/cifar10_resnet18_cifar_lattice_k15_s29" \
           "singles_s29=reports/cifar_grid/cifar10_resnet18_cifar_singles_s29" \
    > reports/cifar_aggregate_seed29.log 2>&1 || echo "AGGREGATE FAIL"
echo "=== held-out pairs $(date +%H:%M:%S)"
$PY scripts/cifar_heldout.py > reports/cifar_heldout.log 2>&1 || echo "HELDOUT FAIL"
echo "=== CIFAR SEED29 DONE $(date +%H:%M:%S)"
