#!/usr/bin/env bash
# Regenerate every CIFAR-10-C summary from the stored per-image flags.
#
# The aggregate, accuracy, held-out and interval reports are all functions of
# runs_cifar/*/reports/cifar_grid/*/correct.npz, so they can be rebuilt at any
# time without a GPU. Adding an arm means adding it to ARMS and rerunning this.
set -u
cd /home/uestc/mount_2T/uestc/lnrio/MaritimeRobust || exit 1
PY=.venv/bin/python

ARMS=("clean_s17=reports/cifar_grid/cifar10_resnet18_cifar_clean_s17"
      "lattice_k15_s17=reports/cifar_grid/cifar10_resnet18_cifar_lattice_k15_s17"
      "lattice_k40_s17=reports/cifar_grid/cifar10_resnet18_cifar_lattice_k40_s17"
      "singles_s17=reports/cifar_grid/cifar10_resnet18_cifar_singles_s17"
      "lattice_k15_s29=reports/cifar_grid/cifar10_resnet18_cifar_lattice_k15_s29"
      "singles_s29=reports/cifar_grid/cifar10_resnet18_cifar_singles_s29"
      "augmix_s17=reports/cifar_grid/cifar10_resnet18_cifar_augmix_s17"
      "augmix_s29=reports/cifar_grid/cifar10_resnet18_cifar_augmix_s29"
      "otheraug_s17=reports/cifar_grid/cifar10_resnet18_cifar_otheraug_s17"
      "ncf_c_a_s17=reports/cifar_grid/cifar10_resnet18_cifar_ncf_c_a_s17"
      "ncf_s_a_s17=reports/cifar_grid/cifar10_resnet18_cifar_ncf_s_a_s17"
      "ncf_c_a_r2_s17=reports/cifar_grid/cifar10_resnet18_cifar_ncf_c_a_r2_s17"
      "ncf_c_b_r2_s17=reports/cifar_grid/cifar10_resnet18_cifar_ncf_c_b_r2_s17"
      "ncf_c_ab_r2_s17=reports/cifar_grid/cifar10_resnet18_cifar_ncf_c_ab_r2_s17")

$PY scripts/cifar_aggregate.py --runs "${ARMS[@]}" > reports/cifar_aggregate.log 2>&1 \
  || { echo "AGGREGATE FAIL"; exit 1; }
$PY scripts/cifar_accuracy.py --runs "${ARMS[@]}" > reports/cifar_accuracy.log 2>&1 \
  || { echo "ACCURACY FAIL"; exit 1; }
$PY scripts/cifar_heldout.py > reports/cifar_heldout.log 2>&1 || { echo "HELDOUT FAIL"; exit 1; }
$PY scripts/cifar_ci.py > reports/cifar_ci.log 2>&1 || { echo "CI FAIL"; exit 1; }
echo "wrote CIFAR_GRID.md, CIFAR_ACCURACY.md, CIFAR_HELDOUT.md, CIFAR_CI.md"
