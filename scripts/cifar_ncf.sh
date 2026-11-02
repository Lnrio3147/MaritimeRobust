#!/usr/bin/env bash
# NCF arm: train on fields *estimated* from unlabeled deployment images.
#
# Two arms, and the difference between them is the module's claim:
#   ncf_c  two sampled fields composed  (joint coverage)
#   ncf_s  one sampled field            (F2 ablation: is composition doing anything)
#
# Usage: bash scripts/cifar_ncf.sh [tier] [modes] [repeats] [ckpt] [seed]
#   modes: c=compound s=single
#   tier a   bank built from all fifteen families (representation check)
#   tier b   bank built with fog/frost/snow excluded (the unknown-family test)
#   tier bb  same exclusion, built from an explicitly named estimator checkpoint
#            (tier bb exists so a new estimator does not overwrite tier b's bank)
set -u
cd /home/uestc/mount_2T/uestc/lnrio/MaritimeRobust || exit 1
PY=.venv/bin/python
TIER="${1:-a}"
REPEATS="${3:-1}"
CKPT="${4:-artifacts/field_estimator.pt}"
SEED="${5:-17}"
BANK="artifacts/fields_${TIER}.npz"

echo "=== bank tier $TIER $(date +%H:%M:%S)"
if [ ! -f "$BANK" ]; then
  EXCL=""
  case "$TIER" in b|bb) EXCL="--exclude fog,frost,snow" ;; esac
  $PY scripts/cifar_field_bank.py $EXCL --ckpt "$CKPT" --out "$BANK" || { echo "BANK FAIL"; exit 1; }
fi

for SHORT in ${2:-c s}; do
  case "$SHORT" in
    c) MODE=compound ;;
    s) MODE=single ;;
    *) echo "unknown mode $SHORT"; exit 1 ;;
  esac
  FLAG=""
  [ "$MODE" = compound ] && FLAG="--field-compound"
  SFX=""
  [ "$REPEATS" != "1" ] && SFX="_r${REPEATS}"
  TAG="cifar10_resnet18_cifar_ncf_${MODE:0:1}_${TIER}${SFX}_s${SEED}"
  echo "=== $TAG train $(date +%H:%M:%S)"
  if [ ! -f "runs_cifar/$TAG/complete.json" ]; then
    $PY scripts/cifar_train.py --arch resnet18 --seed "$SEED" --epochs 200 --workers 8 \
        --aug ncf --field-bank "$BANK" $FLAG --field-repeats "$REPEATS" \
        --suffix "_ncf_${MODE:0:1}_${TIER}${SFX}" \
        > "reports/${TAG}_train.log" 2>&1 || { echo "TRAIN FAIL $TAG"; exit 1; }
  fi
  echo "=== $TAG eval $(date +%H:%M:%S)"
  if [ ! -f "reports/cifar_grid/$TAG/summary.json" ]; then
    $PY scripts/cifar_eval.py --run-dir "runs_cifar/$TAG" --which all \
        > "reports/${TAG}_eval.log" 2>&1 || { echo "EVAL FAIL $TAG"; exit 1; }
  fi
done
echo "=== CIFAR NCF $TIER DONE $(date +%H:%M:%S)"
