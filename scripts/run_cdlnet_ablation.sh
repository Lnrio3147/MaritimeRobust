#!/usr/bin/env bash
# Sequential, idempotent ablation queue for the repaired CDLNet branches.
# Each run trains on the research split with the CDLNet data pipeline and
# evaluates the 43 degradation conditions. Completed runs are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG=reports/cdlnet_ablation.log
: > "$LOG"
for scope in research_all research_civilian; do
  manifest="data/manifests/fgsc23_${scope}.jsonl"
  for variant in resnet50 no_gate no_region no_branch; do
    run="runs_external/fgsc23_${scope}/ablation_${variant}_s17"
    echo "== ${scope} ${variant} ==" | tee -a "$LOG"
    if [ ! -f "$run/complete.json" ]; then
      .venv/bin/python external/cdlnet_repro/ablation.py train \
        --manifest "$manifest" --run-dir "$run" --variant "$variant" 2>&1 | tee -a "$LOG"
    fi
    .venv/bin/python external/cdlnet_repro/ablation.py eval \
      --run-dir "$run" 2>&1 | tee -a "$LOG"
  done
done
echo "ablation queue complete" | tee -a "$LOG"
