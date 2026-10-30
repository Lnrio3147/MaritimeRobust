#!/usr/bin/env python
"""Serial, restartable combined-degradation augmentation baseline.

The recipe is budget-matched with the first-round single-degradation arm:
equal clean probability; half of degraded draws are single conditions and half
are haze+glare / haze+cloud. Models, seeds and evaluation conditions match the
existing clean/single comparisons.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from maritime_robust.common import ROOT, digest, write_json

SEEDS = {"resnet50": [17], "convnext_tiny": [17, 29, 43], "swin_t": [17, 29, 43]}


def run(*args):
    subprocess.run([sys.executable, "-m", "maritime_robust.cli", *map(str, args)], cwd=ROOT, check=True)


def script(name):
    subprocess.run([sys.executable, str(ROOT / "scripts" / name)], cwd=ROOT, check=True)


def run_dir(scope, model, seed):
    return ROOT / f"runs/fgsc23_research_{scope}/{model}_combo_s{seed}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pilot", "all", "report"], default="all")
    args = parser.parse_args()
    status = {"stage": args.stage, "status": "running", "started": time.time(),
              "pid": os.getpid(), "recipe_sha256": digest(ROOT / "configs/combined_augmentation.json")}
    status_path = ROOT / "reports/combined_queue_status.json"
    write_json(status_path, status)
    plan = []
    for scope in ("all", "civilian"):
        for model, seeds in SEEDS.items():
            for seed in seeds:
                if args.stage == "pilot" and seed != 17:
                    continue
                plan.append((scope, model, seed))
    try:
        if args.stage in ("all", "pilot"):
            for scope, model, seed in plan:
                manifest = ROOT / f"data/manifests/fgsc23_research_{scope}.jsonl"
                run("train", "--manifest", manifest, "--model", model, "--augmentation", "combined", "--seed", seed)
                run("evaluate", "--run", run_dir(scope, model, seed))
                if seed == 17:
                    run("profile", "--run", run_dir(scope, model, seed))
        if args.stage in ("all", "report"):
            run("report")
            script("diagnostics_report.py")
            script("combined_baseline_report.py")
        status.update(status="complete", ended=time.time(), runs=len(plan))
    except BaseException as e:
        status.update(status="failed", ended=time.time(), error=repr(e))
        raise
    finally:
        write_json(status_path, status)


if __name__ == "__main__":
    main()
