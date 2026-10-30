#!/usr/bin/env python
"""Aggregate the composition study across folds, seeds and matched-drop targets.

Reads every ``interaction_fold*_s*.json`` and ``predictors_fold*_s*.json`` and
reports, per pair, the interaction across runs and how often its sign is stable;
per run, the family main effects and the additive fit; and per predictor, the
rank correlation across runs.

    python scripts/composition_aggregate.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from maritime_robust.common import ROOT as PROJECT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="reports/composition")
    ap.add_argument("--field", default="delta_pp")
    ap.add_argument("--target", type=float, default=20.0)
    args = ap.parse_args()

    base = PROJECT / args.out_dir
    inter_paths = sorted(base.glob(f"interaction_fold*_s*_t{args.target:g}.json"))
    if not inter_paths:
        print("no interaction files")
        return

    per_pair = defaultdict(list)
    per_run = []
    for path in inter_paths:
        d = json.loads(path.read_text())
        tag = f"fold{d['fold']} s{d['seed']} @{d['target_pp']:g}"
        per_run.append({"tag": tag, "path": path.name,
                        "additive_r2": d["additive_r2"]["delta"],
                        "main_effects": d["main_effects_delta"],
                        "spearman_cos_delta": d["spearman"]["cos_delta"][0],
                        "mean_delta": float(np.mean([r["delta_pp"] for r in d["pairs"]]))})
        for r in d["pairs"]:
            per_pair[r["pair"]].append(r[args.field])

    print(f"{'pair':22s} {'n':>2s} {'mean':>7s} {'sd':>6s} {'pos':>5s}  values")
    for pair in sorted(per_pair, key=lambda p: -np.mean(per_pair[p])):
        v = np.array(per_pair[pair])
        print(f"{pair:22s} {len(v):2d} {v.mean():+7.1f} {v.std(ddof=1) if len(v) > 1 else 0:6.1f} "
              f"{int((v > 0).sum())}/{len(v):<3d} "
              + " ".join(f"{x:+.1f}" for x in v))

    print()
    print(f"{'run':22s} {'mean_delta':>10s} {'add_R2':>7s} {'rho(cos)':>9s}  main effects (noise/rain/pixelate)")
    for r in per_run:
        m = r["main_effects"]
        print(f"{r['tag']:22s} {r['mean_delta']:+10.1f} {r['additive_r2']:7.3f} "
              f"{r['spearman_cos_delta']:+9.3f}  "
              f"{m.get('noise', float('nan')):+.1f}/{m.get('rain', float('nan')):+.1f}/"
              f"{m.get('pixelate', float('nan')):+.1f}")

    # Family main effects averaged over runs.
    fams = sorted({f for r in per_run for f in r["main_effects"]})
    print()
    print("family main effect on delta, mean over runs:")
    for f in sorted(fams, key=lambda f: -np.mean([r["main_effects"][f] for r in per_run])):
        v = np.array([r["main_effects"][f] for r in per_run])
        print(f"  {f:9s} {v.mean():+6.1f} pp  (range {v.min():+.1f} .. {v.max():+.1f})")

    pred_paths = sorted(base.glob(f"predictors_fold*_s*_t{args.target:g}.json"))
    if pred_paths:
        rows = []
        for path in pred_paths:
            d = json.loads(path.read_text())
            rows.append((f"fold{d['fold']} s{d['seed']}", d["spearman"]))
        print()
        print("predictor rank correlations (Spearman vs delta):")
        keys = sorted({k for _, s in rows for k in s})
        print(f"{'run':16s} " + " ".join(f"{k:>22s}" for k in keys))
        for tag, s in rows:
            print(f"{tag:16s} " + " ".join(f"{s[k][0]:+9.3f} (p={s[k][1]:.2f})"
                                          for k in keys))
        print("mean: " + " ".join(
            f"{np.mean([s[k][0] for _, s in rows]):+9.3f}" for k in keys))


if __name__ == "__main__":
    main()
