#!/usr/bin/env python
"""Apply the pre-registered replication criteria mechanically.

Reads ``reports/PREREG_COMPOSITION_REPLICATION.md``'s C1-C5 and evaluates them
against every available run. Nothing here is chosen after seeing the numbers:
the sign of every criterion, its threshold and the pass counts are fixed in the
pre-registration.

    python scripts/composition_gate.py [--out-dir reports/composition] [--target 20]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from maritime_robust.common import ROOT as PROJECT  # noqa: E402

LOWPASS = {"defocus", "motion", "pixelate"}


def load(out_dir, target):
    runs = {}
    for path in sorted((PROJECT / out_dir).glob(f"interaction_fold*_s*_t{target:g}.json")):
        d = json.loads(path.read_text())
        if abs(d["target_pp"] - target) > 1e-6:
            continue
        runs[(d["fold"], d["seed"])] = d
    preds = {}
    for path in sorted((PROJECT / out_dir).glob(f"predictors_fold*_s*_t{target:g}.json")):
        d = json.loads(path.read_text())
        if abs(d["target_pp"] - target) > 1e-6:
            continue
        preds[(d["fold"], d["seed"])] = d
    return runs, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="reports/composition")
    ap.add_argument("--target", type=float, default=20.0)
    args = ap.parse_args()

    runs, preds = load(args.out_dir, args.target)
    if len(runs) < 2:
        print(f"need at least 2 runs at {args.target:g} pp, have {len(runs)}")
        return
    discovery_key = (0, 17) if (0, 17) in runs else sorted(runs)[0]
    discovery = runs[discovery_key]
    disc = {r["pair"]: r["delta_pp"] for r in discovery["pairs"]}
    others = [k for k in sorted(runs) if k != discovery_key]

    print(f"discovery cell: fold{discovery_key[0]} s{discovery_key[1]}, "
          f"{len(others)} replication cells, target {args.target:g} pp")
    print()

    # C1 -- sign agreement on pairs that were large in the discovery cell.
    anchor = {p: v for p, v in disc.items() if abs(v) >= 5.0}
    c1_cells = []
    for k in others:
        m = {r["pair"]: r["delta_pp"] for r in runs[k]["pairs"]}
        agree = [p for p in anchor if p in m and np.sign(m[p]) == np.sign(anchor[p])]
        c1_cells.append((k, len(agree), len([p for p in anchor if p in m])))
    c1_pass = sum(1 for _, a, n in c1_cells if n and a / n >= 0.8)
    c1 = c1_pass >= 4

    # C2 -- low-pass x low-pass mean delta < 0.
    def mean_for(d, keep):
        vals = [r["delta_pp"] for r in d["pairs"]
                if keep(r["a"], r["b"])]
        return float(np.mean(vals)) if vals else float("nan")

    low = lambda a, b: a in LOWPASS and b in LOWPASS  # noqa: E731
    noisy = lambda a, b: "noise" in (a, b)            # noqa: E731
    c2_vals = {k: mean_for(d, low) for k, d in runs.items()}
    c3_vals = {k: mean_for(d, noisy) for k, d in runs.items()}
    c2 = sum(1 for v in c2_vals.values() if v < 0) >= 5
    c3 = sum(1 for v in c3_vals.values() if v > 0) >= 5

    # C4 -- additive main-effect fit.
    c4_vals = {k: d["additive_r2"]["delta"] for k, d in runs.items()}
    c4 = sum(1 for v in c4_vals.values() if v >= 0.40) >= 5

    # C5 -- no single-corruption predictor is strong and stable.
    keys = ("alignment_delta", "hf_product_delta", "overlap_delta",
            "class_error_corr_delta")
    strong = {}
    for k, d in preds.items():
        for name in keys:
            rho = d["spearman"].get(name, [0.0])[0]
            if abs(rho) >= 0.60:
                strong.setdefault(name, []).append(k)
    # strong-stable = present in a majority of predictor runs
    stable = {n: len(v) >= max(2, len(preds) // 2 + 1) for n, v in strong.items()}
    c5 = sum(stable.values()) <= 1

    def fmt(k):
        return f"fold{k[0]} s{k[1]}"

    print(f"{'pair':22s} {'disc':>7s} " + " ".join(f"{fmt(k):>11s}" for k in others))
    for p in sorted(disc, key=lambda p: -abs(disc[p])):
        line = f"{p:22s} {disc[p]:+7.1f} "
        for k in others:
            m = {r["pair"]: r["delta_pp"] for r in runs[k]["pairs"]}
            line += f"{m.get(p, float('nan')):+11.1f} "
        print(line)

    print()
    print("C1 sign agreement on |delta|>=5 pairs:")
    for k, a, n in c1_cells:
        print(f"   {fmt(k):11s} {a}/{n}")
    print(f"C1 passed in {c1_pass}/{len(others)} cells -> {'PASS' if c1 else 'FAIL'}")
    print()
    print("C2 lowpass+lowpass mean delta (<0):")
    for k, v in c2_vals.items():
        print(f"   {fmt(k):11s} {v:+6.1f}")
    print(f"C2 negative in {sum(1 for v in c2_vals.values() if v < 0)}/{len(c2_vals)} "
          f"-> {'PASS' if c2 else 'FAIL'}")
    print()
    print("C3 pairs containing noise mean delta (>0):")
    for k, v in c3_vals.items():
        print(f"   {fmt(k):11s} {v:+6.1f}")
    print(f"C3 positive in {sum(1 for v in c3_vals.values() if v > 0)}/{len(c3_vals)} "
          f"-> {'PASS' if c3 else 'FAIL'}")
    print()
    print("C4 additive R^2 on delta (>=0.40):")
    for k, v in c4_vals.items():
        print(f"   {fmt(k):11s} {v:5.3f}")
    print(f"C4 >=0.40 in {sum(1 for v in c4_vals.values() if v >= 0.40)}/{len(c4_vals)} "
          f"-> {'PASS' if c4 else 'FAIL'}")
    print()
    print("C5 no stable strong predictor (|rho|>=0.60):")
    for name in keys:
        vals = [preds[k]["spearman"].get(name, [float('nan')])[0] for k in sorted(preds)]
        if not vals:
            continue
        print(f"   {name:20s} mean {np.nanmean(vals):+.3f}  "
              f"strong in {len(strong.get(name, []))}/{len(preds)}")
    print(f"C5 -> {'PASS' if c5 else 'FAIL'}")

    verdict = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5}
    print()
    print("verdict:", verdict)
    all_pass = all(verdict.values())
    print("ALL PASS -> report the composition structure as replicated"
          if all_pass else
          "NOT ALL PASS -> report only as single-cell exploratory observation")
    path = PROJECT / args.out_dir / f"gate_target{args.target:g}.json"
    path.write_text(json.dumps({
        "target_pp": args.target,
        "discovery": fmt(discovery_key),
        "n_cells": len(runs),
        "criteria": verdict,
        "c1_cells": {fmt(k): [a, n] for k, a, n in c1_cells},
        "c2_lowpass_mean": {fmt(k): v for k, v in c2_vals.items()},
        "c3_noise_mean": {fmt(k): v for k, v in c3_vals.items()},
        "c4_additive_r2": {fmt(k): v for k, v in c4_vals.items()},
        "c5_strong_predictor_counts": {n: len(v) for n, v in strong.items()},
        "all_pass": all_pass}, indent=2))
    print("wrote", path.relative_to(PROJECT))


if __name__ == "__main__":
    main()
