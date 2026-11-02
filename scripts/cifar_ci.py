"""Uncertainty on the compound-accuracy gain of each CIFAR-10-C arm.

``cifar_accuracy.py`` reports the absolute accuracy of every arm on the pair
conditions; this script puts an interval around the gain over the clean
reference and around the contrast between arms.

The unit of resampling is the corruption pair, not the image and not the
condition. The 525 pair conditions are five severities of 105 pairs, so the
five severities of one pair are not independent draws and resampling
conditions would overstate precision; resampling pairs keeps that clustering.
Images are shared across arms, so every arm contrast is computed per pair
first and then resampled, which makes the interval for a difference paired.

    python scripts/cifar_ci.py                     # every arm in the grid json
    python scripts/cifar_ci.py --ref clean_s17 --minutes k15-vs-augmix

Writes reports/CIFAR_CI.md and reports/cifar_ci.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def acc_map(payload, arm):
    return {(r["a"], r["b"], r["severity"]): 1.0 - r["error_ab"]
            for r in payload[arm]["rows"]}


def pair_gains(payload, arm, ref):
    """Mean gain in pp over ``ref`` for each of the 105 corruption pairs."""
    a, b = acc_map(payload, arm), acc_map(payload, ref)
    keys = [k for k in a if k in b]
    groups: dict[frozenset, list[float]] = {}
    for k in keys:
        groups.setdefault(frozenset((k[0], k[1])), []).append((a[k] - b[k]) * 100)
    names = sorted(sorted(g) for g in groups)
    return names, np.array([np.mean(groups[frozenset(n)]) for n in names])


def boot(values, n_boot, rng):
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="reports/cifar_grid.json")
    ap.add_argument("--ref", default="clean_s17")
    ap.add_argument("--contrast", default="lattice_k15_s17:augmix_s17",
                    help="arm_a:arm_b, reported as a-b")
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/CIFAR_CI.md")
    ap.add_argument("--out-json", default="reports/cifar_ci.json")
    args = ap.parse_args()

    payload = json.loads((PROJECT / args.json).read_text())
    rng = np.random.default_rng(args.seed)
    lines = ["# CIFAR-10-C: compound accuracy gain, with pair-level intervals", "",
             f"Gain over `{args.ref}` on the 525 pair conditions, in percentage points.",
             f"Intervals are percentile bootstrap over the 105 corruption pairs "
             f"({args.bootstrap} resamples), which keeps the five severities of a pair "
             "together.", "",
             "| arm | gain (pp) | 95% CI | worst pair | best pair |",
             "|---|---:|---|---:|---:|"]
    summary, gains = {}, {}
    for arm in payload:
        if arm == args.ref:
            continue
        names, values = pair_gains(payload, arm, args.ref)
        lo, hi = boot(values, args.bootstrap, rng)
        gains[arm] = values
        summary[arm] = {"names": ["+".join(n) for n in names],
                        "per_pair_pp": values.round(3).tolist(),
                        "gain_pp": float(values.mean()), "ci95": [lo, hi],
                        "n_pairs": len(values)}
        lines.append(f"| {arm} | {values.mean():+.2f} | [{lo:+.2f}, {hi:+.2f}] | "
                     f"{values.min():+.1f} | {values.max():+.1f} |")

    a_arm, b_arm = args.contrast.split(":")
    if a_arm in gains and b_arm in gains:
        diff = gains[a_arm] - gains[b_arm]
        lo, hi = boot(diff, args.bootstrap, rng)
        wins = int((diff > 0).sum())
        summary["_contrast"] = {"a": a_arm, "b": b_arm, "mean_pp": float(diff.mean()),
                                "ci95": [lo, hi], "pairs_a_wins": wins,
                                "n_pairs": int(len(diff))}
        lines += ["", f"## {a_arm} minus {b_arm}", "",
                  f"Mean {diff.mean():+.2f} pp, 95% CI [{lo:+.2f}, {hi:+.2f}]; "
                  f"{a_arm} is higher on {wins} of {len(diff)} pairs."]

    # Two-seed reproduction, where both seeds exist for an arm family.
    seeds: dict[str, dict[str, float]] = {}
    for arm in payload:
        m = re.match(r"(.+)_s(\d+)$", arm)
        if m:
            values = gains.get(arm)
            seeds.setdefault(m.group(1), {})[m.group(2)] = (
                float(values.mean()) if values is not None else None)
    lines += ["", "## Across seeds", ""]
    for family, by_seed in seeds.items():
        vals = {s: v for s, v in by_seed.items() if v is not None}
        if len(vals) > 1:
            spread = max(vals.values()) - min(vals.values())
            summary.setdefault("_seeds", {})[family] = vals
            joined = ", ".join(f"s{s}: {v:+.2f}" for s, v in sorted(vals.items()))
            lines.append(f"- {family}: {joined} pp; spread {spread:.2f} pp")

    (PROJECT / args.out).write_text("\n".join(lines) + "\n")
    (PROJECT / args.out_json).write_text(json.dumps(summary, indent=1) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
