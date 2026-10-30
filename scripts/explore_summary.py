#!/usr/bin/env python
"""Rank S1 screening runs against the same-fold, same-seed baseline.

Reads reports/s1/<fold>__<run>.json written by scripts/explore_eval.py and the
fold manifests, differences every candidate against `convnext_tiny_combo_s<seed>`
on the same fold, and reports the mean delta over the 8 combo conditions with a
source-group bootstrap CI.

    python scripts/explore_summary.py [--baseline-prefix convnext_tiny_combo]
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
from maritime_robust.common import ROOT as PROJECT, read_jsonl  # noqa: E402


def boot(delta, groups, reps=2000, seed=0):
    uniq = {g: np.where(groups == g)[0] for g in np.unique(groups)}
    keys = list(uniq)
    rng = np.random.default_rng(seed)
    stats = np.empty(reps)
    for i in range(reps):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([uniq[keys[j]] for j in pick])
        stats[i] = 100 * delta[idx].mean()
    return 100 * float(delta.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="reports/s1")
    ap.add_argument("--out", default="reports/s1_summary.json")
    ap.add_argument("--suffix", default="", help="run-name suffix shared by candidate and baseline")
    args = ap.parse_args()

    runs = {}
    for path in sorted(glob.glob(str(PROJECT / args.dir / "*.json"))):
        if Path(path).name.startswith("_"):
            continue
        blob = json.loads(Path(path).read_text())
        fold = blob["run"].split("/")[1]
        name = Path(blob["run"]).name
        runs[(fold, name)] = blob

    groups_for, labels_for = {}, {}
    for fold in {f for f, _ in runs}:
        manifest = PROJECT / f"data/manifests/folds/{fold}.jsonl"
        rows = [r for r in read_jsonl(manifest) if r["split"] == "val"]
        groups_for[fold] = np.array([r["source_group_id"] for r in rows])
        labels_for[fold] = np.array([r["label"] for r in rows])

    out = {"runs": len(runs), "entries": []}
    for (fold, name), blob in sorted(runs.items()):
        seed = blob["seed"]
        # same-fold same-seed reference arm; fall back to the engine-produced
        # combo run when only that exists
        sfx = args.suffix
        base = (runs.get((fold, f"convnext_tiny_none{sfx}_s{seed}"))
                or runs.get((fold, f"convnext_tiny_combo_s{seed}")))
        if base is None or name in (f"convnext_tiny_none{sfx}_s{seed}",
                                    f"convnext_tiny_combo_s{seed}"):
            continue
        combo = blob.get("combo_conditions") or [c for c in blob["conditions"] if c != "clean"]
        # per-sample correctness delta, averaged over the 8 combo conditions
        labels = labels_for[fold]
        cand = np.mean([np.array(blob["conditions"][c]["predictions"]) == labels for c in combo], axis=0)
        base_ok = np.mean([np.array(base["conditions"][c]["predictions"]) == labels for c in combo], axis=0)
        mean, lo, hi = boot(cand - base_ok, groups_for[fold])
        clean_d = 100 * (blob["conditions"]["clean"]["accuracy"]
                         - base["conditions"]["clean"]["accuracy"])
        out["entries"].append({
            "fold": fold, "run": name, "seed": seed,
            "combo_mean_pp": float(np.mean([blob["conditions"][c]["accuracy"] for c in combo])) * 100,
            "base_combo_mean_pp": float(np.mean([base["conditions"][c]["accuracy"] for c in combo])) * 100,
            "delta_pp": mean, "ci": [lo, hi], "clean_delta_pp": clean_d})
        print(json.dumps(out["entries"][-1]), flush=True)

    (PROJECT / args.out).write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
