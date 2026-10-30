#!/usr/bin/env python
"""Pool a screening/confirmation directory into one delta estimate per method.

Differences every run against the same-fold same-seed reference on the mean of
the 8 combo_both conditions, clusters the per-sample deltas by
(fold, source_group_id) so both seeds of a cluster stay together, and
bootstraps over clusters.

    python scripts/explore_pooled.py --dir reports/s2 --suffix _e40 --base none
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from maritime_robust.common import ROOT as PROJECT, read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="reports/s1")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--base", default="convnext_tiny_none")
    ap.add_argument("--reps", type=int, default=4000)
    args = ap.parse_args()

    runs = {}
    for path in sorted((PROJECT / args.dir).glob("*.json")):
        if path.name.startswith("_"):
            continue
        blob = json.loads(path.read_text())
        runs[(blob["run"].split("/")[1], Path(blob["run"]).name)] = blob

    labels, groups = {}, {}
    for fold in {f for f, _ in runs}:
        rows = [r for r in read_jsonl(PROJECT / f"data/manifests/folds/{fold}.jsonl")
                if r["split"] == "val"]
        labels[fold] = np.array([r["label"] for r in rows])
        groups[fold] = np.array([r["source_group_id"] for r in rows])

    methods = sorted({n.rsplit("_s", 1)[0] for _, n in runs})
    out = {"dir": args.dir, "suffix": args.suffix, "methods": {}}
    for method in methods:
        per = defaultdict(list)
        for (fold, name), blob in runs.items():
            if not name.startswith(method + "_s"):
                continue
            seed = blob["seed"]
            base = runs.get((fold, f"{args.base}{args.suffix}_s{seed}"))
            if base is None or base is blob:
                continue
            combo = blob.get("combo_conditions") or [c for c in blob["conditions"] if c != "clean"]
            cand = np.mean([np.array(blob["conditions"][c]["predictions"]) == labels[fold]
                            for c in combo], 0)
            ref = np.mean([np.array(base["conditions"][c]["predictions"]) == labels[fold]
                           for c in combo], 0)
            for g, v in zip(groups[fold], cand - ref):
                per[(fold, g)].append(v)
        if not per:
            continue
        keys = list(per)
        values = np.array([np.mean(per[k]) for k in keys])
        rng = np.random.default_rng(0)
        stats = np.array([100 * values[rng.integers(0, len(values), len(values))].mean()
                          for _ in range(args.reps)])
        out["methods"][method] = {
            "delta_pp": 100 * float(values.mean()),
            "ci": [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))],
            "clusters": len(keys),
            "measurements": sum(len(v) for v in per.values()),
        }
        e = out["methods"][method]
        print(f"{method:28} Δ={e['delta_pp']:+6.2f} "
              f"CI[{e['ci'][0]:+6.2f},{e['ci'][1]:+6.2f}] clusters={e['clusters']}")
    (PROJECT / args.dir / "_pooled.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
