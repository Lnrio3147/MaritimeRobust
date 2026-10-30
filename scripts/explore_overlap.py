#!/usr/bin/env python
"""Error-fix overlap: which samples a candidate fixes vs breaks, relative to the
same-fold same-seed reference arm.

For every (fold, seed) pair it counts, over the 8 combo conditions and per
sample:

    fixed   reference wrong, candidate right
    broken  reference right, candidate wrong
    net     fixed - broken            (== delta in correct predictions)

and reports the pooled counts plus the per-condition breakdown, so a gain can
be read as "repairs more than it damages" rather than as a single accuracy
number.

    python scripts/explore_overlap.py --dir reports/s2b --suffix _e40 --method i2
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


def load(directory, suffix):
    runs = {}
    for path in sorted((PROJECT / directory).glob("*.json")):
        if path.name.startswith("_"):
            continue
        blob = json.loads(path.read_text())
        runs[(blob["run"].split("/")[1], Path(blob["run"]).name)] = blob
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="reports/s2b")
    ap.add_argument("--suffix", default="_e40")
    ap.add_argument("--base", default="convnext_tiny_none")
    ap.add_argument("--methods", nargs="*", default=None)
    args = ap.parse_args()

    runs = load(args.dir, args.suffix)
    labels = {}
    for fold in {f for f, _ in runs}:
        labels[fold] = np.array([r["label"] for r in
                                 read_jsonl(PROJECT / f"data/manifests/folds/{fold}.jsonl")
                                 if r["split"] == "val"])
    methods = args.methods or sorted({n.rsplit("_s", 1)[0] for _, n in runs
                                      if not n.startswith(args.base)})

    print(f"{'method':22} {'pairs':>5} {'fixed':>6} {'broken':>6} {'net':>5} "
          f"{'per-cond net (top/bottom 2)'}")
    for method in methods:
        fixed = broken = pairs = 0
        per_condition = defaultdict(float)
        for (fold, name), blob in sorted(runs.items()):
            if not name.startswith(method + "_s"):
                continue
            ref = runs.get((fold, f"{args.base}{args.suffix}_s{blob['seed']}"))
            if ref is None:
                continue
            pairs += 1
            truth = labels[fold]
            for cond in blob.get("combo_conditions", []):
                cand = np.array(blob["conditions"][cond]["predictions"]) == truth
                base = np.array(ref["conditions"][cond]["predictions"]) == truth
                fixed += int((~base & cand).sum())
                broken += int((base & ~cand).sum())
                per_condition[cond] += float(cand.sum() - base.sum())
        if not pairs:
            continue
        ranked = sorted(per_condition.items(), key=lambda kv: kv[1])
        detail = " ".join(f"{k.split('_', 1)[1]}:{v:+.0f}" for k, v in ranked[-2:][::-1])
        detail += " | " + " ".join(f"{k.split('_', 1)[1]}:{v:+.0f}" for k, v in ranked[:2])
        print(f"{method:22} {pairs:5d} {fixed:6d} {broken:6d} {fixed - broken:+5d}  {detail}")


if __name__ == "__main__":
    main()
