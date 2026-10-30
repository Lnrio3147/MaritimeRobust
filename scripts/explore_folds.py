#!/usr/bin/env python
"""Build group-disjoint 3-fold manifests over the 618 training images.

Screening (plan stage S1) must not touch val: the 157 val images are the
confirmation set and the 236 test images the one-shot final set. This splits
only `split=="train"` rows, by `source_group_id` so that no vessel/scene
appears in both sides of a fold.

Writes data/manifests/folds/fgsc23_civ8_fold{k}.jsonl (k=0..2), each with the
fold's rows re-labelled split=train / split=val.

    python scripts/explore_folds.py [--folds 3]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from maritime_robust.common import ROOT as PROJECT, read_jsonl, write_jsonl  # noqa: E402

MANIFEST = PROJECT / "data/manifests/fgsc23_research_civilian.jsonl"
OUT = PROJECT / "data/manifests/folds"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--manifest", default=str(MANIFEST))
    args = ap.parse_args()

    rows = [r for r in read_jsonl(args.manifest) if r["split"] == "train"]
    if not rows:
        raise SystemExit("no train rows")
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["source_group_id"]].append(r)
    groups = sorted(by_group)
    print(json.dumps({"train_rows": len(rows), "groups": len(groups),
                      "folds": args.folds}), flush=True)

    # Greedy balancing: largest groups first into the currently smallest fold.
    load = [0] * args.folds
    assign = {}
    for g in sorted(groups, key=lambda g: (-len(by_group[g]), g)):
        k = min(range(args.folds), key=lambda i: (load[i], i))
        assign[g] = k
        load[k] += len(by_group[g])

    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for k in range(args.folds):
        train = [r for r in rows if assign[r["source_group_id"]] != k]
        val = [r for r in rows if assign[r["source_group_id"]] == k]
        train_groups = {r["source_group_id"] for r in train}
        val_groups = {r["source_group_id"] for r in val}
        overlap = train_groups & val_groups
        if overlap:
            raise SystemExit(f"fold {k}: {len(overlap)} shared source groups")
        payload = []
        for r in train:
            row = dict(r, split="train")
            payload.append(row)
        for r in val:
            row = dict(r, split="val")
            payload.append(row)
        path = OUT / f"fgsc23_civ8_fold{k}.jsonl"
        write_jsonl(path, payload)
        per_class = Counter(r["label"] for r in val)
        missing = [c for c in range(8) if per_class.get(c, 0) == 0]
        summary.append({"fold": k, "path": str(path.relative_to(PROJECT)),
                        "train": len(train), "val": len(val),
                        "train_groups": len(train_groups), "val_groups": len(val_groups),
                        "val_per_class": dict(sorted(per_class.items())),
                        "classes_missing_from_val": missing,
                        "protocol": payload[0].get("protocol")})
    write_jsonl(OUT / "summary.jsonl", summary)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
