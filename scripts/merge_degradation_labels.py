#!/usr/bin/env python
"""Validate a review export and write the confirmed label table.

The review page exports every image, confirmed or not.  This script refuses to
produce labels unless every image is confirmed exactly once and the values are
consistent with the taxonomy.  It never fills in or overrides a label.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maritime_robust.common import ROOT as PROJECT, digest  # noqa: E402

CIVILIAN = {14: 0, 15: 1, 17: 2, 18: 3, 19: 4, 20: 5, 21: 6, 22: 7}
TYPES = {"cloud", "haze", "shadow", "glare", "blur"}
SEVERITIES = {"none", "mild", "medium", "severe", "unable"}
DEGRADED_SEVERITIES = {"mild", "medium", "severe", "unable"}


def fail(message):
    raise SystemExit(f"label merge refused: {message}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("fgsc23_val", "fgsrcs_dev"), required=True)
    parser.add_argument("--labels", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    root = PROJECT / f"reviews/real_degradation/{args.dataset}"
    source = Path(args.labels) if args.labels else root / f"labels_{args.dataset}.json"
    index = json.loads((root / "index.json").read_text())
    exported = json.loads(source.read_text())

    by_review = {row["review_index"]: row for row in index}
    if len(by_review) != len(index):
        fail("duplicate review_index in index.json")
    seen = {}
    for row in exported:
        number = row.get("review_index")
        if number not in by_review:
            fail(f"unknown review_index {number}")
        if number in seen:
            fail(f"duplicate review_index {number}")
        seen[number] = row
    missing = sorted(set(by_review) - set(seen))
    if missing:
        fail(f"{len(missing)} images missing from export: {missing[:10]}")

    records = []
    for number in sorted(by_review):
        row, meta = seen[number], by_review[number]
        if row.get("sample_id") != meta["sample_id"]:
            fail(f"sample_id mismatch at {number}")
        if row.get("confirmed") is not True:
            fail(f"image {number} ({meta['sample_id']}) is not confirmed")
        types = row.get("degradation_types")
        if not isinstance(types, list) or not set(types) <= TYPES:
            fail(f"invalid degradation_types at {number}: {types!r}")
        if len(types) != len(set(types)):
            fail(f"duplicate degradation_types at {number}")
        severity = row.get("severity")
        if severity not in SEVERITIES:
            fail(f"invalid severity at {number}: {severity!r}")
        if types and severity not in DEGRADED_SEVERITIES:
            fail(f"image {number} has types but severity {severity!r}")
        if not types and severity in {"mild", "medium", "severe"}:
            fail(f"image {number} has severity {severity!r} but no type")
        label = meta.get("label")
        if label is None:
            original = (meta.get("mapping") or {}).get("fgsc23_original_label")
            if original is None or int(original) not in CIVILIAN:
                fail(f"image {number} has no civilian label")
            label = CIVILIAN[int(original)]
        records.append({
            "review_index": number, "sample_id": meta["sample_id"],
            "label": label, "label_name": meta["label_name"],
            "mapping": meta.get("mapping"), "source_group_id": meta["source_group_id"],
            "path": meta["path"], "degradation_types": sorted(types),
            "severity": severity, "clutter": bool(row.get("clutter")),
            "uncertain": bool(row.get("uncertain")), "note": row.get("note") or "",
        })

    output = Path(args.output) if args.output else root / "labels.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    summary = {
        "dataset": args.dataset, "n": len(records),
        "source_export": str(source.relative_to(PROJECT)) if source.is_relative_to(PROJECT) else str(source),
        "source_sha256": digest(source),
        "labels": str(output.relative_to(PROJECT)) if output.is_relative_to(PROJECT) else str(output),
        "labels_sha256": digest(output),
        "clear": sum(1 for r in records if not r["degradation_types"]),
        "degraded": sum(1 for r in records if r["degradation_types"]),
        "uncertain": sum(1 for r in records if r["uncertain"]),
        "types": {t: sum(1 for r in records if t in r["degradation_types"])
                  for t in sorted(TYPES)},
        "severity": {s: sum(1 for r in records if r["severity"] == s)
                     for s in sorted(SEVERITIES)},
        "clutter": sum(1 for r in records if r["clutter"]),
    }
    summary_path = (root / "labels_summary.json" if output == root / "labels.jsonl"
                    else output.with_suffix(".summary.json"))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
