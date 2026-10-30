#!/usr/bin/env python
"""Freeze per-sample predictions for real-degradation strata.

FGSC-23 validation reuses the clean identity predictions already saved by
`restorer_eval.py`; FGSRCS development crops are scored once by the seven
frozen combined-augmentation classifiers with the same contain/pad pipeline.
Predictions are label-independent, so they can be cached before review ends.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from maritime_robust.common import (ROOT as PROJECT, Budget, digest, environment,  # noqa: E402
                                    read_jsonl, write_json, write_jsonl)
from maritime_robust.engine import load_run  # noqa: E402
from restorer_eval import classifier_runs, content, pad_norm, probs_for  # noqa: E402

FGSC23_RECORDS = PROJECT / "reports/restorer_eval_val.records.jsonl"
FGSC23_MANIFEST = PROJECT / "data/manifests/fgsc23_research_civilian.jsonl"
FGSRCS_MANIFEST = PROJECT / "data/manifests/fgsrcs_known_box_crops.jsonl"


def rows_fgsc23():
    rows = [r for r in read_jsonl(FGSC23_MANIFEST) if r["split"] == "val"]
    return rows, {r["sample_id"]: r["label"] for r in rows}


def rows_fgsrcs():
    rows = [r for r in read_jsonl(FGSRCS_MANIFEST)
            if r.get("strict_fgsc23_mapping") and r["split"] == "development"]
    labels = {}
    for row in rows:
        original = row["strict_fgsc23_mapping"]["fgsc23_original_label"]
        civilian = {14: 0, 15: 1, 17: 2, 18: 3, 19: 4, 20: 5, 21: 6, 22: 7}
        if original not in civilian:
            raise ValueError(f"{row['sample_id']}: unmapped original label {original}")
        labels[row["sample_id"]] = civilian[original]
    return rows, labels


def fgsc23_predictions():
    records = [r for r in read_jsonl(FGSC23_RECORDS)
               if r["condition"] == "clean" and r["variant"] == "identity"]
    if not records:
        raise ValueError("no clean identity records; run restorer_eval.py --split val first")
    return records


@torch.inference_mode()
def fgsrcs_predictions(rows, labels):
    classifiers = classifier_runs()
    if len(classifiers) != 7:
        raise ValueError(f"expected 7 frozen classifiers, found {len(classifiers)}")
    tensors = [pad_norm(content(row)) for row in rows]
    records = []
    with Budget("evaluation"):
        for name, info in sorted(classifiers.items()):
            model, settings, _ = load_run(info["path"])
            # Labels are the two mapped civilian classes, but classifiers keep
            # their full 8-class heads; only the mapped head is interpreted.
            if settings["num_classes"] != 8:
                raise ValueError(f"{name}: expected 8 classes")
            probs = probs_for(model, tensors)
            for row, p in zip(rows, probs):
                records.append({
                    "classifier": name, "sample_id": row["sample_id"],
                    "label": labels[row["sample_id"]],
                    "prediction": int(p.argmax()),
                    "probabilities": [round(float(v), 6) for v in p],
                })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("fgsc23_val", "fgsrcs_dev"), required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if args.dataset == "fgsc23_val":
        _, labels = rows_fgsc23()
        records = fgsc23_predictions()
        manifest = FGSC23_MANIFEST
    else:
        rows, labels = rows_fgsrcs()
        records = fgsrcs_predictions(rows, labels)
        manifest = FGSRCS_MANIFEST

    output = Path(args.output) if args.output else (
        PROJECT / f"reports/real_degradation_predictions_{args.dataset}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, records)
    meta = {
        "dataset": args.dataset, "records": len(records),
        "classifiers": sorted({r["classifier"] for r in records}),
        "samples": len({r["sample_id"] for r in records}),
        "manifest": str(manifest.relative_to(PROJECT)),
        "manifest_sha256": digest(manifest),
        "output": str(output.relative_to(PROJECT)),
        "output_sha256": digest(output),
        "label_source": ("restorer_eval_val.records.jsonl: clean/identity"
                         if args.dataset == "fgsc23_val" else
                         "frozen classifiers on FGSRCS development crops"),
        "environment": environment(),
    }
    write_json(output.with_suffix(".meta.json"), meta)
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
