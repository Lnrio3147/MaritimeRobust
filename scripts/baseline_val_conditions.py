#!/usr/bin/env python
"""Score the 7 frozen combined-augmentation classifiers on all 43 conditions (val).

Same source images, same corruption parameters and seeds as `scripts/efmnet_eval.py`
(role "efmnet"), same project preprocessing. Predictions are therefore paired
per image across every model, which is what the cross-model comparison needs.

Writes, per classifier:
    reports/conditions_val/<model>/<condition>.jsonl   one line per sample
and a single summary:
    reports/conditions_val/summary.json

    python scripts/baseline_val_conditions.py [--split val]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from maritime_robust.common import ROOT as PROJECT, read_jsonl, write_json  # noqa: E402
from maritime_robust.degradations import conditions, corrupt, corruption_seed  # noqa: E402
from maritime_robust.engine import load_run, metrics  # noqa: E402

MEAN, STD, FILL = [.485, .456, .406], [.229, .224, .225], (124, 116, 104)
ROLE = "efmnet"  # must match scripts/efmnet_eval.py so the images are identical
RUNS = ("convnext_tiny_combo_s17", "convnext_tiny_combo_s29", "convnext_tiny_combo_s43",
        "swin_t_combo_s17", "swin_t_combo_s29", "swin_t_combo_s43", "resnet50_combo_s17")


def content(row):
    with Image.open(PROJECT / row["path"]) as handle:
        image = handle.convert("RGB")
    return ImageOps.contain(image, (224, 224), Image.Resampling.BICUBIC)


def pad_norm(image):
    padded = ImageOps.pad(image, (224, 224), Image.Resampling.BICUBIC, color=FILL)
    return TF.normalize(TF.to_tensor(padded), MEAN, STD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/fgsc23_research_civilian.jsonl")
    ap.add_argument("--split", default="val")
    ap.add_argument("--runs", default=",".join(RUNS))
    ap.add_argument("--out-dir", default="reports/conditions_val")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    out_dir = PROJECT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_jsonl(PROJECT / args.manifest) if r["split"] == args.split]
    if not rows:
        raise SystemExit(f"no {args.split} rows")
    labels = np.array([r["label"] for r in rows])

    models, num_classes = {}, 0
    for name in args.runs.split(","):
        model, settings, _ = load_run(PROJECT / "runs/fgsc23_research_civilian" / name)
        models[name] = model
        num_classes = settings["num_classes"]
    print(json.dumps({"models": list(models), "samples": len(rows)}), flush=True)

    contents = [content(r) for r in rows]
    summary = {"split": args.split, "role": ROLE, "n": int(len(labels)),
               "models": {}, "conditions": {}}

    for cond in conditions():
        batch = []
        for row, image in zip(rows, contents):
            out, params = corrupt(image, kind=cond["kind"], severity=cond["severity"],
                                  seed=corruption_seed(row["sample_id"], ROLE, cond["kind"],
                                                       cond["severity"], cond["domain"]),
                                  domain=cond["domain"], placement=cond["placement"])
            batch.append(pad_norm(out))
        tensor = torch.stack(batch)
        summary["conditions"][cond["name"]] = {k: cond[k] for k in ("kind", "severity", "domain",
                                                                    "placement")}
        for name, model in models.items():
            probs = []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                for start in range(0, len(tensor), args.batch):
                    probs.append(model(tensor[start:start + args.batch].cuda()).float().softmax(1).cpu())
            probs = torch.cat(probs).numpy()
            pred = probs.argmax(1)
            score = metrics(labels.tolist(), pred.tolist(), num_classes)
            summary["models"].setdefault(name, {})[cond["name"]] = {
                "accuracy": score["accuracy"], "macro_f1": score["macro_f1"],
                "per_class_recall": score["per_class_recall"]}
            model_dir = out_dir / name
            model_dir.mkdir(exist_ok=True)
            with (model_dir / f"{cond['name']}.jsonl").open("w") as handle:
                for row, p, vector in zip(rows, pred.tolist(), probs):
                    handle.write(json.dumps({"sample_id": row["sample_id"],
                                             "source_group_id": row["source_group_id"],
                                             "label": row["label"], "prediction": p,
                                             "probability": float(vector[p]),
                                             "probabilities": [float(v) for v in vector]}) + "\n")
        print(json.dumps({cond["name"]: {n: round(summary["models"][n][cond["name"]]["accuracy"], 4)
                                         for n in models}}), flush=True)

    write_json(out_dir / "summary.json", summary)
    print("wrote", out_dir / "summary.json", flush=True)


if __name__ == "__main__":
    main()
