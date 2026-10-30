#!/usr/bin/env python
"""Calibrate each added corruption family by its *marginal* accuracy cost.

Nominal severity is not comparable across families; pairing at matched nominal
level would confound the interaction term with unequal difficulty. This sweeps a
continuous level ladder per family on a frozen baseline and records the measured
drop, so that pairs can later be formed at matched marginal cost.

Results are cached per (family, level) and the run is resumable.

    python scripts/composition_calibrate.py --fold 0 --seed 17
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT, ROOT / "scripts"):
    sys.path.insert(0, str(path))

import explore_eval as ee  # noqa: E402
from maritime_robust.common import ROOT as PROJECT, read_jsonl, write_json  # noqa: E402
from maritime_robust.corruptions_extra import FAMILIES, apply  # noqa: E402
from maritime_robust.engine import metrics  # noqa: E402

MEAN, STD, FILL = [.485, .456, .406], [.229, .224, .225], (124, 116, 104)
LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def pad_norm(image):
    padded = ImageOps.pad(image, (224, 224), Image.Resampling.BICUBIC, color=FILL)
    return TF.normalize(TF.to_tensor(padded), MEAN, STD)


def score(model, tensor, labels, n, batch=64):
    preds = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for start in range(0, len(tensor), batch):
            preds.append(model(tensor[start:start + batch].cuda()).float().argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    return metrics(labels.tolist(), preds.tolist(), n)["accuracy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--model", default="convnext_tiny")
    ap.add_argument("--out-dir", default="reports/composition")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    args = ap.parse_args()

    run = PROJECT / (f"runs/fgsc23_civ8_fold{args.fold}/"
                     f"{args.model}_none_e40_s{args.seed}")
    model, settings = ee.load_any(run)
    rows = [r for r in read_jsonl(Path(settings["manifest"])) if r["split"] == "val"]
    labels = np.array([r["label"] for r in rows])
    contents = [ee.content(r) for r in rows]
    n = settings["num_classes"]

    out_dir = PROJECT / args.out_dir / f"ladder_fold{args.fold}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cells = {}
    todo = [("clean", 0.0)] + [(f, l) for f in FAMILIES for l in LEVELS]
    for family, level in todo:
        path = out_dir / f"{family}_{level:.1f}.json"
        if path.exists():
            cells[f"{family}_{level:.1f}"] = json.loads(path.read_text())
            continue
        if args.max_seconds and time.time() - started > args.max_seconds:
            print(f"budget reached at {len(cells)}/{len(todo)}", flush=True)
            break
        if family == "clean":
            tensor = torch.stack([pad_norm(image) for image in contents])
        else:
            tensor = torch.stack([pad_norm(apply(image, family, level, seed=i))
                                  for i, image in enumerate(contents)])
        acc = score(model, tensor, labels, n)
        cells[f"{family}_{level:.1f}"] = {"family": family, "level": level, "accuracy": acc}
        write_json(path, cells[f"{family}_{level:.1f}"])
        print(f"{family:9s} level {level:.1f}  acc {100*acc:6.2f}", flush=True)

    if len(cells) < len(todo):
        print(f"incomplete {len(cells)}/{len(todo)}; rerun to continue")
        return

    clean = cells["clean_0.0"]["accuracy"]
    table = {}
    for family in FAMILIES:
        table[family] = [{"level": l,
                          "accuracy": cells[f"{family}_{l:.1f}"]["accuracy"],
                          "drop_pp": 100 * (clean - cells[f"{family}_{l:.1f}"]["accuracy"])}
                         for l in LEVELS]
    write_json(PROJECT / args.out_dir / f"ladder_fold{args.fold}_s{args.seed}.json",
               {"run": str(run.relative_to(PROJECT)), "fold": args.fold, "seed": args.seed,
                "n_val": len(rows), "clean_accuracy": clean, "families": table})
    print()
    print(f"{'family':10s} " + " ".join(f"{l:6.1f}" for l in LEVELS))
    for family in FAMILIES:
        print(f"{family:10s} " + " ".join(f"{t['drop_pp']:6.1f}" for t in table[family]))


if __name__ == "__main__":
    main()
