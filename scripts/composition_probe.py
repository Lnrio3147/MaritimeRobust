#!/usr/bin/env python
"""Feasibility probe: does superadditivity generalise beyond the weather families?

Evaluates single corruptions from ``corruptions_extra`` and a few pairs at the
same nominal severity on a frozen baseline, then reports
``E = drop(pair) - [drop(a) + drop(b)]``. No training. This decides whether a
full composition study has anything to study.

    python scripts/composition_probe.py --seed 17 --severities 3 5
"""
from __future__ import annotations

import argparse
import itertools
import sys
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
from maritime_robust.corruptions_extra import FAMILIES, apply, compose  # noqa: E402
from maritime_robust.engine import metrics  # noqa: E402

MEAN, STD, FILL = [.485, .456, .406], [.229, .224, .225], (124, 116, 104)
PAIRS = (("defocus", "noise"), ("defocus", "rain"), ("motion", "noise"),
         ("darken", "contrast"), ("darken", "noise"))


def pad_norm(image):
    padded = ImageOps.pad(image, (224, 224), Image.Resampling.BICUBIC, color=FILL)
    return TF.normalize(TF.to_tensor(padded), MEAN, STD)


def score(model, tensor, labels, n, batch=64):
    preds = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for start in range(0, len(tensor), batch):
            preds.append(model(tensor[start:start + batch].cuda()).float().argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    return metrics(labels.tolist(), preds.tolist(), n)["accuracy"], preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--severities", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--out", default="reports/composition_probe.json")
    args = ap.parse_args()

    run = PROJECT / f"runs/fgsc23_civ8_fold{args.fold}/convnext_tiny_none_e40_s{args.seed}"
    model, settings = ee.load_any(run)
    rows = [r for r in read_jsonl(Path(settings["manifest"])) if r["split"] == "val"]
    labels = np.array([r["label"] for r in rows])
    contents = [ee.content(r) for r in rows]
    n = settings["num_classes"]

    def evaluate(fn, tag):
        tensor = torch.stack([pad_norm(fn(i, row)) for i, row in enumerate(rows)])
        acc, preds = score(model, tensor, labels, n)
        print(f"{tag:28s} acc {100*acc:6.2f}", flush=True)
        return acc, preds

    cells = {}
    acc, preds = evaluate(lambda i, r: contents[i], "clean")
    cells["clean"] = {"accuracy": acc, "predictions": preds.tolist()}
    clean = acc
    for s in args.severities:
        for family in FAMILIES:
            cells[f"{family}_{s}"] = dict(zip(("accuracy", "predictions"), evaluate(
                lambda i, r, f=family, s=s: apply(contents[i], f, s, seed=i), f"{family}_{s}")))
        for a, b in PAIRS:
            cells[f"{a}+{b}_{s}"] = dict(zip(("accuracy", "predictions"), evaluate(
                lambda i, r, A=a, B=b, s=s: compose(contents[i], A, B, s, seed=i),
                f"{a}+{b}_{s}")))

    print()
    print(f"{'pair':24s} {'s':>2s} {'drop_pair':>10s} {'drop_a':>8s} {'drop_b':>8s} {'E_pp':>7s}")
    excess = {}
    for s in args.severities:
        for a, b in PAIRS:
            dp = 100 * (clean - cells[f"{a}+{b}_{s}"]["accuracy"])
            da = 100 * (clean - cells[f"{a}_{s}"]["accuracy"])
            db = 100 * (clean - cells[f"{b}_{s}"]["accuracy"])
            e = dp - (da + db)
            excess[f"{a}+{b}_{s}"] = {"drop_pair_pp": dp, "drop_a_pp": da, "drop_b_pp": db,
                                      "E_pp": e}
            print(f"{a+'+'+b:24s} {s:2d} {dp:10.2f} {da:8.2f} {db:8.2f} {e:+7.2f}")

    write_json(PROJECT / args.out, {"run": str(run.relative_to(PROJECT)), "fold": args.fold,
                                    "seed": args.seed, "n_val": len(rows),
                                    "cells": {k: {"accuracy": v["accuracy"]}
                                              for k, v in cells.items()},
                                    "excess": excess})


if __name__ == "__main__":
    main()
