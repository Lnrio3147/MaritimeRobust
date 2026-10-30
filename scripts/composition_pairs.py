#!/usr/bin/env python
"""Pair every added corruption family at matched *marginal* cost.

Each family's level is chosen so that, alone, it costs the same number of
accuracy points on the frozen baseline (10 / 20 / 30 pp). Pairs are then formed
at equal marginal cost, and the interaction is

    E = drop(a+b) - [drop(a) + drop(b)]

which is comparable across pairs because the marginals are equated by
construction. Results are cached per (condition) and the run is resumable.

    python scripts/composition_pairs.py --fold 0 --seed 17 --targets 10 20 30
"""
from __future__ import annotations

import argparse
import itertools
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


def level_for(ladder, family, target_pp):
    """Smallest level whose measured drop reaches the target, interpolated."""
    pts = [(t["level"], t["drop_pp"]) for t in ladder["families"][family]]
    for (l0, d0), (l1, d1) in zip(pts, pts[1:]):
        if d0 <= target_pp <= d1 and d1 > d0:
            return l0 + (target_pp - d0) / (d1 - d0) * (l1 - l0)
    return pts[-1][0] if pts[-1][1] >= target_pp else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--targets", type=float, nargs="+", default=[10.0, 20.0, 30.0])
    ap.add_argument("--ladder", default=None)
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

    ladder_path = (PROJECT / args.ladder if args.ladder else
                   PROJECT / args.out_dir / f"ladder_fold{args.fold}_s{args.seed}.json")
    ladder = json.loads(ladder_path.read_text())

    levels = {}
    for target in args.targets:
        for family in FAMILIES:
            level = level_for(ladder, family, target)
            if level is None:
                print(f"skip {family} @ {target}: ladder tops out at "
                      f"{ladder['families'][family][-1]['drop_pp']:.1f} pp")
                continue
            levels[(family, target)] = round(float(level), 3)
    print("calibrated levels:", json.dumps({f"{k[0]}@{k[1]:.0f}": v
                                            for k, v in levels.items()}), flush=True)

    cell_dir = PROJECT / args.out_dir / f"pairs_fold{args.fold}_s{args.seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cells = {}

    def evaluate(key, fn):
        acc = score(model, torch.stack([pad_norm(fn(i, im)) for i, im in enumerate(contents)]),
                    labels, n)
        cells[key] = {"accuracy": acc}
        write_json(cell_dir / f"{key}.json", cells[key])
        print(f"{key:34s} acc {100*acc:6.2f}", flush=True)
        return acc

    clean_path = cell_dir / "clean.json"
    if clean_path.exists():
        cells["clean"] = json.loads(clean_path.read_text())
    elif not (args.max_seconds and time.time() - started > args.max_seconds):
        evaluate("clean", lambda i, im: im)

    for (family, target), level in levels.items():
        key = f"single_{family}_{target:.0f}"
        if (cell_dir / f"{key}.json").exists():
            cells[key] = json.loads((cell_dir / f"{key}.json").read_text())
            continue
        if args.max_seconds and time.time() - started > args.max_seconds:
            print("budget reached"); break
        evaluate(key, lambda i, im, f=family, l=level: apply(im, f, l, seed=i))

    pairs = list(itertools.combinations(FAMILIES, 2))
    for target in args.targets:
        usable = [f for f in FAMILIES if (f, target) in levels]
        for a, b in pairs:
            if a not in usable or b not in usable:
                continue
            key = f"pair_{a}+{b}_{target:.0f}"
            if (cell_dir / f"{key}.json").exists():
                cells[key] = json.loads((cell_dir / f"{key}.json").read_text())
                continue
            if args.max_seconds and time.time() - started > args.max_seconds:
                print("budget reached"); break
            la, lb = levels[(a, target)], levels[(b, target)]
            evaluate(key, lambda i, im, A=a, B=b, LA=la, LB=lb: apply(
                apply(im, A, LA, seed=i), B, LB, seed=i + 1))

    if "clean" not in cells:
        print("incomplete: no clean cell yet")
        return
    clean = cells["clean"]["accuracy"]
    excess = {}
    for target in args.targets:
        usable = [f for f in FAMILIES if (f, target) in levels]
        for a, b in itertools.combinations(usable, 2):
            key = f"pair_{a}+{b}_{target:.0f}"
            if key not in cells:
                continue
            da = 100 * (clean - cells[f"single_{a}_{target:.0f}"]["accuracy"])
            db = 100 * (clean - cells[f"single_{b}_{target:.0f}"]["accuracy"])
            dp = 100 * (clean - cells[key]["accuracy"])
            excess[key] = {"target_pp": target, "drop_a_pp": da, "drop_b_pp": db,
                           "drop_pair_pp": dp, "E_pp": dp - (da + db)}
    if not excess:
        print("incomplete: no pairs yet; rerun to continue")
        return

    print()
    print(f"{'pair':32s} {'target':>6s} {'drop_a':>7s} {'drop_b':>7s} {'drop_ab':>8s} {'E':>7s}")
    for key, v in sorted(excess.items(), key=lambda kv: -kv[1]["E_pp"]):
        print(f"{key[5:-3]:32s} {v['target_pp']:6.0f} {v['drop_a_pp']:7.1f} "
              f"{v['drop_b_pp']:7.1f} {v['drop_pair_pp']:8.1f} {v['E_pp']:+7.1f}")
    write_json(PROJECT / args.out_dir / f"pairs_fold{args.fold}_s{args.seed}.json",
               {"run": str(run.relative_to(PROJECT)), "fold": args.fold, "seed": args.seed,
                "n_val": len(rows), "clean_accuracy": clean,
                "levels": {f"{k[0]}@{k[1]:.0f}": v for k, v in levels.items()},
                "cells": {k: {"accuracy": v["accuracy"]} for k, v in cells.items()},
                "excess": excess})


if __name__ == "__main__":
    main()
