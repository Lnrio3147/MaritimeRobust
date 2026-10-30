#!/usr/bin/env python
"""Stage 1 of the renderer-validity test (reports/PREREG_PHYSICAL_RENDERER.md).

Re-renders haze, glare and haze+glare with the physical model in
``maritime_robust.physical`` and scores an existing frozen baseline on them.
No training: the question is whether the superadditive structure survives a
different rendering of the same nominal severity.

Severity is matched on measurable quantities: haze by mean transmission, glare
by saturated-pixel fraction, both to what ``degradations.corrupt`` produces on
the same images.

    python scripts/physical_eval.py --fold 0 --seed 17
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
from maritime_robust.degradations import RECIPE, corrupt, corruption_seed  # noqa: E402
from maritime_robust.engine import metrics  # noqa: E402
from maritime_robust.physical import render  # noqa: E402

MEAN, STD, FILL = [.485, .456, .406], [.229, .224, .225], (124, 116, 104)
ROLE = "physical"
SEVERITIES = (2, 3, 4, 5)
SATURATED = 254 / 255


def pad_norm(image):
    padded = ImageOps.pad(image, (224, 224), Image.Resampling.BICUBIC, color=FILL)
    return TF.normalize(TF.to_tensor(padded), MEAN, STD)


def old_statistics(rows, contents, kind, severity):
    """Mean saturated fraction and transmission the original renderer yields."""
    saturated, transmission = [], []
    for row, image in zip(rows, contents):
        out, params = corrupt(image, kind=kind, severity=severity,
                              seed=corruption_seed(row["sample_id"], "target", kind,
                                                   severity, "matched"),
                              domain="matched", placement="random")
        a = np.array(out, dtype=np.float32) / 255
        saturated.append(float((a.max(2) >= SATURATED).mean()))
        transmission.append(params.get("transmission_mean",
                                       RECIPE["haze_transmission"][severity - 1]))
    return float(np.mean(saturated)), float(np.mean(transmission))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out-dir", default="reports/physical")
    ap.add_argument("--targets-only", action="store_true",
                    help="compute and cache the severity-matching targets, then exit")
    ap.add_argument("--run", default=None,
                    help="run directory to score (default: the frozen baseline for --seed)")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="stop cleanly after this many seconds; results are cached per "
                         "condition, so the next invocation resumes where this one stopped")
    args = ap.parse_args()

    manifest = PROJECT / f"data/manifests/folds/fgsc23_civ8_fold{args.fold}.jsonl"
    run = (PROJECT / args.run if args.run else
           PROJECT / f"runs/fgsc23_civ8_fold{args.fold}/convnext_tiny_none_e40_s{args.seed}")
    model, settings = ee.load_any(run)
    rows = [r for r in read_jsonl(manifest) if r["split"] == "val"]
    labels = np.array([r["label"] for r in rows])
    contents = [ee.content(r) for r in rows]
    n = settings["num_classes"]

    # targets: what the original renderer does at each severity on these images.
    # They depend only on the images and the corruption seeds, not on the model,
    # so they are cached once and shared by every seed.
    targets_path = PROJECT / "reports/physical/targets.json"
    if targets_path.exists():
        targets = {int(k): v for k, v in json.loads(targets_path.read_text()).items()}
    else:
        targets = {}
        for severity in SEVERITIES:
            glare_sat, _ = old_statistics(rows, contents, "glare", severity)
            compound_sat, _ = old_statistics(rows, contents, "haze+glare", severity)
            transmission = RECIPE["haze_transmission"][severity - 1]
            targets[severity] = {"haze_transmission": transmission,
                                 "glare_saturation": glare_sat,
                                 "compound_saturation": compound_sat}
            print(f"target s={severity}: t={transmission:.3f} glare_sat={glare_sat:.3f} "
                  f"compound_sat={compound_sat:.3f}", flush=True)
        targets_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(targets_path, {str(k): v for k, v in targets.items()})
    for severity, target in targets.items():
        print(f"target s={severity}: {json.dumps(target)}", flush=True)
    if args.targets_only:
        return

    conditions = [{"name": "clean", "kind": "clean", "severity": 0, "uniform": False}]
    for severity in SEVERITIES:
        conditions += [
            {"name": f"physical_haze_{severity}", "kind": "haze", "severity": severity,
             "uniform": False},
            {"name": f"physical_glare_{severity}", "kind": "glare", "severity": severity,
             "uniform": False},
            {"name": f"physical_haze+glare_{severity}", "kind": "haze+glare",
             "severity": severity, "uniform": False},
        ]
    # depth ablation: same mean transmission, no depth dependence, in both arms
    conditions += [{"name": f"physical_haze_uniform_{s}", "kind": "haze", "severity": s,
                    "uniform": True} for s in (3, 5)]
    conditions += [{"name": f"physical_haze_uniform+glare_{s}", "kind": "haze+glare",
                    "severity": s, "uniform": True} for s in (3, 5)]

    # One file per condition: a killed run loses at most the condition in flight.
    cell_dir = PROJECT / args.out_dir / "cells" / run.name
    cell_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cells = {}
    for cond in conditions:
        path = cell_dir / f"{cond['name']}.json"
        if path.exists():
            cells[cond["name"]] = json.loads(path.read_text())
            continue
        if args.max_seconds and time.time() - started > args.max_seconds:
            print(f"time budget reached after {len(cells)}/{len(conditions)} conditions",
                  flush=True)
            break
        batch, stats = [], []
        for row, image in zip(rows, contents):
            if cond["kind"] == "clean":
                batch.append(pad_norm(image))
                continue
            target = targets[cond["severity"]]
            if cond["kind"] == "haze":
                saturation = 0.0
            elif cond["kind"] == "glare":
                saturation = target["glare_saturation"]
            else:
                saturation = target["compound_saturation"]
            rendered, params = render(
                image, cond["kind"], target_transmission=target["haze_transmission"],
                target_saturation=saturation,
                seed=corruption_seed(row["sample_id"], ROLE, cond["kind"],
                                     cond["severity"], "matched"),
                uniform_haze=cond["uniform"])
            batch.append(pad_norm(rendered))
            stats.append(params)
        tensor = torch.stack(batch)
        preds = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for start in range(0, len(tensor), args.batch):
                preds.append(model(tensor[start:start + args.batch].cuda()).float()
                             .argmax(1).cpu())
        preds = torch.cat(preds).numpy()
        score = metrics(labels.tolist(), preds.tolist(), n)
        cells[cond["name"]] = {
            "kind": cond["kind"], "severity": cond["severity"], "uniform_haze": cond["uniform"],
            "accuracy": score["accuracy"],
            "measured_saturated_fraction": float(np.mean([s["saturated_fraction"] for s in stats]))
            if stats else None,
            "measured_transmission": float(np.mean([s["haze_transmission_mean"] for s in stats]))
            if stats and "haze_transmission_mean" in stats[0] else None,
            "predictions": preds.tolist()}
        write_json(path, cells[cond["name"]])
        print(f"{cond['name']:28s} acc {100*score['accuracy']:6.2f} "
              f"sat {100*(cells[cond['name']]['measured_saturated_fraction'] or 0):5.1f}%",
              flush=True)

    if len(cells) < len(conditions):
        missing = [c["name"] for c in conditions if c["name"] not in cells]
        print(f"incomplete: {len(cells)}/{len(conditions)} conditions cached")
        print("missing:", " ".join(missing))
        return

    clean = cells["clean"]["accuracy"]
    excess = {}
    for severity in SEVERITIES:
        a = cells[f"physical_haze_{severity}"]["accuracy"]
        b = cells[f"physical_glare_{severity}"]["accuracy"]
        c = cells[f"physical_haze+glare_{severity}"]["accuracy"]
        excess[severity] = {
            "E_pp": 100 * ((clean - c) - ((clean - a) + (clean - b))),
            "drop_compound_pp": 100 * (clean - c),
            "drop_haze_pp": 100 * (clean - a),
            "drop_glare_pp": 100 * (clean - b),
            "target_saturation": targets[severity]["compound_saturation"],
            "measured_saturation": cells[f"physical_haze+glare_{severity}"]
            ["measured_saturated_fraction"],
        }
    # P4: the same excess with a depth-uniform haze, both arms built the same way
    ablation = {}
    for severity in (3, 5):
        a = cells[f"physical_haze_uniform_{severity}"]["accuracy"]
        b = cells[f"physical_glare_{severity}"]["accuracy"]
        c = cells[f"physical_haze_uniform+glare_{severity}"]["accuracy"]
        uniform = 100 * ((clean - c) - ((clean - a) + (clean - b)))
        ablation[severity] = {"E_uniform_pp": uniform, "E_depth_pp": excess[severity]["E_pp"],
                              "delta_pp": excess[severity]["E_pp"] - uniform}
    out ={"run": str(run.relative_to(PROJECT)), "fold": args.fold, "seed": args.seed,
           "n_val": len(rows), "targets": {str(k): v for k, v in targets.items()},
           "cells": cells, "excess": {str(k): v for k, v in excess.items()},
           "depth_ablation": {str(k): v for k, v in ablation.items()}}
    out_dir = PROJECT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"fold{args.fold}_s{args.seed}.json" if not args.run else f"{run.name}.json"
    write_json(out_dir / name, out)

    print()
    print(f"{'s':>2s} {'E_pp':>7s} {'drop_comp':>10s} {'drop_haze':>10s} {'drop_glare':>11s} "
          f"{'sat_target':>11s} {'sat_meas':>9s}")
    for severity, cell in excess.items():
        print(f"{severity:2d} {cell['E_pp']:+7.2f} {cell['drop_compound_pp']:10.2f} "
              f"{cell['drop_haze_pp']:10.2f} {cell['drop_glare_pp']:11.2f} "
              f"{cell['target_saturation']:11.3f} {cell['measured_saturation']:9.3f}")


if __name__ == "__main__":
    main()
