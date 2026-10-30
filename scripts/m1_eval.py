#!/usr/bin/env python
"""Kill test for D1 (clipping-aware recognition, method ``m1``).

Scores two runs on the identical 43 conditions of one fold's val rows, so every
number is paired per image:

    * the S2 baseline arm (plain ConvNeXt-T, combined augmentation);
    * the m1 arm (4-channel input, zero-init mask stem, censored-evidence pooling).

The m1 arm receives the clipping mask as a fourth channel, computed exactly as
at training time (``methods.clip_mask`` on the padded [0, 1] tensor).

    python scripts/m1_eval.py --baseline runs/fgsc23_civ8_fold0/convnext_tiny_none_e40_s17 \
        --treatment runs/fgsc23_civ8_fold0/convnext_tiny_m1_e40_s17
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
sys.path.insert(0, str(ROOT / "scripts"))

import explore_eval as ee  # noqa: E402
from maritime_robust.common import ROOT as PROJECT, read_jsonl, write_json  # noqa: E402
from maritime_robust.degradations import conditions, corrupt, corruption_seed  # noqa: E402
from maritime_robust.engine import metrics  # noqa: E402
from maritime_robust.methods import clip_mask  # noqa: E402

MEAN, STD, FILL = [.485, .456, .406], [.229, .224, .225], (124, 116, 104)
ROLE = "efmnet"


def batch_of(rows, contents, cond, with_mask):
    out = []
    for row, image in zip(rows, contents):
        if cond["kind"] == "clean":
            degraded = image
        else:
            degraded, _ = corrupt(
                image, kind=cond["kind"], severity=cond["severity"],
                seed=corruption_seed(row["sample_id"], ROLE, cond["kind"],
                                     cond["severity"], cond["domain"]),
                domain=cond["domain"], placement=cond["placement"])
        raw = TF.to_tensor(ImageOps.pad(degraded, (224, 224), Image.Resampling.BICUBIC, color=FILL))
        tensor = TF.normalize(raw, MEAN, STD)
        if with_mask:
            tensor = torch.cat([tensor, clip_mask(raw)], 0)
        out.append(tensor)
    return torch.stack(out)


def run_predictions(model, rows, contents, conds, with_mask, batch=64):
    preds = {}
    for cond in conds:
        tensor = batch_of(rows, contents, cond, with_mask)
        got = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for start in range(0, len(tensor), batch):
                got.append(model(tensor[start:start + batch].cuda()).float().argmax(1).cpu())
        preds[cond["name"]] = torch.cat(got).numpy()
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--treatment", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default="reports/kill_test_m1.json")
    ap.add_argument("--dump-preds", default=None,
                    help="optional .npz path: per-image predictions for both arms, "
                         "so a paired test is possible without re-running the models")
    args = ap.parse_args()

    base_run = PROJECT / args.baseline
    treat_run = PROJECT / args.treatment
    base_model, base_settings = ee.load_any(base_run)
    if base_settings.get("method") != "none":
        raise SystemExit(f"baseline arm is method={base_settings.get('method')}, expected none")
    treat_model, treat_settings = ee.load_any(treat_run)
    if treat_settings.get("method") != "m1":
        raise SystemExit(f"treatment arm is method={treat_settings.get('method')}, expected m1")
    if treat_settings["manifest_sha256"] != base_settings["manifest_sha256"]:
        raise SystemExit("the two arms were trained on different manifests")

    manifest = PROJECT / args.manifest if args.manifest else Path(base_settings["manifest"])
    rows = [r for r in read_jsonl(manifest) if r["split"] == "val"]
    labels = np.array([r["label"] for r in rows])
    contents = [ee.content(r) for r in rows]
    n = base_settings["num_classes"]
    conds = ee.all_conditions()

    base_preds = run_predictions(base_model, rows, contents, conds, with_mask=False)
    treat_preds = run_predictions(treat_model, rows, contents, conds, with_mask=True)

    def acc(preds, name):
        return metrics(labels.tolist(), preds[name].tolist(), n)["accuracy"]

    cells = {}
    for cond in conds:
        name = cond["name"]
        a, b = acc(base_preds, name), acc(treat_preds, name)
        agree = float(np.mean(base_preds[name] == treat_preds[name]))
        cells[name] = {"kind": cond["kind"], "severity": cond["severity"],
                       "domain": cond["domain"], "layer": cond["layer"],
                       "baseline_accuracy": a, "m1_accuracy": b,
                       "delta_pp": 100 * (b - a), "prediction_agreement": agree}

    def mean_over(pred, names):
        return float(np.mean([cells[n][pred] for n in names]))

    layers = {layer: [k for k, v in cells.items() if v["layer"] == layer]
              for layer in ("clean", "single", "combo")}
    sev5 = [k for k, v in cells.items() if v["layer"] == "combo" and v["severity"] == 5]
    sev5_matched = [k for k in sev5 if cells[k]["domain"] == "matched"]

    summary = {}
    for layer, names in layers.items():
        summary[layer] = {"n": len(names),
                          "baseline_mean_accuracy": mean_over("baseline_accuracy", names),
                          "m1_mean_accuracy": mean_over("m1_accuracy", names),
                          "delta_pp": mean_over("delta_pp", names)}
    for tag, names in (("combo_sev5_all", sev5), ("combo_sev5_matched", sev5_matched)):
        summary[tag] = {"n": len(names), "conditions": names,
                        "baseline_mean_accuracy": mean_over("baseline_accuracy", names),
                        "m1_mean_accuracy": mean_over("m1_accuracy", names),
                        "delta_pp": mean_over("delta_pp", names)}

    out = {"baseline_run": args.baseline, "treatment_run": args.treatment,
           "manifest": str(manifest), "n_val": len(rows), "seed": treat_settings.get("seed"),
           "epochs": treat_settings.get("epochs"),
           "cells": cells, "summary": summary}
    write_json(PROJECT / args.out, out)

    if args.dump_preds:
        np.savez_compressed(
            PROJECT / args.dump_preds,
            labels=labels, conditions=np.array([c["name"] for c in conds]),
            baseline=np.stack([base_preds[c["name"]] for c in conds]),
            m1=np.stack([treat_preds[c["name"]] for c in conds]))

    print(f"{'condition':22s} {'base':>7s} {'m1':>7s} {'delta':>7s}")
    for name in sorted(cells):
        c = cells[name]
        print(f"{name:22s} {100*c['baseline_accuracy']:7.1f} {100*c['m1_accuracy']:7.1f} "
              f"{c['delta_pp']:+7.1f}")
    print()
    for key, s in summary.items():
        print(f"{key:20s} base {100*s['baseline_mean_accuracy']:6.2f}  "
              f"m1 {100*s['m1_mean_accuracy']:6.2f}  delta {s['delta_pp']:+6.2f} pp")


if __name__ == "__main__":
    main()
