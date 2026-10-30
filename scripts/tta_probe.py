#!/usr/bin/env python
"""Test-time feature recentring probe (reports/PREREG_TTA_RECENTERING.md).

Frozen checkpoint, no training. For each condition the whole validation batch is
run once and four readouts are compared on the same features: identity, a
logit-mean bias correction, feature recentering onto the clean train mean, and a
full first/second-moment transfer. The test batch supplies only unlabelled
statistics.

    python scripts/tta_probe.py --fold 0 --seed 17
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT, ROOT / "scripts"):
    sys.path.insert(0, str(path))

import explore_eval as ee  # noqa: E402
import m1_eval as me  # noqa: E402
import m2_eval as m2  # noqa: E402
from maritime_robust.common import ROOT as PROJECT, read_jsonl, write_json  # noqa: E402
from maritime_robust.engine import metrics  # noqa: E402

VARIANTS = ("C0", "C1", "R1", "R2")


def features_and_logits(model, tensor, batch=64):
    feats, logits = [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for start in range(0, len(tensor), batch):
            f = model.features_of(tensor[start:start + batch].cuda()).float()
            feats.append(f.cpu())
            logits.append(model.head(f).float().cpu())
    return torch.cat(feats).numpy(), torch.cat(logits).numpy()


def clean_train_stats(model, train_rows):
    contents = [ee.content(r) for r in train_rows]
    cond = {"name": "clean", "kind": "clean", "severity": 0, "domain": "matched",
            "placement": "random"}
    tensor = me.batch_of(train_rows, contents, cond, with_mask=False)
    feats, _ = features_and_logits(model, tensor)
    return feats.mean(0), np.maximum(feats.std(0), 1e-3)


def readouts(feats, logits, mu_tr, sigma_tr):
    mu_b = feats.mean(0)
    sigma_b = np.maximum(feats.std(0), 1e-3)
    base = {"C0": logits.argmax(1),
            "C1": (logits - logits.mean(0, keepdims=True)).argmax(1)}
    return base, feats - mu_b + mu_tr, (feats - mu_b) / sigma_b * sigma_tr + mu_tr


def head_logits(model, feats):
    with torch.inference_mode():
        return model.head(torch.from_numpy(feats).cuda().float()).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--run", default=None)
    ap.add_argument("--out-dir", default="reports/tta")
    ap.add_argument("--resamples", type=int, default=2000)
    args = ap.parse_args()

    run = (PROJECT / args.run if args.run else
           PROJECT / f"runs/fgsc23_civ8_fold{args.fold}/convnext_tiny_none_e40_s{args.seed}")
    model, settings = ee.load_any(run)
    manifest = Path(settings["manifest"])
    rows_all = read_jsonl(manifest)
    rows = [r for r in rows_all if r["split"] == "val"]
    train_rows = [r for r in rows_all if r["split"] == "train"]
    labels = np.array([r["label"] for r in rows])
    groups = np.array([r["source_group_id"] for r in rows])
    contents = [ee.content(r) for r in rows]
    n = settings["num_classes"]
    conds = m2.conditions_47()

    mu_tr, sigma_tr = clean_train_stats(model, train_rows)
    print(f"train stats: {mu_tr.shape} dims, norm {np.linalg.norm(mu_tr):.2f}", flush=True)

    preds = {v: {} for v in VARIANTS}
    for cond in conds:
        tensor = me.batch_of(rows, contents, cond, with_mask=False)
        feats, logits = features_and_logits(model, tensor)
        base, recentred, standardized = readouts(feats, logits, mu_tr, sigma_tr)
        preds["C0"][cond["name"]] = base["C0"]
        preds["C1"][cond["name"]] = base["C1"]
        preds["R1"][cond["name"]] = head_logits(model, recentred).argmax(1)
        preds["R2"][cond["name"]] = head_logits(model, standardized).argmax(1)
        print(f"{cond['name']:26s} C0 {100*metrics(labels.tolist(), preds['C0'][cond['name']].tolist(), n)['accuracy']:6.2f} "
              f"R1 {100*metrics(labels.tolist(), preds['R1'][cond['name']].tolist(), n)['accuracy']:6.2f}",
              flush=True)

    base_pred = preds["C0"]
    cells = {}
    for cond in conds:
        name = cond["name"]
        base_acc = metrics(labels.tolist(), base_pred[name].tolist(), n)["accuracy"]
        cells[name] = {"kind": cond["kind"], "severity": cond["severity"],
                       "domain": cond["domain"], "layer": cond["layer"],
                       "baseline_accuracy": base_acc}
        for v in VARIANTS[1:]:
            cells[name][f"{v}_accuracy"] = metrics(labels.tolist(), preds[v][name].tolist(), n)["accuracy"]
            cells[name][f"{v}_delta_pp"] = 100 * (cells[name][f"{v}_accuracy"] - base_acc)

    def group(names):
        sel = [x for x in names if x in cells]
        out = {"n": len(sel), "conditions": sorted(sel)}
        for v in VARIANTS[1:]:
            out[v] = {"mean_delta_pp": float(np.mean([cells[x][f"{v}_delta_pp"] for x in sel])),
                      "bootstrap": m2.cluster_bootstrap(labels, base_pred, preds[v], groups,
                                                        sel, resamples=args.resamples)}
        return out

    combos = [k for k, v in cells.items() if v["layer"] == "combo"]
    combo_low = [k for k in combos if cells[k]["severity"] <= 4 and cells[k]["domain"] == "matched"]
    summary = {
        "combo_sev5_matched": group([k for k in combos if cells[k]["severity"] == 5
                                     and cells[k]["domain"] == "matched"]),
        "combo_low": group(combo_low),
        "combo_all": group(combos),
        "single": group([k for k, v in cells.items() if v["layer"] == "single"]),
        "clean": group(["clean"]),
    }

    out = {"run": str(run.relative_to(PROJECT)), "fold": args.fold, "seed": args.seed,
           "n_val": len(rows), "cells": cells, "summary": summary}
    out_dir = PROJECT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"fold{args.fold}_s{args.seed}.json", out)

    print()
    for gname, g in summary.items():
        line = f"{gname:20s} n={g['n']:2d}"
        for v in VARIANTS[1:]:
            b = g[v]["bootstrap"]
            line += f"  {v} {b['mean_pp']:+6.2f} [{b['ci95_pp'][0]:+6.2f},{b['ci95_pp'][1]:+6.2f}]"
        print(line)


if __name__ == "__main__":
    main()
