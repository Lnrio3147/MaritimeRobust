#!/usr/bin/env python
"""Test-time error detection under compound degradation.

Pre-registration: reports/PREREG_SELECTIVE_PROBE.md. No training and no
fine-tuning: one frozen baseline per (fold, seed), clean class centroids from
that fold's *train* split, then per-condition AUROC for five scores.

    python scripts/selective_probe.py --fold 0 --seed 17

``drift_nearest`` is the candidate: the cosine distance from a test feature to
the nearest clean class centroid. It needs no clean version of the test image,
only statistics collected at training time. ``drift_pred`` is the circularity
control: it uses the model's own prediction to pick the reference centroid.
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
SCORES = ("ent", "unc", "margin", "drift_nearest", "drift_pred")


def pooled_and_logits(model, x):
    """Pooled descriptor and logits in one pass (``MethodModel``, method none)."""
    with torch.autocast("cuda", dtype=torch.float16):
        f = model.features_of(x)
        logits = model.head(f)
    return f.float(), logits.float()


def auroc(scores, errors):
    """Mann-Whitney AUROC; ``errors`` are the positives. None if degenerate."""
    errors = np.asarray(errors, dtype=bool)
    n_pos, n_neg = int(errors.sum()), int((~errors).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties so the statistic stays exact
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    for k, count in enumerate(counts):
        if count > 1:
            ranks[inverse == k] = ranks[inverse == k].mean()
    return float((ranks[errors].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out-dir", default="reports/selective")
    args = ap.parse_args()

    manifest = PROJECT / f"data/manifests/folds/fgsc23_civ8_fold{args.fold}.jsonl"
    run = PROJECT / f"runs/fgsc23_civ8_fold{args.fold}/convnext_tiny_none_e40_s{args.seed}"
    model, settings = ee.load_any(run)
    model = model.cuda().eval()
    rows = read_jsonl(manifest)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    n = settings["num_classes"]
    labels = np.array([r["label"] for r in val_rows])

    # clean centroids from the train split only
    train_tensor = torch.stack([ee.pad_norm(ee.content(r)) for r in train_rows])
    feats = []
    with torch.inference_mode():
        for start in range(0, len(train_tensor), args.batch):
            feats.append(pooled_and_logits(model, train_tensor[start:start + args.batch].cuda())[0]
                         .cpu().numpy())
    train_features = np.concatenate(feats)
    train_labels = np.array([r["label"] for r in train_rows])
    train_features /= np.linalg.norm(train_features, axis=1, keepdims=True)
    centroids = np.stack([train_features[train_labels == k].mean(0) for k in range(n)])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids_t = torch.tensor(centroids, device="cuda", dtype=torch.float32)

    contents = [ee.content(r) for r in val_rows]
    cells = {}
    for cond in m2.conditions_47():
        tensor = me.batch_of(val_rows, contents, cond, with_mask=False)
        f_all, logits_all = [], []
        with torch.inference_mode():
            for start in range(0, len(tensor), args.batch):
                f, logits = pooled_and_logits(model, tensor[start:start + args.batch].cuda())
                f_all.append(f.cpu())
                logits_all.append(logits.cpu())
        features = torch.cat(f_all).numpy()
        logits = torch.cat(logits_all)
        probs = logits.softmax(1).numpy()
        preds = probs.argmax(1)
        features /= np.linalg.norm(features, axis=1, keepdims=True)

        cos = torch.tensor(features, device="cuda") @ centroids_t.T
        nearest = cos.max(1).values.cpu().numpy()
        to_pred = cos.gather(1, torch.tensor(preds, device="cuda").unsqueeze(1)).squeeze(1)
        to_pred = to_pred.cpu().numpy()

        top2 = np.sort(probs, 1)[:, -2:]
        errors = preds != labels
        scores = {
            "ent": -np.sum(probs * np.log(np.clip(probs, 1e-12, 1)), 1) / np.log(n),
            "unc": 1.0 - probs.max(1),
            "margin": -(top2[:, 1] - top2[:, 0]),
            "drift_nearest": 1.0 - nearest,
            "drift_pred": 1.0 - to_pred,
        }
        cells[cond["name"]] = {
            "kind": cond["kind"], "severity": cond["severity"], "domain": cond["domain"],
            "layer": cond["layer"], "accuracy": float((~errors).mean()),
            "error_rate": float(errors.mean()),
            "auroc": {name: auroc(scores[name], errors) for name in SCORES},
        }
        print(f"{cond['name']:24s} acc {100*(~errors).mean():6.2f} "
              + " ".join(f"{k[:12]}={cells[cond['name']]['auroc'][k]:.3f}"
                         for k in SCORES), flush=True)

    def group(names):
        chosen = [c for c in names if c in cells]
        return {"n": len(chosen), "conditions": sorted(chosen),
                "auc": {s: float(np.mean([cells[c]["auroc"][s] for c in chosen
                                          if cells[c]["auroc"][s] is not None]))
                        for s in SCORES},
                "mean_error_rate": float(np.mean([cells[c]["error_rate"] for c in chosen]))}

    combos = [k for k, v in cells.items() if v["layer"] == "combo"]
    summary = {
        "combo_s5": group([k for k in combos if cells[k]["severity"] == 5]),
        "combo_low": group([k for k in combos if cells[k]["severity"] <= 4]),
        "single": group([k for k, v in cells.items() if v["layer"] == "single"]),
        "clean": group(["clean"]),
    }
    out = {"run": str(run.relative_to(PROJECT)), "fold": args.fold, "seed": args.seed,
           "manifest": str(manifest.relative_to(PROJECT)), "n_val": len(val_rows),
           "n_train": len(train_rows), "cells": cells, "summary": summary}
    out_dir = PROJECT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"fold{args.fold}_s{args.seed}.json", out)
    print()
    for key, s in summary.items():
        print(f"{key:12s} n={s['n']:2d} err {100*s['mean_error_rate']:5.1f}% "
              + " ".join(f"{k}={v:.3f}" for k, v in s["auc"].items()))


if __name__ == "__main__":
    main()
