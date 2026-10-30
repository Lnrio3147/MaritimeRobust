#!/usr/bin/env python
"""D2 probe: is the superadditive loss already in the representation?

For one baseline run, measures the cosine drift of the final descriptor
(`MethodModel.features_of`, 768-d) between the clean image and every one of
the 43 conditions, then asks whether compound drift exceeds the sum of the
two marginal drifts (the same decomposition the accuracy analysis uses).

    python scripts/feature_drift_probe.py --run runs/fgsc23_civ8_fold0/convnext_tiny_none_e40_s17

Writes reports/feature_drift_s{seed}.json plus per-image drifts in
reports/feature_drift_s{seed}.npz. Pre-registration and decision rule:
reports/PREREG_FEATURE_DRIFT.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import explore_eval as ee  # noqa: E402
import m1_eval as me  # noqa: E402
from maritime_robust.common import read_jsonl, write_json  # noqa: E402
from maritime_robust.engine import metrics  # noqa: E402

PAIRS = {"haze+glare": ("haze", "glare"), "haze+cloud": ("haze", "cloud")}
# degradations.conditions() only registers compound severities 3 and 5; the
# pre-registered 2/4 points are built the same way in severity_curve_eval.py.
EXTRA = [{"name": f"matched_{kind}_{sev}", "kind": kind, "severity": sev,
          "domain": "matched", "placement": "random", "layer": "combo"}
         for kind in PAIRS for sev in (2, 4)]


def features(model, rows, contents, conds, batch=64):
    out = {}
    for cond in conds:
        tensor = me.batch_of(rows, contents, cond, with_mask=False)
        got = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for start in range(0, len(tensor), batch):
                got.append(model.features_of(tensor[start:start + batch].cuda()).float().cpu())
        out[cond["name"]] = torch.cat(got).numpy()
    return out


def drift(clean, feats):
    a = clean / np.linalg.norm(clean, axis=1, keepdims=True)
    b = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    return 1.0 - (a * b).sum(1)


def cluster_ci(values_by_group, reps=2000, seed=0):
    """Bootstrap the mean over source_group_id clusters (one image per group)."""
    rng = np.random.default_rng(seed)
    groups = list(values_by_group)
    means = []
    for _ in range(reps):
        pick = rng.choice(len(groups), len(groups), replace=True)
        vals = np.concatenate([values_by_group[groups[i]] for i in pick])
        means.append(vals.mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    run = ROOT / args.run
    model, settings = ee.load_any(run)
    seed = settings["seed"]
    manifest = ROOT / args.manifest if args.manifest else Path(settings["manifest"])
    rows = [r for r in read_jsonl(manifest) if r["split"] == "val"]
    labels = np.array([r["label"] for r in rows])
    groups = np.array([r["source_group_id"] for r in rows])
    contents = [ee.content(r) for r in rows]
    n = settings["num_classes"]

    conds = ee.all_conditions() + EXTRA
    clean_cond = next(c for c in conds if c["kind"] == "clean")
    rest = [c for c in conds if c["kind"] != "clean"]

    clean_feat = features(model, rows, contents, [clean_cond])[clean_cond["name"]]
    feats = features(model, rows, contents, rest)
    per_image = {name: drift(clean_feat, f) for name, f in feats.items()}

    # Accuracy per condition, same predictions the drift was measured on.
    preds = me.run_predictions(model, rows, contents, conds, with_mask=False)
    acc = {c["name"]: metrics(labels.tolist(), preds[c["name"]].tolist(), n)["accuracy"]
           for c in conds}

    cells = {}
    for name, d in per_image.items():
        cells[name] = {"kind": next(c["kind"] for c in conds if c["name"] == name), "drift": float(d.mean()),
                       "ci": cluster_ci({g: d[groups == g] for g in set(groups)}),
                       "accuracy": acc[name]}

    def e_feat(pair, s):
        a, b = PAIRS[pair]
        comp = f"matched_{pair}_{s}"
        da, db = f"matched_{a}_{s}", f"matched_{b}_{s}"
        e_im = per_image[comp] - (per_image[da] + per_image[db])
        return {"value": float(e_im.mean()),
                "ci": cluster_ci({g: e_im[groups == g] for g in set(groups)}),
                "per_image": e_im}

    def e_acc(pair, s):
        a, b = PAIRS[pair]
        c0 = acc["clean"]
        return 100 * ((c0 - acc[f"matched_{pair}_{s}"])
                      - (c0 - acc[f"matched_{a}_{s}"]) - (c0 - acc[f"matched_{b}_{s}"]))

    curve = {pair: {str(s): {**{k: v for k, v in e_feat(pair, s).items() if k != "per_image"},
                             "accuracy_pp": e_acc(pair, s)}
                    for s in (2, 3, 4, 5)}
             for pair in PAIRS}

    # P4: does drift separate the images no single degradation gets wrong?
    comp = "matched_haze+glare_5"
    wrong_any_single = np.zeros(len(rows), bool)
    for name, p in preds.items():
        if name.startswith("matched_") and "+" not in name and name != "matched_clean":
            wrong_any_single |= (p != labels)
    new_err = (preds[comp] != labels) & ~wrong_any_single
    d = per_image[comp]
    auc = float(np.mean(d[new_err][:, None] > d[~new_err][None, :])) if new_err.any() else float("nan")

    # P3 is about the *excesses*, not the raw drifts: compare E_feat against
    # E_acc over the eight (pair, severity) points of the curve.
    pts = [v for pair in curve.values() for v in pair.values()]
    rho = spearmanr([p["value"] for p in pts], [p["accuracy_pp"] for p in pts])
    rho_raw = spearmanr([cells[n_]["drift"] for n_ in feats], [acc[n_] for n_ in feats])

    out = {"run": args.run, "seed": seed, "n_val": len(rows),
           "clean_accuracy": acc["clean"],
           "curve": curve,
           "p4": {"n_new_errors": int(new_err.sum()),
                  "drift_mean_new_errors": float(d[new_err].mean()) if new_err.any() else None,
                  "drift_mean_rest": float(d[~new_err].mean()),
                  "auc": auc},
           "p3_spearman_e_feat_vs_e_acc": {"rho": float(rho.statistic), "p": float(rho.pvalue),
                                           "n_points": len(pts)},
           "raw_spearman_drift_vs_accuracy": {"rho": float(rho_raw.statistic),
                                              "p": float(rho_raw.pvalue)},
           "cells": cells}
    out_path = ROOT / (args.out or f"reports/feature_drift_s{seed}.json")
    write_json(out_path, out)
    if args.dump:
        np.savez_compressed(ROOT / args.dump, labels=labels, groups=groups,
                            new_error=new_err,
                            **{f"d_{k.replace('+', 'plus')}": v for k, v in per_image.items()})

    print(f"seed {seed}  clean acc {100*acc['clean']:.2f}")
    for pair, pts in curve.items():
        print(f"  {pair:11s} " + "  ".join(
            f"s{s}: E_feat {v['value']:+.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] "
            f"E_acc {v['accuracy_pp']:+.1f}pp" for s, v in pts.items()))
    print(f"  P4 new errors {out['p4']['n_new_errors']}  drift {out['p4']['drift_mean_new_errors']:.4f}"
          f" vs rest {out['p4']['drift_mean_rest']:.4f}  AUC {auc:.3f}")
    print(f"  P3 Spearman(E_feat, E_acc) over {len(pts)} points rho={rho.statistic:+.3f} "
          f"p={rho.pvalue:.3g}; raw drift-vs-acc rho={rho_raw.statistic:+.3f}")
    print(f"wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
