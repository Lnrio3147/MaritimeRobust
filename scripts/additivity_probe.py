"""Does the baseline's feature space obey the additive composition law?

The interaction analysis found that accuracy-space interaction decomposes
additively into per-family main effects. If that additive structure also holds
at the level of the penultimate feature, then a loss that *enforces* it has
nothing left to fix and the route is closed before any training is spent.

For a frozen checkpoint this measures, on the validation split,

    r = phi(a+b) - phi(a) - phi(b) + phi(clean)

normalised by the size of the two single-corruption shifts, and asks whether
|r| predicts per-image failure under the compound.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

import explore_eval as ee  # noqa: E402
import m1_eval as me  # noqa: E402
import m2_eval as m2  # noqa: E402
from maritime_robust.common import read_jsonl, write_json  # noqa: E402


def feats(model, tensor):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        f = model.features_of(tensor.cuda())
    return f.float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--model", default="convnext_tiny")
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--out", default="reports/additivity_probe.json")
    args = ap.parse_args()

    run = PROJECT / f"runs/fgsc23_civ8_fold{args.fold}/{args.model}_none_e40_s{args.seed}"
    model, settings = ee.load_any(run)
    rows = [r for r in read_jsonl(Path(settings["manifest"])) if r["split"] == "val"]
    labels = np.array([r["label"] for r in rows])
    contents = [ee.content(r) for r in rows]
    conds = {c["name"]: c for c in m2.conditions_47()}

    def get(name):
        return me.batch_of(rows, contents, conds[name], with_mask=False)

    f_clean = feats(model, get("clean"))
    f_haze = feats(model, get(f"matched_haze_{args.severity}"))
    f_glare = feats(model, get(f"matched_glare_{args.severity}"))
    f_hg = feats(model, get(f"matched_haze+glare_{args.severity}"))
    f_cloud = feats(model, get(f"matched_cloud_{args.severity}"))
    f_hc = feats(model, get(f"matched_haze+cloud_{args.severity}"))

    # Predictions, so the residual can be related to actual failure.
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        pred = model(get(f"matched_haze+glare_{args.severity}").cuda()).float().argmax(1).cpu().numpy()
    wrong = (pred != labels).astype(float)

    out = {"run": str(run.relative_to(PROJECT)), "fold": args.fold, "seed": args.seed,
           "model": args.model, "severity": args.severity, "accuracy": float(np.mean(pred == labels)),
           "pairs": {}}

    for name, fa, fb, fab in (("haze+glare", f_haze, f_glare, f_hg),
                              ("haze+cloud", f_haze, f_cloud, f_hc)):
        da, db = fa - f_clean, fb - f_clean
        r = fab - fa - fb + f_clean
        # Scale-free residual: how big the interaction is next to the two shifts
        # it is supposed to be explained by.
        scale = np.linalg.norm(da, axis=1) + np.linalg.norm(db, axis=1)
        rel = np.linalg.norm(r, axis=1) / np.maximum(scale, 1e-8)
        rho, p = spearmanr(rel, wrong)
        out["pairs"][name] = {
            "rel_residual_mean": float(rel.mean()),
            "rel_residual_median": float(np.median(rel)),
            "rel_residual_p90": float(np.percentile(rel, 90)),
            "rel_residual_wrong": float(rel[wrong > 0].mean()) if wrong.any() else None,
            "rel_residual_right": float(rel[wrong == 0].mean()) if (wrong == 0).any() else None,
            "spearman_residual_vs_error": float(rho), "p_value": float(p),
            "n_wrong": int(wrong.sum()),
        }
        print(f"{name}  rel|r| mean {rel.mean():.3f}  median {np.median(rel):.3f}  "
              f"wrong {out['pairs'][name]['rel_residual_wrong']:.3f} vs "
              f"right {out['pairs'][name]['rel_residual_right']:.3f}  "
              f"rho(|r|, err) {rho:+.3f} (p={p:.3g})", flush=True)

    write_json(PROJECT / args.out, out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
