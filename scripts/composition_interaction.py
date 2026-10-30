#!/usr/bin/env python
"""Decompose the pairwise interaction into error overlap and true interaction.

``E = drop(a+b) - drop(a) - drop(b)`` conflates two different things:

* **overlap** — the images that fail under ``a`` already fail under ``b``. Two
  corruptions that break the same images cannot add up; ``E`` goes negative for
  purely mechanical reasons, with no interaction at all.
* **interaction** — the compound fails images that neither single corruption
  fails on its own.

The independent-failure baseline is the union of the two error sets,

    e_indep = |E_a u E_b| / N
            = e_a + e_b - |E_a n E_b| / N

and the interaction proper is the excess error rate over that baseline,

    delta = e(a+b) - e_indep

Both terms must be in error-rate space. ``E`` is in drop-from-clean space, and
``delta = E - e_clean + overlap``, so the two differ by more than a constant.

This script measures both from per-image predictions and additionally tests
whether ``E_union`` decomposes additively over families,

    E_union(a,b) ~ m_a + m_b

If that model fits, the interaction is a property of each corruption channel,
not of the pair, and any unseen pair can be predicted from per-family scores.

    python scripts/composition_interaction.py --fold 0 --seed 17 --target 20
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.stats import spearmanr
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT, ROOT / "scripts"):
    sys.path.insert(0, str(path))

import explore_eval as ee  # noqa: E402
from maritime_robust.common import ROOT as PROJECT, read_jsonl, write_json  # noqa: E402
from maritime_robust.corruptions_extra import FAMILIES, apply  # noqa: E402

MEAN, STD, FILL = [.485, .456, .406], [.229, .224, .225], (124, 116, 104)


def pad_norm(image):
    padded = ImageOps.pad(image, (224, 224), Image.Resampling.BICUBIC, color=FILL)
    return TF.normalize(TF.to_tensor(padded), MEAN, STD)


def features_and_preds(model, tensor, batch=64):
    feats, logits = [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for start in range(0, len(tensor), batch):
            f = model.features_of(tensor[start:start + batch].cuda()).float()
            feats.append(f.cpu())
            logits.append(model.head(f).float().cpu())
    return torch.cat(feats).numpy(), torch.cat(logits).argmax(1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--target", type=float, default=20.0)
    ap.add_argument("--out-dir", default="reports/composition")
    args = ap.parse_args()

    pairs = json.loads((PROJECT / args.out_dir /
                        f"pairs_fold{args.fold}_s{args.seed}.json").read_text())
    run = PROJECT / pairs["run"]
    model, settings = ee.load_any(run)
    rows = [r for r in read_jsonl(Path(settings["manifest"])) if r["split"] == "val"]
    labels = np.array([r["label"] for r in rows])
    contents = [ee.content(r) for r in rows]
    n_img = len(rows)

    levels = {k.split("@")[0]: v for k, v in pairs["levels"].items()
              if float(k.split("@")[1]) == args.target}
    usable = [f for f in FAMILIES if f in levels]

    def run_cond(fn):
        tensor = torch.stack([pad_norm(fn(i, im)) for i, im in enumerate(contents)])
        feats, preds = features_and_preds(model, tensor)
        return feats, preds, float(np.mean(preds == labels))

    clean_feats, clean_preds, clean_acc = run_cond(lambda i, im: im)
    errors = {"clean": clean_preds != labels}
    singles, dirs = {}, {}
    for family in usable:
        level = levels[family]
        feats, preds, acc = run_cond(
            lambda i, im, f=family, l=level: apply(im, f, l, seed=i))
        errors[family] = preds != labels
        singles[family] = acc
        u = (feats - clean_feats).mean(0)
        dirs[family] = u / max(np.linalg.norm(u), 1e-9)
        print(f"{family:9s} level {level:.3f}  acc {100*acc:6.2f}", flush=True)

    records = []
    for a, b in itertools.combinations(usable, 2):
        la, lb = levels[a], levels[b]
        _, preds, acc = run_cond(lambda i, im, A=a, B=b, LA=la, LB=lb: apply(
            apply(im, A, LA, seed=i), B, LB, seed=i + 1))
        err_ab = preds != labels
        err_a, err_b = errors[a], errors[b]
        emp = float(np.mean(err_a & err_b)) * 100          # both fail
        union = float(np.mean(err_a | err_b)) * 100        # either fails
        drop_ab = 100 * (clean_acc - acc)
        drop_a = 100 * (clean_acc - singles[a])
        drop_b = 100 * (clean_acc - singles[b])
        e_ab = 100 * (1 - acc)
        e_indep = union                                     # = e_a + e_b - overlap
        records.append({
            "pair": f"{a}+{b}", "a": a, "b": b,
            "drop_a_pp": drop_a, "drop_b_pp": drop_b, "drop_pair_pp": drop_ab,
            "error_a_pp": 100 * float(np.mean(err_a)),
            "error_b_pp": 100 * float(np.mean(err_b)),
            "error_pair_pp": e_ab, "error_indep_pp": e_indep,
            "overlap_pp": emp,
            "E_pp": drop_ab - (drop_a + drop_b),
            "delta_pp": e_ab - e_indep,
            "cos_shift": float(dirs[a] @ dirs[b])})

    E = np.array([r["E_pp"] for r in records])
    EU = np.array([r["delta_pp"] for r in records])
    cos = np.array([r["cos_shift"] for r in records])
    ov = np.array([r["overlap_pp"] for r in records])

    print()
    print(f"{'pair':22s} {'err_ab':>7s} {'indep':>7s} {'overlap':>8s} "
          f"{'delta':>7s} {'E':>7s} {'cos':>7s}")
    for r in sorted(records, key=lambda r: -r["delta_pp"]):
        print(f"{r['pair']:22s} {r['error_pair_pp']:7.1f} {r['error_indep_pp']:7.1f} "
              f"{r['overlap_pp']:8.1f} {r['delta_pp']:+7.1f} {r['E_pp']:+7.1f} "
              f"{r['cos_shift']:+7.3f}")

    # Additive main-effect fit: E ~ m_a + m_b (constraint: main effects mean zero).
    def fit_additive(values):
        size = len(usable)
        X = np.zeros((len(records), size + 1))
        X[:, 0] = 1.0
        for row, rec in enumerate(records):
            X[row, 1 + usable.index(rec["a"])] += 1
            X[row, 1 + usable.index(rec["b"])] += 1
        coef, *_ = np.linalg.lstsq(X, values, rcond=None)
        pred = X @ coef
        ss_res = float(np.sum((values - pred) ** 2))
        ss_tot = float(np.sum((values - values.mean()) ** 2))
        return 1 - ss_res / max(ss_tot, 1e-12), coef, pred

    r2_e, coef_e, _ = fit_additive(E)
    r2_u, coef_u, pred_u = fit_additive(EU)
    main_u = {f: float(coef_u[1 + usable.index(f)]) for f in usable}

    rho_cos_eu, p_cos_eu = spearmanr(cos, EU)
    rho_cos_e, p_cos_e = spearmanr(cos, E)
    rho_ov, p_ov = spearmanr(ov, E)
    rho_ovd, p_ovd = spearmanr(ov, EU)

    print()
    print(f"Spearman(cos, E)        = {rho_cos_e:+.3f} (p={p_cos_e:.3f})")
    print(f"Spearman(cos, delta)    = {rho_cos_eu:+.3f} (p={p_cos_eu:.3f})")
    print(f"Spearman(overlap, E)    = {rho_ov:+.3f} (p={p_ov:.3f})")
    print(f"Spearman(overlap, delta)= {rho_ovd:+.3f} (p={p_ovd:.3f})")
    print(f"additive R^2 on E       = {r2_e:.3f}")
    print(f"additive R^2 on delta   = {r2_u:.3f}")
    print(f"mean delta = {EU.mean():+.1f} pp, mean E = {E.mean():+.1f} pp")
    print(f"main effects on delta: " + "  ".join(
        f"{f} {v:+.1f}" for f, v in sorted(main_u.items(), key=lambda kv: -kv[1])))

    write_json(PROJECT / args.out_dir / f"interaction_fold{args.fold}_s{args.seed}_t{args.target:g}.json",
               {"run": pairs["run"], "fold": args.fold, "seed": args.seed,
                "target_pp": args.target, "levels": levels, "n_val": n_img,
                "clean_accuracy": clean_acc,
                "single_accuracy": singles, "pairs": records,
                "spearman": {"cos_E": [float(rho_cos_e), float(p_cos_e)],
                             "cos_delta": [float(rho_cos_eu), float(p_cos_eu)],
                             "overlap_E": [float(rho_ov), float(p_ov)],
                             "overlap_delta": [float(rho_ovd), float(p_ovd)]},
                "additive_r2": {"E": float(r2_e), "delta": float(r2_u)},
                "main_effects_delta": main_u})


if __name__ == "__main__":
    main()
