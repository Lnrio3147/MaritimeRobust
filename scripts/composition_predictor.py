#!/usr/bin/env python
"""Does a family's *single-corruption* feature shift predict which pairs explode?

For each family we take its calibrated level (the level whose marginal drop
matches a target, from ``composition_pairs.py``), run it alone, and record the
mean penultimate-feature displacement from clean.

    u_f = mean_i [ phi(x_i + f) - phi(x_i) ] / || . ||

The candidate predictor of the pairwise interaction

    E(a,b) = drop(a+b) - drop(a) - drop(b)

is the cosine between the two shift directions, cos(u_a, u_b). If two
corruptions push the representation along independent directions they should
compose badly (large E); if they push along the same direction the second one is
partly redundant, and E should be small or negative.

The predictor uses **only single-corruption** features: the compound is never
measured for it. It is therefore testable and, if it holds, usable.

Two corrections are reported alongside the raw E, because E has a mechanical
floor at the accuracy ceiling:

    E_norm = E / (100 - drop_a - drop_b)

which divides out the headroom the pair has left before accuracy hits zero.

    python scripts/composition_predictor.py --fold 0 --seed 17
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


def features(model, tensor, batch=64):
    out = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for start in range(0, len(tensor), batch):
            out.append(model.features_of(tensor[start:start + batch].cuda()).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--target", type=float, default=20.0,
                    help="marginal drop (pp) at which directions are taken")
    ap.add_argument("--out-dir", default="reports/composition")
    args = ap.parse_args()

    pairs_path = PROJECT / args.out_dir / f"pairs_fold{args.fold}_s{args.seed}.json"
    pairs = json.loads(pairs_path.read_text())

    run = PROJECT / pairs["run"]
    model, settings = ee.load_any(run)
    rows = [r for r in read_jsonl(Path(settings["manifest"])) if r["split"] == "val"]
    contents = [ee.content(r) for r in rows]

    levels = {k.split("@")[0]: v for k, v in pairs["levels"].items()
              if float(k.split("@")[1]) == args.target}
    missing = [f for f in FAMILIES if f not in levels]
    if missing:
        print(f"families without a calibrated level at {args.target:g} pp: {missing}")
        return

    clean = features(model, torch.stack([pad_norm(im) for im in contents]))
    directions, shifts = {}, {}
    for family in FAMILIES:
        level = levels[family]
        tensor = torch.stack([pad_norm(apply(im, family, level, seed=i))
                              for i, im in enumerate(contents)])
        feats = features(model, tensor)
        shift = (feats - clean).mean(0)
        shifts[family] = shift
        directions[family] = shift / max(np.linalg.norm(shift), 1e-9)
        print(f"{family:9s} level {level:.3f}  ||shift|| {np.linalg.norm(shift):8.3f}",
              flush=True)

    usable = [f for f in FAMILIES if f in levels]
    cells, rows_out = pairs["cells"], []
    for a, b in itertools.combinations(usable, 2):
        key20 = f"pair_{a}+{b}_{args.target:g}"
        if key20 not in cells:
            continue
        clean_acc = cells["clean"]["accuracy"]
        drop_a = 100 * (clean_acc - cells[f"single_{a}_{args.target:g}"]["accuracy"])
        drop_b = 100 * (clean_acc - cells[f"single_{b}_{args.target:g}"]["accuracy"])
        drop_ab = 100 * (clean_acc - cells[key20]["accuracy"])
        e = drop_ab - (drop_a + drop_b)
        headroom = max(1e-9, 100.0 - drop_a - drop_b)
        rows_out.append({"pair": f"{a}+{b}", "a": a, "b": b,
                         "drop_a_pp": drop_a, "drop_b_pp": drop_b,
                         "drop_pair_pp": drop_ab, "E_pp": e,
                         "E_norm": e / headroom, "headroom_pp": headroom,
                         "cos_shift": float(directions[a] @ directions[b]),
                         "cos_mean_shift": float(shifts[a] @ shifts[b] /
                                                 (np.linalg.norm(shifts[a]) *
                                                  np.linalg.norm(shifts[b])))})

    if len(rows_out) < 5:
        print("too few pairs to correlate")
        return

    cos = np.array([r["cos_shift"] for r in rows_out])
    e = np.array([r["E_pp"] for r in rows_out])
    e_norm = np.array([r["E_norm"] for r in rows_out])
    rho_e, p_e = spearmanr(cos, e)
    rho_n, p_n = spearmanr(cos, e_norm)
    rho_floor, p_floor = spearmanr(cos, np.array([r["headroom_pp"] for r in rows_out]))

    rows_out.sort(key=lambda r: -r["E_pp"])
    print()
    print(f"{'pair':22s} {'drop_ab':>8s} {'E':>7s} {'E_norm':>8s} {'cos':>7s}")
    for r in rows_out:
        print(f"{r['pair']:22s} {r['drop_pair_pp']:8.1f} {r['E_pp']:+7.1f} "
              f"{r['E_norm']:+8.3f} {r['cos_shift']:+7.3f}")

    print()
    print(f"Spearman(cos, E)       = {rho_e:+.3f}  (p = {p_e:.3f}, n = {len(rows_out)})")
    print(f"Spearman(cos, E_norm)  = {rho_n:+.3f}  (p = {p_n:.3f})")
    print(f"Spearman(cos, headroom)= {rho_floor:+.3f}  (p = {p_floor:.3f})")

    # Leave-one-out rank check: does cos sort the superadditive pairs to the top?
    order = sorted(rows_out, key=lambda r: r["cos_shift"])
    pos = sorted(range(len(order)), key=lambda i: -order[i]["E_pp"])
    ranks = np.empty(len(order), int)
    ranks[pos] = np.arange(len(order))
    print(f"top-half by E: {[order[i]['pair'] for i in pos[:len(pos)//2]]}")

    write_json(PROJECT / args.out_dir / f"predictor_fold{args.fold}_s{args.seed}.json",
               {"run": pairs["run"], "fold": args.fold, "seed": args.seed,
                "target_pp": args.target, "levels": levels,
                "family_shift_norm": {f: float(np.linalg.norm(s))
                                      for f, s in shifts.items()},
                "pairs": rows_out,
                "spearman_cos_E": {"rho": float(rho_e), "p": float(p_e)},
                "spearman_cos_E_norm": {"rho": float(rho_n), "p": float(p_n)},
                "spearman_cos_headroom": {"rho": float(rho_floor), "p": float(p_floor)}})


if __name__ == "__main__":
    main()
