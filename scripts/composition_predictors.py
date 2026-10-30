#!/usr/bin/env python
"""Two mechanism-based predictors of the pairwise interaction.

The mean-shift cosine fails because it compares *average* displacement while
the interaction is about *per-image* redundancy. Two corruptions are redundant
when they move the same images in the same direction, even if their mean shifts
point elsewhere. This script tests two single-corruption-only predictors:

1. ``alignment`` — mean over images of cos(delta_i^a, delta_i^b), where
   ``delta_i^f = phi(x_i + f) - phi(x_i)``. High alignment means the second
   corruption largely repeats the first: expect sub-additivity.

2. ``hf_product`` — the product of the two families' retained high-frequency
   energy, measured in image space with no model at all. Low-pass families
   destroy the same frequency band, so two of them are redundant.

3. ``class_error_corr`` — Pearson correlation of the two families' per-class
   error-rate vectors. Two corruptions that break the same classes are
   destroying the same discriminative structure: expect sub-additivity.

4. ``overlap_pp`` — the size of the intersection of the two per-image error
   sets. Also single-corruption-only, since the compound is not involved.

All predictors use single corruptions only; the compound is never measured for
them. Ground truth ``delta_pp`` comes from ``composition_interaction.py``.

    python scripts/composition_predictors.py --fold 0 --seed 17 --target 20
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
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT, ROOT / "scripts"):
    sys.path.insert(0, str(path))

import explore_eval as ee  # noqa: E402
from maritime_robust.common import ROOT as PROJECT, read_jsonl, write_json  # noqa: E402
from maritime_robust.corruptions_extra import FAMILIES, apply  # noqa: E402

MEAN, STD, FILL = [.485, .456, .406], [.229, .224, .225], (124, 116, 104)

# Coarse mechanism labels, fixed before looking at the interaction numbers:
# pixel-level stochastic corruption vs. spatial low-pass damage.
MECHANISM = {"noise": "stochastic", "rain": "stochastic", "impulse": "stochastic",
             "defocus": "lowpass", "motion": "lowpass", "pixelate": "lowpass"}


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


def hf_energy(image):
    a = np.asarray(image.convert("RGB"), np.float32) / 255
    low = np.stack([gaussian_filter(a[..., c], 1.5) for c in range(3)], -1)
    return float(np.mean((a - low) ** 2))


def _group(record):
    ma, mb = MECHANISM[record["a"]], MECHANISM[record["b"]]
    return f"{ma}+{ma}" if ma == mb else "mixed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--target", type=float, default=20.0)
    ap.add_argument("--out-dir", default="reports/composition")
    args = ap.parse_args()

    pairs = json.loads((PROJECT / args.out_dir /
                        f"pairs_fold{args.fold}_s{args.seed}.json").read_text())
    inter = json.loads((PROJECT / args.out_dir /
                        f"interaction_fold{args.fold}_s{args.seed}_t{args.target:g}.json").read_text())
    delta = {r["pair"]: r["delta_pp"] for r in inter["pairs"]}

    run = PROJECT / pairs["run"]
    model, settings = ee.load_any(run)
    rows = [r for r in read_jsonl(Path(settings["manifest"])) if r["split"] == "val"]
    contents = [ee.content(r) for r in rows]
    labels = np.array([r["label"] for r in rows])
    n_class = int(labels.max()) + 1

    levels = {k.split("@")[0]: v for k, v in pairs["levels"].items()
              if float(k.split("@")[1]) == args.target}
    usable = [f for f in FAMILIES if f in levels]

    clean_feats, clean_preds = features_and_preds(
        model, torch.stack([pad_norm(im) for im in contents]))
    clean_hf = np.mean([hf_energy(im) for im in contents]) + 1e-12
    disp, hf = {}, {}
    errs, per_class = {}, {}
    for family in usable:
        level = levels[family]
        degraded = [apply(im, family, level, seed=i) for i, im in enumerate(contents)]
        feats, preds = features_and_preds(model, torch.stack([pad_norm(im) for im in degraded]))
        d = feats - clean_feats
        disp[family] = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        hf[family] = float(np.mean([hf_energy(im) for im in degraded]) / clean_hf)
        errs[family] = preds != labels
        per_class[family] = np.array([errs[family][labels == c].mean()
                                      for c in range(n_class)])
        print(f"{family:9s} level {level:.3f}  hf_ratio {hf[family]:.3f}", flush=True)
    records = []
    for a, b in itertools.combinations(usable, 2):
        key = f"{a}+{b}"
        alignment = float(np.mean(np.sum(disp[a] * disp[b], axis=1)))
        same = MECHANISM[a] == MECHANISM[b]
        overlap = float(np.mean(errs[a] & errs[b])) * 100
        sa, sb = np.std(per_class[a]), np.std(per_class[b])
        cec = (float(np.corrcoef(per_class[a], per_class[b])[0, 1])
               if sa > 0 and sb > 0 else 0.0)
        records.append({"pair": key, "a": a, "b": b, "alignment": alignment,
                        "overlap_pp": overlap, "class_error_corr": cec,
                        "hf_a": hf[a], "hf_b": hf[b],
                        "hf_product": hf[a] * hf[b],
                        "same_mechanism": same,
                        "delta_pp": delta.get(key)})
    records = [r for r in records if r["delta_pp"] is not None]

    al = np.array([r["alignment"] for r in records])
    hp = np.array([r["hf_product"] for r in records])
    ov = np.array([r["overlap_pp"] for r in records])
    cec = np.array([r["class_error_corr"] for r in records])
    sm = np.array([float(r["same_mechanism"]) for r in records])
    dl = np.array([r["delta_pp"] for r in records])

    stats = {}
    for name, values in (("alignment", al), ("hf_product", hp), ("overlap", ov),
                         ("class_error_corr", cec), ("same_mechanism", sm)):
        rho, p = spearmanr(values, dl)
        stats[f"{name}_delta"] = [float(rho), float(p)]

    print()
    print(f"{'pair':22s} {'align':>7s} {'overlap':>8s} {'class_r':>8s} "
          f"{'hf_prod':>8s} {'delta':>7s}")
    for r in sorted(records, key=lambda r: -r["delta_pp"]):
        print(f"{r['pair']:22s} {r['alignment']:+7.3f} {r['overlap_pp']:8.1f} "
              f"{r['class_error_corr']:+8.3f} {r['hf_product']:8.3f} {r['delta_pp']:+7.1f}")
    print()
    for name in ("alignment", "hf_product", "overlap", "class_error_corr",
                 "same_mechanism"):
        rho, p = stats[f"{name}_delta"]
        print(f"Spearman({name:16s}, delta) = {rho:+.3f} (p={p:.3f})")
    print()
    for mechanism, idx in (("lowpass+lowpass", 0), ("stochastic+stochastic", 1),
                           ("mixed", 2)):
        mask = np.array([_group(r) == mechanism for r in records])
        if mask.sum() == 0:
            continue
        vals = dl[mask]
        print(f"  {mechanism:22s} n={len(vals):2d}  mean delta {vals.mean():+6.1f} pp  "
              f"range [{vals.min():+.1f}, {vals.max():+.1f}]")

    write_json(PROJECT / args.out_dir / f"predictors_fold{args.fold}_s{args.seed}_t{args.target:g}.json",
               {"run": pairs["run"], "fold": args.fold, "seed": args.seed,
                "target_pp": args.target, "hf_ratio": hf,
                "pairs": records, "spearman": stats})


if __name__ == "__main__":
    main()
