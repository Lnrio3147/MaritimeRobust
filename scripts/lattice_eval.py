"""Evaluate a coverage-trained checkpoint on the whole composition lattice.

The 47-condition grid covers haze/glare/cloud only, so it cannot score the
held-out synthetic pairs that the coverage study actually generalises to. This
script scores every single family and every pair from ``lattice`` on one
checkpoint, using the same union decomposition as ``overlap_main_grid.py``:

    delta = e(a+b) - |E_a u E_b|

so the numbers are directly comparable to the ones already in ``reports/``.
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

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from maritime_robust.common import ROOT, read_jsonl, stable_seed, write_json  # noqa: E402
from maritime_robust.lattice import (FAMILIES, HELD_OUT, all_pairs, apply_family,
                                     apply_pair)  # noqa: E402
from maritime_robust.methods import build as build_method  # noqa: E402

MEAN, STD, FILL = [.485, .456, .406], [.229, .224, .225], (124, 116, 104)


def batch(rows, fn):
    images = []
    for row in rows:
        with Image.open(ROOT / row["path"]) as f:
            image = f.convert("RGB")
        image = ImageOps.contain(image, (224, 224), Image.Resampling.BICUBIC)
        if fn is not None:
            seed = stable_seed(row["sample_id"], "lattice_eval", 0)
            image = fn(image, seed)
        image = ImageOps.pad(image, (224, 224), method=Image.Resampling.BICUBIC, color=FILL)
        images.append(TF.normalize(TF.to_tensor(image), MEAN, STD))
    return torch.stack(images)


@torch.inference_mode()
def logits_of(model, tensor, chunk=64):
    out = []
    for i in range(0, len(tensor), chunk):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(tensor[i:i + chunk].cuda()).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--model", default="convnext_tiny")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    manifest = PROJECT / f"data/manifests/folds/fgsc23_civ8_fold{args.fold}.jsonl"
    rows = [r for r in read_jsonl(manifest) if r["split"] == "val"]
    labels = np.array([r["label"] for r in rows])

    torch.hub.set_dir(str(ROOT / "artifacts/weights"))
    model = build_method(args.model, 8, "none").cuda()
    checkpoint = torch.load(ROOT / args.run_dir / "best.pt", map_location="cuda", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    errors, accuracy = {}, {}

    def score(name, fn):
        pred = logits_of(model, batch(rows, fn)).argmax(1)
        errors[name] = pred != labels
        accuracy[name] = float(np.mean(pred == labels))

    score("clean", None)
    for family in FAMILIES:
        score(family, lambda im, s, f=family: apply_family(im, f, s))

    records = []
    for a, b in all_pairs():
        score(f"{a}+{b}", lambda im, s, p=(a, b): apply_pair(im, p, s))
        union = 100 * float(np.mean(errors[a] | errors[b]))
        records.append({
            "pair": f"{a}+{b}", "held_out": (a, b) in HELD_OUT,
            "drop_a_pp": 100 * (accuracy["clean"] - accuracy[a]),
            "drop_b_pp": 100 * (accuracy["clean"] - accuracy[b]),
            "drop_pair_pp": 100 * (accuracy["clean"] - accuracy[f"{a}+{b}"]),
            "error_pair_pp": 100 * (1 - accuracy[f"{a}+{b}"]),
            "error_union_pp": union,
            "delta_pp": 100 * (1 - accuracy[f"{a}+{b}"]) - union,
        })

    out_dir = PROJECT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"fold{args.fold}_s{args.seed}.json", {
        "run": args.run_dir, "fold": args.fold, "seed": args.seed, "model": args.model,
        "clean_accuracy": accuracy["clean"], "single_accuracy": {
            f: accuracy[f] for f in FAMILIES}, "pairs": records})

    held = [r for r in records if r["held_out"]]
    seen = [r for r in records if not r["held_out"]]
    print(f"clean {100 * accuracy['clean']:.1f}  "
          f"held-out delta {np.mean([r['delta_pp'] for r in held]):+.1f}  "
          f"seen delta {np.mean([r['delta_pp'] for r in seen]):+.1f}")
    for r in records:
        print(f"{r['pair']:20s} {'HELD' if r['held_out'] else '    '} "
              f"drop {r['drop_pair_pp']:5.1f}  delta {r['delta_pp']:+5.1f}")


if __name__ == "__main__":
    main()
