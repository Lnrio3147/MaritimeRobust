#!/usr/bin/env python
"""Qualitative contact sheet and texture check for the paired restorer.

Renders clean / degraded / restored tiles for a few validation samples and
reports mean gradient energy, so pixel-L1 gains can be compared with how much
fine structure survives.  Read-only diagnostic; no metric here feeds admission.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from maritime_robust.common import ROOT as PROJECT, read_jsonl  # noqa: E402
from maritime_robust.degradations import corrupt, corruption_seed  # noqa: E402
from modules.pair_restorer import PairRestorer, restore_images  # noqa: E402

FILL = (124, 116, 104)
TILE = 224
HEADER = 20


def content(row):
    with Image.open(PROJECT / row["path"]) as f:
        image = f.convert("RGB")
    return ImageOps.contain(image, (TILE, TILE), Image.Resampling.BICUBIC)


def gradient_energy(image):
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    return float((gx + gy) / 2)


def l1_psnr(a, b):
    x = np.asarray(a, dtype=np.float32) / 255.0
    y = np.asarray(b, dtype=np.float32) / 255.0
    mse = float(((x - y) ** 2).mean())
    return float(np.abs(x - y).mean()), (10 * math.log10(1 / mse) if mse > 1e-12 else None)


def load_restorer(seed, manifest):
    path = PROJECT / f"restorers/{Path(manifest).stem}/pairunet_s{seed}/best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = PairRestorer(**checkpoint["settings"]["model_config"]).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def tile(image, label):
    square = ImageOps.pad(image, (TILE, TILE), Image.Resampling.BICUBIC, color=FILL)
    canvas = Image.new("RGB", (TILE, TILE + HEADER), (24, 24, 24))
    canvas.paste(square, (0, HEADER))
    ImageDraw.Draw(canvas).text((4, 3), label, fill=(235, 235, 235))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--manifest", default="data/manifests/fgsc23_research_civilian.jsonl")
    parser.add_argument("--kind", default="haze+glare")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--restorer-seeds", type=int, nargs="+", default=[17])
    parser.add_argument("--output", default="reports/figures/restorer_contact_sheet.png")
    args = parser.parse_args()

    rows = [r for r in read_jsonl(args.manifest) if r["split"] == args.split]
    picked, seen = [], set()
    for row in rows:
        if row["label"] in seen:
            continue
        seen.add(row["label"])
        picked.append(row)
        if len(picked) == args.samples:
            break

    role = "test" if args.split == "test" else "val"
    cleans, degraded = [], []
    for row in picked:
        clean = content(row)
        seed = corruption_seed(row["sample_id"], role, args.kind, args.severity,
                               "matched", 0)
        deg, _ = corrupt(clean, seed=seed, kind=args.kind, severity=args.severity,
                         domain="matched", placement="random")
        cleans.append(clean)
        degraded.append(deg)
    restored = {seed: restore_images(load_restorer(seed, args.manifest), degraded,
                                     device="cuda")
                for seed in args.restorer_seeds}

    stats = []
    for index, row in enumerate(picked):
        entry = {"sample_id": row["sample_id"], "label": row["label"],
                 "clean_gradient": gradient_energy(cleans[index]),
                 "degraded_gradient": gradient_energy(degraded[index]),
                 "restored_gradient": {},
                 "restored_l1": {}, "restored_psnr": {}}
        for seed in args.restorer_seeds:
            image = restored[seed][index]
            l1, psnr = l1_psnr(image, cleans[index])
            entry["restored_gradient"][str(seed)] = gradient_energy(image)
            entry["restored_l1"][str(seed)] = l1
            entry["restored_psnr"][str(seed)] = psnr
        stats.append(entry)

    columns = 2 + len(args.restorer_seeds)
    canvas = Image.new("RGB", (columns * TILE, len(picked) * (TILE + HEADER)),
                       (24, 24, 24))
    for index, row in enumerate(picked):
        y = index * (TILE + HEADER)
        canvas.paste(tile(cleans[index], "clean"), (0, y))
        canvas.paste(tile(degraded[index],
                          f"degraded {args.kind}{args.severity}"), (TILE, y))
        for offset, seed in enumerate(args.restorer_seeds):
            entry = stats[index]
            text = (f"restored s{seed}  L1={entry['restored_l1'][str(seed)]:.3f}  "
                    f"grad={entry['restored_gradient'][str(seed)] / entry['clean_gradient']:.2f}x")
            canvas.paste(tile(restored[seed][index], text), ((2 + offset) * TILE, y))

    output = PROJECT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)

    ratios = {str(seed): float(np.mean([s["restored_gradient"][str(seed)] /
                                        s["clean_gradient"] for s in stats]))
              for seed in args.restorer_seeds}
    degraded_ratio = float(np.mean([s["degraded_gradient"] / s["clean_gradient"]
                                    for s in stats]))
    summary = {"output": str(output.relative_to(PROJECT)), "samples": len(picked),
               "kind": args.kind, "severity": args.severity,
               "degraded_gradient_ratio": degraded_ratio,
               "restored_gradient_ratio": ratios,
               "mean_restored_l1": {str(seed): float(np.mean(
                   [s["restored_l1"][str(seed)] for s in stats]))
                   for seed in args.restorer_seeds},
               "per_sample": stats}
    (output.parent / (output.stem + ".json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
