#!/usr/bin/env python
"""Contact sheet: depth-conditioned fog vs the depth-blind control, for eyeballing only."""
import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maritime_robust.common import ROOT as PROJECT, read_jsonl  # noqa: E402
from maritime_robust.depth_fog import DepthProvider, apply_fog, depth_fog_seed  # noqa: E402

SEVERITIES = (1, 3, 5)
PROVIDERS = ("flat", "horizon", "depth-anything")
TILE = 224


def label(image, text):
    canvas = Image.new("RGB", (image.width, image.height + 16), (255, 255, 255))
    canvas.paste(image, (0, 16))
    ImageDraw.Draw(canvas).text((3, 3), text, fill=(0, 0, 0))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/fgsc23_research_civilian.jsonl")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--output", default="reports/figures/depth_fog_preview.png")
    parser.add_argument("--providers", default=",".join(PROVIDERS))
    args = parser.parse_args()

    rows = read_jsonl(PROJECT / args.manifest)[:args.count]
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    blocks = []
    for row in rows:
        image = ImageOps.contain(Image.open(PROJECT / row["path"]).convert("RGB"), (TILE, TILE),
                                 Image.Resampling.BICUBIC)
        row_tiles = [label(image, f"{row['label_name']} clean")]
        for name in providers:
            depth = DepthProvider(name)(image)
            for severity in SEVERITIES:
                out, _ = apply_fog(image, depth, severity, turbulence=0,
                                   seed=depth_fog_seed(row["sample_id"], "preview", "depth_haze",
                                                       severity, base_seed=0))
                row_tiles.append(label(out, f"{name} s{severity}"))
        blocks.append(row_tiles)

    width = max(sum(t.width for t in r) for r in blocks)
    height = sum(max(t.height for t in r) for r in blocks)
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    y = 0
    for row_tiles in blocks:
        x = 0
        for tile in row_tiles:
            sheet.paste(tile, (x, y))
            x += tile.width
        y += max(t.height for t in row_tiles)
    out_path = PROJECT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(str(out_path))


if __name__ == "__main__":
    main()
