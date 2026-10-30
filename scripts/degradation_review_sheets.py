#!/usr/bin/env python
"""Build contact sheets for manual real-degradation review.

One tile per validation image, labelled with a stable review index and class.
The review index is the join key for the annotation table; sample_id and path
are written alongside so labels can be audited against the manifest.
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maritime_robust.common import ROOT as PROJECT, read_jsonl  # noqa: E402

FILL = (124, 116, 104)
TILE = 224
HEADER = 18


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("fgsc23", "fgsrcs"), default="fgsc23")
    parser.add_argument("--split", default="")
    parser.add_argument("--per-sheet", type=int, default=20)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if args.dataset == "fgsc23":
        split = args.split or "val"
        rows = [r for r in read_jsonl(PROJECT / "data/manifests/fgsc23_research_civilian.jsonl")
                if r["split"] == split]
        output = PROJECT / (args.output or "reviews/real_degradation/fgsc23_val")
    else:
        split = args.split or "development"
        rows = [r for r in read_jsonl(PROJECT / "data/manifests/fgsrcs_known_box_crops.jsonl")
                if r.get("strict_fgsc23_mapping") and r["split"] == split]
        output = PROJECT / (args.output or "reviews/real_degradation/fgsrcs_dev")
    sheets = output / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)

    index = []
    tiles = []
    for number, row in enumerate(rows, 1):
        with Image.open(PROJECT / row["path"]) as f:
            image = f.convert("RGB")
        contained = ImageOps.contain(image, (TILE, TILE), Image.Resampling.BICUBIC)
        square = ImageOps.pad(contained, (TILE, TILE), Image.Resampling.BICUBIC, color=FILL)
        canvas = Image.new("RGB", (TILE, TILE + HEADER), (24, 24, 24))
        canvas.paste(square, (0, HEADER))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 3), f"{number:03d}  {row['label_name']}", fill=(235, 235, 235))
        tiles.append(canvas)
        index.append({"review_index": number, "sample_id": row["sample_id"],
                      "label": row.get("label"), "label_name": row["label_name"],
                      "mapping": row.get("strict_fgsc23_mapping"),
                      "width": row["width"], "height": row["height"],
                      "source_group_id": row["source_group_id"], "path": row["path"]})

    rows_per_sheet = args.columns
    sheet_number = 0
    for start in range(0, len(tiles), args.per_sheet):
        chunk = tiles[start:start + args.per_sheet]
        sheet_number += 1
        lines = (len(chunk) + rows_per_sheet - 1) // rows_per_sheet
        sheet = Image.new("RGB", (args.columns * TILE, lines * (TILE + HEADER)),
                          (24, 24, 24))
        for position, tile in enumerate(chunk):
            sheet.paste(tile, ((position % args.columns) * TILE,
                               (position // args.columns) * (TILE + HEADER)))
        sheet.save(sheets / f"sheet_{sheet_number:02d}.png")
    (output / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2))
    print(json.dumps({"images": len(rows), "sheets": sheet_number,
                      "output": str(output.relative_to(PROJECT))}))


if __name__ == "__main__":
    main()
