"""Build an ImageFolder layout for EFM-Net that matches the project protocol.

EFM-Net's own loader expects ``<data_path>/<dataset>/{train,test}`` with one
directory per class. The project protocol has three splits, so the mapping is:

    train  <- manifest split "train"   (network sees these)
    test   <- manifest split "val"     (model selection; EFM-Net calls it "test")
    official test split is NOT linked  (frozen until method freeze)

Images are symlinked, so no pixels are copied.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/manifests/fgsc23_research_civilian.jsonl"
DEFAULT_OUT = ROOT / "external/efmnet_repro/datasets/fgsc23_civ8"
SPLIT_MAP = {"train": "train", "val": "test"}


def slug(name: str) -> str:
    return name.replace(" ", "_")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if args.out.exists():
        if not args.force:
            raise SystemExit(f"{args.out} exists; pass --force to rebuild")
        shutil.rmtree(args.out)

    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        target = SPLIT_MAP.get(row["split"])
        if target is None:
            continue
        src = ROOT / row["path"]
        if not src.exists():
            raise SystemExit(f"missing source image: {src}")
        cls = slug(row["label_name"])
        dst_dir = args.out / target / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if not dst.exists():
            os.symlink(src.resolve(), dst)
        counts.setdefault(target, {}).setdefault(cls, 0)
        counts[target][cls] += 1

    for split in sorted(counts):
        per_class = counts[split]
        print(f"{split}: {sum(per_class.values())} images, {len(per_class)} classes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
