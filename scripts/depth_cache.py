#!/usr/bin/env python
"""Cache monocular depth maps for a manifest so fog evaluation never re-runs the depth model."""
import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maritime_robust.common import ROOT as PROJECT, read_jsonl, write_json  # noqa: E402
from maritime_robust.depth_fog import DEPTH_CHECKPOINT, DepthProvider, save_cached_depth  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/fgsc23_research_civilian.jsonl")
    parser.add_argument("--provider", default="depth-anything", choices=("horizon", "flat", "depth-anything"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output = Path(args.output) if args.output else PROJECT / "artifacts/depth_cache" / args.provider
    rows = read_jsonl(PROJECT / args.manifest)
    if args.limit:
        rows = rows[:args.limit]
    pending = [r for r in rows if not (output / f"{r['sample_id']}.npy").exists()]
    print(json.dumps({"manifest": args.manifest, "rows": len(rows), "pending": len(pending),
                      "output": str(output)}), flush=True)
    if args.dry_run or not pending:
        return

    provider = DepthProvider(args.provider, device=args.device)
    started = time.time()
    written = 0
    for row in pending:
        image = Image.open(PROJECT / row["path"]).convert("RGB")
        depth = provider(image)
        save_cached_depth(output, row["sample_id"], depth, size=(args.size, args.size))
        written += 1
        if written % 100 == 0:
            rate = written / max(time.time() - started, 1e-6)
            print(json.dumps({"written": written, "remaining": len(pending) - written,
                              "images_per_s": round(rate, 2)}), flush=True)
    write_json(output / "index.json", {
        "provider": args.provider,
        "checkpoint": DEPTH_CHECKPOINT if args.provider == "depth-anything" else None,
        "size": args.size, "files": written + (len(rows) - len(pending)),
        "seconds": round(time.time() - started, 1), "manifest": args.manifest,
        "convention": "normalized depth, 1 = farthest",
    })
    print(json.dumps({"written": written, "seconds": round(time.time() - started, 1)}))


if __name__ == "__main__":
    main()
