"""Evaluate the compound conditions at severities 2 and 4 (pre-registered).

Only builds the four extra condition entries; corrupt() already supports every
severity 1..5 for compound kinds, so degradations.py is untouched and the
corruption instances are identical to what explore_eval.py would produce.

    python scripts/severity_curve_eval.py --run runs/fgsc23_civ8_fold0/convnext_tiny_none_e40_s17 \
        --manifest data/manifests/folds/fgsc23_civ8_fold0.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import explore_eval as ee  # noqa: E402
from maritime_robust.common import read_jsonl  # noqa: E402
from maritime_robust.degradations import corrupt, corruption_seed  # noqa: E402
from maritime_robust.engine import metrics  # noqa: E402

EXTRA = [
    {"name": f"matched_{kind}_{sev}", "kind": kind, "severity": sev,
     "domain": "matched", "placement": "random", "layer": "combo"}
    for kind in ("haze+glare", "haze+cloud")
    for sev in (2, 4)
]
EXTRA.insert(0, {"name": "clean", "kind": "clean", "severity": 0,
                 "domain": "matched", "placement": "random", "layer": "clean"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default="reports/severity_curve")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    run = ee.PROJECT / args.run
    model, settings = ee.load_any(run)
    rows = [r for r in read_jsonl(ee.PROJECT / args.manifest) if r["split"] == "val"]
    labels = torch.tensor([r["label"] for r in rows])
    contents = [ee.content(r) for r in rows]

    out = {"run": args.run, "manifest": args.manifest, "n": len(rows),
           "seed": settings.get("seed"), "conditions": {}}
    for cond in EXTRA:
        batch = []
        for row, image in zip(rows, contents):
            if cond["kind"] == "clean":
                batch.append(ee.pad_norm(image))
                continue
            degraded, _ = corrupt(
                image, kind=cond["kind"], severity=cond["severity"],
                seed=corruption_seed(row["sample_id"], ee.ROLE, cond["kind"],
                                     cond["severity"], cond["domain"]),
                domain=cond["domain"], placement=cond["placement"])
            batch.append(ee.pad_norm(degraded))
        tensor = torch.stack(batch)
        preds = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for start in range(0, len(tensor), args.batch):
                logits = model(tensor[start:start + args.batch].cuda())
                preds.append(logits.float().argmax(1).cpu())
        preds = torch.cat(preds).numpy()
        score = metrics(labels.tolist(), preds.tolist(), settings["num_classes"])
        out["conditions"][cond["name"]] = {
            "layer": cond["layer"], "accuracy": score["accuracy"],
            "macro_f1": score["macro_f1"], "per_class_recall": score["per_class_recall"],
            "predictions": preds.tolist()}

    name = f"{Path(args.run).parent.name}__{Path(args.run).name}.json"
    out_dir = ee.PROJECT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(json.dumps(out) + "\n")
    print(json.dumps({"run": args.run,
                      **{k: round(v["accuracy"], 4) for k, v in out["conditions"].items()}}), flush=True)


if __name__ == "__main__":
    main()
