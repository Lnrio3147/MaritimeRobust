#!/usr/bin/env python
"""Score one screening run on the 8 combo conditions + clean, on its fold's val rows.

Same image pipeline, corruption parameters and seeds as
scripts/baseline_val_conditions.py (role "efmnet"), so every run's predictions
are paired per image and can be differenced directly.

    python scripts/explore_eval.py --run runs/fgsc23_civ8_fold0/convnext_tiny_combo_s17 \
        --manifest data/manifests/folds/fgsc23_civ8_fold0.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from maritime_robust.common import ROOT as PROJECT, read_jsonl, write_json  # noqa: E402
from maritime_robust.degradations import conditions, corrupt, corruption_seed  # noqa: E402
from maritime_robust.engine import load_run, metrics  # noqa: E402

MEAN, STD, FILL = [.485, .456, .406], [.229, .224, .225], (124, 116, 104)
ROLE = "efmnet"


def load_any(run):
    """Load either a frozen-pipeline run or an explore_train method run."""
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    settings = checkpoint["settings"]
    if "method" not in settings:
        model, settings, _ = load_run(run)
        return model, settings
    from maritime_robust.methods import build
    model = build(settings["model"], settings["num_classes"], settings["method"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model.cuda(), settings


def latency_ms(model, settings, batch, warmup=10, iters=50):
    """Single-image latency, median over 50 timed forwards after warmup."""
    model.eval()
    x = torch.randn(1, 3, 224, 224, device="cuda")
    times = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(warmup):
            model(x)
        for _ in range(iters):
            torch.cuda.synchronize()
            started = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - started) * 1000)
    return float(np.median(times))


def content(row):
    with Image.open(PROJECT / row["path"]) as handle:
        image = handle.convert("RGB")
    return ImageOps.contain(image, (224, 224), Image.Resampling.BICUBIC)


def pad_norm(image):
    padded = ImageOps.pad(image, (224, 224), Image.Resampling.BICUBIC, color=FILL)
    return TF.normalize(TF.to_tensor(padded), MEAN, STD)


def all_conditions():
    """every registered condition (clean included), tagged clean/single/combo."""
    tagged = []
    for c in conditions():
        kind = c["kind"]
        layer = "clean" if kind == "clean" else ("combo" if "+" in kind else "single")
        tagged.append(dict(c, layer=layer))
    return tagged


def combo_names():
    return [c["name"] for c in conditions() if "+" in c["kind"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default="reports/s1")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    model, settings = load_any(PROJECT / args.run)
    rows = [r for r in read_jsonl(PROJECT / args.manifest) if r["split"] == "val"]
    if not rows:
        raise SystemExit("no val rows in manifest")
    labels = np.array([r["label"] for r in rows])
    contents = [content(r) for r in rows]
    out = {"run": args.run, "manifest": args.manifest, "n": int(len(rows)),
           "seed": settings.get("seed"), "conditions": {}}

    for cond in all_conditions():
        batch = []
        for row, image in zip(rows, contents):
            if cond["kind"] == "clean":
                batch.append(pad_norm(image))
                continue
            degraded, _ = corrupt(image, kind=cond["kind"], severity=cond["severity"],
                                  seed=corruption_seed(row["sample_id"], ROLE, cond["kind"],
                                                       cond["severity"], cond["domain"]),
                                  domain=cond["domain"], placement=cond["placement"])
            batch.append(pad_norm(degraded))
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

    combo = combo_names()
    out["combo_conditions"] = combo
    out["combo_mean_accuracy"] = float(np.mean([out["conditions"][c]["accuracy"] for c in combo]))
    out["combo_mean_macro_f1"] = float(np.mean([out["conditions"][c]["macro_f1"] for c in combo]))
    for layer in ("clean", "single", "combo"):
        names = [n for n, c in out["conditions"].items() if c["layer"] == layer]
        out[f"{layer}_mean_accuracy"] = float(np.mean([out["conditions"][n]["accuracy"]
                                                       for n in names]))
    out["latency_ms_per_image"] = latency_ms(model, settings, args.batch)
    # include the fold directory: run basenames repeat across folds
    name = f"{Path(args.run).parent.name}__{Path(args.run).name}"
    out_dir = PROJECT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"{name}.json", out)
    print(json.dumps({"run": args.run, "clean": out["conditions"]["clean"]["accuracy"],
                      "combo_mean": out["combo_mean_accuracy"],
                      "combo_min": min(out["conditions"][c]["accuracy"] for c in combo)}), flush=True)


if __name__ == "__main__":
    main()
