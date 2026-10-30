#!/usr/bin/env python
"""Minimal depth-haze check on the validation split with frozen classifiers.

Three conditions per severity, all on the same samples:
  flat_haze   the existing `degradations.corrupt(kind="haze")` recipe
  depth_haze  depth-conditioned scattering, far-field transmittance aligned to the same grid
  equal_flat  flat fog with the per-image *mean* transmittance of `depth_haze`
`depth_haze - equal_flat` isolates the spatial structure; `depth_haze - flat_haze` mixes in
the lower mean opacity that depth conditioning implies.  Validation only; nothing is trained.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from maritime_robust.common import ROOT as PROJECT, read_jsonl, write_json  # noqa: E402
from maritime_robust.degradations import corrupt, corruption_seed  # noqa: E402
from maritime_robust.depth_fog import (apply_fog, depth_fog_seed,  # noqa: E402
                                       load_cached_depth, transmission_map)
from maritime_robust.engine import load_run, metrics  # noqa: E402

MEAN, STD, FILL = [.485, .456, .406], [.229, .224, .225], (124, 116, 104)
BACKBONES = ("convnext_tiny", "swin_t", "resnet50")


def content(row):
    with Image.open(PROJECT / row["path"]) as handle:
        image = handle.convert("RGB")
    return ImageOps.contain(image, (224, 224), Image.Resampling.BICUBIC)


def pad_norm(image):
    padded = ImageOps.pad(image, (224, 224), Image.Resampling.BICUBIC, color=FILL)
    return TF.normalize(TF.to_tensor(padded), MEAN, STD)


@torch.inference_mode()
def probs(model, batch):
    model.eval()
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(batch.cuda())
    return logits.float().softmax(1).cpu().numpy()


def classifiers():
    found = {}
    for backbone in BACKBONES:
        for path in sorted((PROJECT / "runs/fgsc23_research_civilian").glob(f"{backbone}_combo_s*")):
            if (path / "evaluation/summary.json").exists():
                found[path.name] = path
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/fgsc23_research_civilian.jsonl")
    parser.add_argument("--split", default="val")
    parser.add_argument("--severities", default="1,3,5")
    parser.add_argument("--depth-cache", default="artifacts/depth_cache/depth-anything")
    parser.add_argument("--output", default="reports/depth_fog_check.json")
    parser.add_argument("--domain", default="matched")
    args = parser.parse_args()

    severities = [int(s) for s in args.severities.split(",")]
    rows = [r for r in read_jsonl(PROJECT / args.manifest) if r["split"] == args.split]
    if not rows:
        raise SystemExit(f"no {args.split} rows in {args.manifest}")
    cache = PROJECT / args.depth_cache
    labels = np.array([r["label"] for r in rows])

    models, num_classes = {}, 0
    for name, path in classifiers().items():
        model, settings, _ = load_run(path)
        models[name] = model.eval()
        num_classes = settings["num_classes"]
    if not models:
        raise SystemExit("no frozen combined-augmentation classifiers found")
    print(json.dumps({"classifiers": list(models), "samples": len(rows),
                      "severities": severities}), flush=True)

    tensors, summaries = {}, {}

    def add(name, images, params):
        tensors[name] = torch.stack([pad_norm(im) for im in images])
        mean = lambda key: float(np.mean([p.get(key, 1.0) for p in params]))
        summaries[name] = {"transmission_mean": mean("transmission_mean"),
                           "transmission_min": mean("transmission_min"),
                           "transmission_max": mean("transmission_max")}

    contents = [content(r) for r in rows]
    clean_images, clean_params = [], []
    for row, image in zip(rows, contents):
        out, params = corrupt(image, kind="clean", seed=0, domain=args.domain)
        clean_images.append(out)
        clean_params.append(params)
    add("clean", clean_images, clean_params)

    for severity in severities:
        flat, flat_params, depth, depth_params = [], [], [], []
        for row, image in zip(rows, contents):
            f_out, f_params = corrupt(image, kind="haze", severity=severity, domain=args.domain,
                                      seed=corruption_seed(row["sample_id"], "depth_fog_check", "haze",
                                                           severity, args.domain))
            field = load_cached_depth(cache, row["sample_id"], image.size)
            d_out, d_params = apply_fog(image, field, severity, domain=args.domain, turbulence=0,
                                        seed=depth_fog_seed(row["sample_id"], "depth_fog_check",
                                                            "depth_haze", severity, args.domain))
            flat.append(f_out)
            flat_params.append(f_params)
            depth.append(d_out)
            depth_params.append(d_params)
        add(f"flat_haze_s{severity}", flat, flat_params)
        add(f"depth_haze_s{severity}", depth, depth_params)

        equal, equal_params = [], []
        for row, image, params in zip(rows, contents, depth_params):
            field = load_cached_depth(cache, row["sample_id"], image.size)
            t_map = np.full(field.shape, params["transmission_mean"], np.float32)
            out, p = apply_fog(image, field, severity, domain=args.domain, turbulence=0,
                               seed=depth_fog_seed(row["sample_id"], "depth_fog_check", "depth_haze",
                                                   severity, args.domain), t_override=t_map)
            equal.append(out)
            equal_params.append(p)
        add(f"equal_flat_s{severity}", equal, equal_params)

    results = {name: {} for name in models}
    for condition, batch in tensors.items():
        for name, model in models.items():
            prediction = probs(model, batch).argmax(1)
            score = metrics(labels.tolist(), prediction.tolist(), num_classes)
            results[name][condition] = {"accuracy": score["accuracy"],
                                        "macro_f1": score["macro_f1"],
                                        "n": int(len(labels))}
    for condition in summaries:
        for name in models:
            results[name][condition].update(summaries[condition])

    summary = {}
    for severity in severities:
        row = {}
        for condition in (f"flat_haze_s{severity}", f"depth_haze_s{severity}", f"equal_flat_s{severity}"):
            row[condition] = float(np.mean([results[n][condition]["accuracy"] for n in models]))
            row[condition + "_t"] = summaries[condition]["transmission_mean"]
        row["depth_minus_flat_pp"] = 100 * (row[f"depth_haze_s{severity}"] - row[f"flat_haze_s{severity}"])
        row["depth_minus_equal_pp"] = 100 * (row[f"depth_haze_s{severity}"] - row[f"equal_flat_s{severity}"])
        summary[str(severity)] = row

    write_json(PROJECT / args.output, {
        "manifest": args.manifest, "split": args.split, "n": int(len(labels)),
        "depth_cache": args.depth_cache, "domain": args.domain, "severities": severities,
        "classifiers": {n: results[n] for n in results}, "summary": summary,
        "note": "validation only; macro-F1 over 8 civilian classes; no training",
    })
    print(json.dumps({"summary": summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
