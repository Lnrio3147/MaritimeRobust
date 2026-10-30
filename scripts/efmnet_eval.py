#!/usr/bin/env python
"""Evaluate a reproduced EFM-Net under the project protocol.

EFM-Net is trained by its own code (external/efmnet_repro) on the 8-class
civilian split, with the project val split used as its selection set. This
script re-loads the best checkpoint and scores it with the *project* pipeline
(contain -> corrupt -> pad -> ImageNet normalize), so that its degradation
numbers are comparable to the frozen in-project classifiers.

Two class orders exist and must not be confused:
  project label int  = label from data/manifests/fgsc23_research_civilian.jsonl
  efmnet index       = alphabetical ImageFolder order used during training
The script maps efmnet index -> project label before scoring.

Usage:
    python scripts/efmnet_eval.py --run-dir runs_external/fgsc23_civ8/<run> --split val
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/efmnet_repro"))

from maritime_robust.common import ROOT as PROJECT, read_jsonl, write_json  # noqa: E402
from maritime_robust.degradations import conditions, corrupt, corruption_seed  # noqa: E402
from maritime_robust.engine import metrics  # noqa: E402

MEAN, STD, FILL = [.485, .456, .406], [.229, .224, .225], (124, 116, 104)
ROLE = "efmnet"


def efmnet_classes(rows):
    """Alphabetical class order produced by torchvision ImageFolder."""
    names = sorted({r["label_name"].replace(" ", "_") for r in rows})
    return names


def project_to_efmnet(rows, names):
    """project label int -> efmnet index, derived from the manifest itself."""
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], set()).add(row["label_name"].replace(" ", "_"))
    mapping = {}
    for label, name_set in by_label.items():
        if len(name_set) != 1:
            raise SystemExit(f"project label {label} maps to several names: {name_set}")
        name = next(iter(name_set))
        if name not in names:
            raise SystemExit(f"{name} missing from the EFM-Net class list")
        mapping[label] = names.index(name)
    if sorted(mapping) != list(range(len(names))):
        raise SystemExit(f"label space mismatch: {sorted(mapping)} vs {len(names)} classes")
    return mapping


def build_model(classes, checkpoint, device):
    from models.miner import convnext_base

    model = convnext_base(pretrained=False, M=16, num_classes=len(classes),
                          drop_path_rate=0.2, layer_scale_init_value=1e-6,
                          head_init_scale=0.001, use_mha=True, use_ref=True)
    state = torch.load(checkpoint, map_location="cpu")
    state = state.get("model", state)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(json.dumps({"missing": list(missing)[:8], "unexpected": list(unexpected)[:8]}),
              flush=True)
    return model.to(device).eval()


def content(row, preprocess):
    with Image.open(PROJECT / row["path"]) as handle:
        image = handle.convert("RGB")
    if preprocess == "efmnet":
        return image
    return ImageOps.contain(image, (224, 224), Image.Resampling.BICUBIC)


def pad_norm(image):
    padded = ImageOps.pad(image, (224, 224), Image.Resampling.BICUBIC, color=FILL)
    return TF.normalize(TF.to_tensor(padded), MEAN, STD)


def native_transform():
    """EFM-Net's own eval transform, taken from its own builder (resize 256 + crop 224)."""
    from datasets_builder import build_transform

    class _Args:
        imagenet_default_mean_and_std = True
        input_size = 224
        data_set = "image_folder"
        crop_pct = None

    return build_transform(False, _Args())


@torch.inference_mode()
def probs(model, batch, device):
    out = model(batch.to(device))
    logits = out[0] if isinstance(out, (tuple, list)) else out
    return logits.float().softmax(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default="checkpoint-best.pth")
    ap.add_argument("--manifest", default="data/manifests/fgsc23_research_civilian.jsonl")
    ap.add_argument("--split", default="val")
    ap.add_argument("--conditions", default="all", help="'all', 'clean', or comma names")
    ap.add_argument("--output", default="reports/efmnet_eval.json")
    ap.add_argument("--preprocess", choices=("project", "efmnet"), default="project",
                    help="project: contain+gray pad; efmnet: the model's own eval transform")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = PROJECT / args.run_dir
    rows = [r for r in read_jsonl(PROJECT / args.manifest) if r["split"] == args.split]
    if not rows:
        raise SystemExit(f"no {args.split} rows")

    names = efmnet_classes(rows)
    label_map = project_to_efmnet(rows, names)
    model = build_model(names, run_dir / args.checkpoint, device)

    every = conditions()
    if args.conditions == "clean":
        every = every[:1]
    elif args.conditions != "all":
        wanted = {c.strip() for c in args.conditions.split(",")}
        every = [c for c in every if c["name"] in wanted]

    contents = [content(r, args.preprocess) for r in rows]
    native = native_transform() if args.preprocess == "efmnet" else None
    labels = np.array([r["label"] for r in rows])
    inverse = {v: k for k, v in label_map.items()}
    results = {"clean": {}, "conditions": {}}
    per_sample = {}

    for cond in every:
        batch = []
        for row, image in zip(rows, contents):
            out, _ = corrupt(image, kind=cond["kind"], severity=cond["severity"],
                             seed=corruption_seed(row["sample_id"], ROLE, cond["kind"],
                                                  cond["severity"], cond["domain"]),
                             domain=cond["domain"], placement=cond["placement"])
            batch.append(native(out) if native is not None else pad_norm(out))
        p = probs(model, torch.stack(batch), device)
        # efmnet index -> project label int
        predictions = np.array([inverse[int(i)] for i in p.argmax(1)])
        score = metrics(labels.tolist(), predictions.tolist(), len(names))
        entry = {"accuracy": score["accuracy"], "macro_f1": score["macro_f1"],
                 "n": int(len(labels))}
        if cond["name"] == "clean":
            results["clean"] = entry
        else:
            results["conditions"][cond["name"]] = entry
        per_sample[cond["name"]] = {"label": labels.tolist(), "pred": predictions.tolist(),
                                    "probs": p[:, [label_map[i] for i in range(len(names))]].tolist()}
        print(json.dumps({cond["name"]: round(entry["accuracy"], 4)}), flush=True)

    payload = {
        "model": "EFM-Net (ConvNeXt-B + MHA + BPP), architecture reimplementation",
        "run_dir": args.run_dir, "checkpoint": args.checkpoint,
        "manifest": args.manifest, "split": args.split, "n": int(len(labels)),
        "project_to_efmnet": label_map, "efmnet_classes": names,
        "results": results,
        "preprocess": args.preprocess,
        "note": ("project pipeline (contain->corrupt->pad->normalize)" if args.preprocess == "project"
                 else "EFM-Net's own eval transform (resize 256 squash -> center crop 224)"),
        "role": ROLE,
    }
    write_json(PROJECT / args.output, payload)
    write_json(PROJECT / args.output.replace(".json", "_samples.json"), per_sample)
    clean = results["clean"]["accuracy"]
    worsening = {k: 100 * (clean - v["accuracy"]) for k, v in results["conditions"].items()}
    print(json.dumps({"clean": clean,
                      "worst": sorted(worsening.items(), key=lambda kv: -kv[1])[:5]},
                     indent=1), flush=True)


if __name__ == "__main__":
    main()
