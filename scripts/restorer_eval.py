#!/usr/bin/env python
"""Evaluate paired restorers with frozen combined-augmentation classifiers.

Identity (no restoration) and each restorer seed are scored on the same
samples and conditions.  Restoration happens on contained content before
letterbox padding, matching the frozen evaluation pipeline.  The clean identity
path is asserted against saved predictions (test) or the checkpoint validation
score (val) before any restored result is reported.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from maritime_robust.common import (ROOT as PROJECT, Budget, digest, environment,  # noqa: E402
                                    read_jsonl, write_json, write_jsonl)
from maritime_robust.degradations import conditions as all_conditions  # noqa: E402
from maritime_robust.degradations import corrupt, corruption_seed  # noqa: E402
from maritime_robust.engine import load_run, metrics  # noqa: E402
from modules.pair_restorer import PairRestorer, restore_images  # noqa: E402

MEAN = [.485, .456, .406]
STD = [.229, .224, .225]
FILL = (124, 116, 104)
BACKBONES = ("resnet50", "convnext_tiny", "swin_t")
CLASSIFIER_SEEDS = (17, 29, 43)
PRIMARY = ("clean", "matched_haze_5", "matched_glare_5", "matched_haze+glare_5",
           "matched_haze+cloud_5", "shifted_haze+glare_5")


def content(row):
    with Image.open(PROJECT / row["path"]) as f:
        image = f.convert("RGB")
    return ImageOps.contain(image, (224, 224), Image.Resampling.BICUBIC)


def pad_norm(image):
    padded = ImageOps.pad(image, (224, 224), Image.Resampling.BICUBIC, color=FILL)
    return TF.normalize(TF.to_tensor(padded), MEAN, STD)


@torch.inference_mode()
def probs_for(model, tensors, batch_size=64):
    model.eval()
    output = []
    for start in range(0, len(tensors), batch_size):
        x = torch.stack(tensors[start:start + batch_size]).cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        output.append(logits.float().softmax(1).cpu())
    return torch.cat(output)


def score_of(probs, labels, n):
    preds = probs.argmax(1).tolist()
    return metrics(labels, preds, n), preds


def margin_values(probs):
    top2 = probs.topk(2, dim=1).values
    return (top2[:, 0] - top2[:, 1]).numpy()


def summarize(probs, labels, n, reference_preds, clean_preds):
    score, preds = score_of(probs, labels, n)
    count = len(labels)
    changed = [i for i in range(count) if preds[i] != reference_preds[i]]
    rescued = [i for i in changed if preds[i] == labels[i]]
    hurt = [i for i in changed if reference_preds[i] == labels[i]]
    recoverable = [i for i in range(count)
                   if reference_preds[i] != labels[i] and clean_preds[i] == labels[i]]
    stable = [i for i in range(count)
              if reference_preds[i] == labels[i] and clean_preds[i] == labels[i]]
    max_prob = probs.max(1).values.numpy()
    margins = margin_values(probs)
    high_confidence_wrong = sum(1 for i in range(count)
                                if preds[i] != labels[i] and max_prob[i] >= 0.9)
    return {
        "accuracy": score["accuracy"], "macro_f1": score["macro_f1"],
        "per_class_recall": score["per_class_recall"],
        "changed": len(changed), "rescued": len(rescued), "hurt": len(hurt),
        "recoverable_n": len(recoverable),
        "rescue_rate": (sum(preds[i] == labels[i] for i in recoverable) /
                        len(recoverable)) if recoverable else None,
        "stable_n": len(stable),
        "error_induction_rate": (sum(preds[i] != labels[i] for i in stable) /
                                 len(stable)) if stable else None,
        "hce_rate": high_confidence_wrong / count,
        "mean_margin": float(margins.mean()),
        "mean_margin_recoverable": (float(margins[recoverable].mean())
                                    if recoverable else None),
    }


def pixel_stats(images, targets):
    l1, psnr, saturation = [], [], []
    for image, target in zip(images, targets):
        a = np.asarray(image, dtype=np.float32) / 255.0
        b = np.asarray(target, dtype=np.float32) / 255.0
        if a.shape != b.shape:
            raise ValueError("restored image changed content size")
        mse = float(((a - b) ** 2).mean())
        l1.append(float(np.abs(a - b).mean()))
        psnr.append(10 * math.log10(1 / mse) if mse > 1e-12 else None)
        saturation.append(float((np.asarray(image).max(2) >= 250).mean()))
    finite = [x for x in psnr if x is not None]
    return {"l1": float(np.mean(l1)), "psnr": float(np.mean(finite)) if finite else None,
            "psnr_n": len(finite), "saturation_fraction": float(np.mean(saturation))}


def condition_by_name(name):
    for c in all_conditions():
        if c["name"] == name:
            return dict(c)
    raise KeyError(name)


def degraded_contents(rows, cond, split):
    role = "test" if split == "test" else "val"
    result = []
    for row in rows:
        image = content(row)
        if cond["kind"] == "clean":
            result.append(image)
            continue
        seed = corruption_seed(row["sample_id"], role, cond["kind"], cond["severity"],
                               cond["domain"], 0)
        degraded, _ = corrupt(image, seed=seed, kind=cond["kind"], severity=cond["severity"],
                              domain=cond["domain"], placement=cond["placement"])
        result.append(degraded)
    return result


def classifier_runs():
    found = {}
    for backbone in BACKBONES:
        for seed in CLASSIFIER_SEEDS:
            path = PROJECT / f"runs/fgsc23_research_civilian/{backbone}_combo_s{seed}"
            if (path / "evaluation/clean.jsonl").exists():
                found[f"{backbone}_s{seed}"] = {"backbone": backbone, "seed": seed,
                                                "path": path}
    return found


def load_restorer(seed, manifest):
    path = PROJECT / f"restorers/{Path(manifest).stem}/pairunet_s{seed}/best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    settings = checkpoint["settings"]
    if settings["manifest_sha256"] != digest(manifest):
        raise ValueError(f"Restorer {seed} was trained on a different manifest")
    model = PairRestorer(**settings["model_config"]).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint


def clean_parity(name, path, preds, rows, split, summary):
    if split == "test":
        saved = read_jsonl(path / "evaluation/clean.jsonl")
        expected = {r["sample_id"]: r["prediction"] for r in saved}
        bad = [r["sample_id"] for r, p in zip(rows, preds)
               if expected.get(r["sample_id"]) != p]
        if bad:
            raise AssertionError(f"{name}: clean predictions differ in {len(bad)}/"
                                 f"{len(rows)} samples")
        return
    complete = json.loads((path / "complete.json").read_text())
    expected = complete["best_val_macro_f1"]
    if abs(summary["macro_f1"] - expected) > 1e-9:
        raise AssertionError(f"{name}: val clean macro-F1 {summary['macro_f1']} != "
                             f"checkpoint {expected}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--manifest", default="data/manifests/fgsc23_research_civilian.jsonl")
    parser.add_argument("--restorer-seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--condition-names", default="", help="comma-separated subset")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    all_rows = [r for r in read_jsonl(args.manifest) if r["split"] == args.split]
    labels_all = [r["label"] for r in all_rows]
    n = max(labels_all) + 1
    rows = all_rows[:args.limit] if args.limit else all_rows
    labels = [r["label"] for r in rows]
    conditions = ([condition_by_name(x) for x in args.condition_names.split(",")]
                  if args.condition_names else all_conditions())
    names = [c["name"] for c in conditions]
    if "clean" not in names:
        raise ValueError("clean condition required for the parity and recoverable sets")
    conditions = sorted(conditions, key=lambda c: c["name"] != "clean")
    classifiers = classifier_runs()
    if not classifiers:
        raise ValueError("No combined-augmentation classifiers found")
    restorers = {seed: load_restorer(seed, args.manifest) for seed in args.restorer_seeds}
    models = {}
    for name, info in classifiers.items():
        model, settings, _ = load_run(info["path"])
        if settings["num_classes"] != n:
            raise ValueError(f"{name}: classifier classes {settings['num_classes']} != {n}")
        models[name] = model

    result = {"split": args.split, "n": len(rows), "manifest_sha256": digest(args.manifest),
              "conditions": [c["name"] for c in conditions],
              "primary_conditions": [c for c in PRIMARY if c in
                                     {x["name"] for x in conditions}],
              "restorer_seeds": list(restorers),
              "classifiers": {name: {"backbone": info["backbone"], "seed": info["seed"],
                                     "identity": {}, "restorers": {}}
                              for name, info in classifiers.items()},
              "identity_pixel": {}, "pixel": {str(seed): {} for seed in restorers},
              "environment": environment()}
    records = []
    clean_preds_by_classifier = {}
    with Budget("evaluation"):
        for cond in conditions:
            name = cond["name"]
            degraded = degraded_contents(rows, cond, args.split)
            targets = [content(row) for row in rows]
            restored = {seed: restore_images(model, degraded, device="cuda")
                        for seed, (model, _) in restorers.items()}
            result["identity_pixel"][name] = pixel_stats(degraded, targets)
            for seed in restorers:
                result["pixel"][str(seed)][name] = pixel_stats(restored[seed], targets)
            tensors_identity = [pad_norm(image) for image in degraded]
            tensors_restored = {seed: [pad_norm(image) for image in restored[seed]]
                                for seed in restorers}
            for classifier, model in models.items():
                probs = probs_for(model, tensors_identity)
                summary, preds = score_of(probs, labels, n)
                if name == "clean":
                    clean_parity(classifier, classifiers[classifier]["path"], preds,
                                 rows, args.split, summary)
                    clean_preds_by_classifier[classifier] = preds
                clean_preds = clean_preds_by_classifier.get(classifier)
                if clean_preds is None:
                    raise RuntimeError("clean condition must be evaluated first")
                identity_summary = summarize(probs, labels, n, preds, clean_preds)
                identity_summary.update({"delta_accuracy_pp": 0.0,
                                         "delta_macro_f1_pp": 0.0})
                result["classifiers"][classifier]["identity"][name] = identity_summary
                if name in result["primary_conditions"]:
                    for row, label, pred, p in zip(rows, labels, preds, probs):
                        records.append({"classifier": classifier, "condition": name,
                                        "variant": "identity", "sample_id": row["sample_id"],
                                        "label": label, "prediction": pred,
                                        "probabilities": [round(float(v), 6) for v in p]})
                for seed in restorers:
                    probs_r = probs_for(model, tensors_restored[seed])
                    summary_r, preds_r = score_of(probs_r, labels, n)
                    clean_preds = clean_preds_by_classifier.get(classifier)
                    if clean_preds is None:
                        raise RuntimeError("clean condition must be evaluated first")
                    item = summarize(probs_r, labels, n, preds, clean_preds)
                    item.update({
                        "delta_accuracy_pp": 100 * (summary_r["accuracy"] - summary["accuracy"]),
                        "delta_macro_f1_pp": 100 * (summary_r["macro_f1"] - summary["macro_f1"]),
                    })
                    result["classifiers"][classifier]["restorers"].setdefault(
                        str(seed), {})[name] = item
                    if name in result["primary_conditions"]:
                        for row, label, pred, p in zip(rows, labels, preds_r, probs_r):
                            records.append({"classifier": classifier, "condition": name,
                                            "variant": f"pairunet_s{seed}",
                                            "sample_id": row["sample_id"], "label": label,
                                            "prediction": pred,
                                            "probabilities": [round(float(v), 6) for v in p]})
            print(json.dumps({"split": args.split, "condition": name,
                              "n": len(rows)}), flush=True)
    output = Path(args.output) if args.output else (
        PROJECT / f"reports/restorer_eval_{args.split}.json")
    write_json(output, result)
    if records:
        write_jsonl(output.with_suffix(".records.jsonl"), records)


if __name__ == "__main__":
    main()
