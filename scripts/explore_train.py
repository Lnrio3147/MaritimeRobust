#!/usr/bin/env python
"""Train one S1 candidate on one fold with one seed.

Mirrors ``engine.train`` (same dataset, augmentation recipe, optimizer config,
cosine schedule, best-by-val-macro-f1 checkpointing) and adds exactly one
intervention, selected by ``--method``:

    none  plain ConvNeXt-T + combined augmentation (the S1 reference arm)
    r3    residual feature-denoising block before the classifier
    i1    degradation-conditioned FiLM normalisation of the pooled feature
    t1    composition consistency: KL(p(single) || p(combo)), single stop-grad

Nothing in the frozen pipeline is modified; this is a separate entry point.

    python scripts/explore_train.py --fold 0 --seed 17 --method r3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maritime_robust.common import (ROOT as PROJECT, Budget, digest, environment,  # noqa: E402
                                    read_jsonl, seed_all, stable_seed, write_json)
from maritime_robust.degradations import corrupt, corruption_seed  # noqa: E402
from maritime_robust.engine import (ShipDataset, combined_recipe, loader,  # noqa: E402
                                    predict)
from maritime_robust.methods import build as build_method, load_imagenet22k  # noqa: E402
from maritime_robust.methods.augment import KINDS, AugDataset, MaskedClean  # noqa: E402

MEAN, STD, FILL = [.485, .456, .406], [.229, .224, .225], (124, 116, 104)
PAIRS = [("haze", "glare"), ("haze", "cloud")]


class PairDataset(Dataset):
    """Two views of the same image: one single degradation, one composed with it."""

    def __init__(self, rows, seed=17):
        self.rows = rows
        self.seed = seed
        self.epoch = 0
        self.recipe = combined_recipe()
        self.clean_probability = self.recipe["clean_probability"]

    def __len__(self):
        return len(self.rows)

    def view(self, image, sample_id, kind, severity):
        seed = corruption_seed(sample_id, f"train_epoch_{self.epoch}", kind, severity,
                               "matched", self.seed)
        img, _ = corrupt(image, seed=seed, kind=kind, severity=severity,
                         domain="matched", placement="random")
        img = ImageOps.pad(img, (224, 224), method=Image.Resampling.BICUBIC, color=FILL)
        return TF.normalize(TF.to_tensor(img), MEAN, STD)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(PROJECT / row["path"]) as handle:
            image = handle.convert("RGB")
        image = ImageOps.contain(image, (224, 224), Image.Resampling.BICUBIC)
        # identical geometry draws to ShipDataset.image()
        rng = np.random.default_rng(stable_seed(self.seed, self.epoch, row["sample_id"], "geometry"))
        if rng.random() < .5:
            image = ImageOps.mirror(image)
        if rng.random() < .5:
            image = ImageOps.flip(image)
        image = image.rotate(90 * int(rng.integers(4)), expand=True)
        rng = np.random.default_rng(stable_seed(self.seed, self.epoch, row["sample_id"], "augmentation"))
        if rng.random() < self.clean_probability:
            first, second, severity = "clean", None, 0
        else:
            first, second = PAIRS[int(rng.integers(len(PAIRS)))]
            severity = int(rng.choice(self.recipe["combined_severities"]))
        combo = self.view(image, row["sample_id"],
                          first if second is None else f"{first}+{second}", severity)
        single = self.view(image, row["sample_id"], first, severity)
        return combo, single, row["label"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--method", default="none",
                    choices=("none", "r3", "i1", "t1", "i2", "r1", "t2", "t3", "m1"))
    ap.add_argument("--model", default="convnext_tiny")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--lambda-consistency", type=float, default=0.5)
    ap.add_argument("--suffix", default="", help="appended to the run name, e.g. _e40")
    ap.add_argument("--init", default="imagenet1k", choices=("imagenet1k", "timm22k"),
                    help="timm22k = convnext_tiny.fb_in22k_ft_in1k, initialization control only")
    ap.add_argument("--limit", type=int, default=None)
    # Budget's phase table is fixed; screening is augmentation work and fits
    # inside the existing augmentation allowance (24 h).
    ap.add_argument("--phase", default="augmentation")
    args = ap.parse_args()

    manifest = PROJECT / f"data/manifests/folds/fgsc23_civ8_fold{args.fold}.jsonl"
    config = json.loads((ROOT / "configs/protocol.json").read_text())
    rows = read_jsonl(manifest)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = ({r["source_group_id"] for r in rows if r["split"] == a}
                   & {r["source_group_id"] for r in rows if r["split"] == b})
        if overlap:
            raise ValueError(f"Leaking source groups: {a}/{b}")
    n = max(r["label"] for r in rows) + 1
    labels = {r["label"]: r["label_name"] for r in rows}
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    if args.limit:
        train_rows = train_rows[:args.limit]
        val_rows = val_rows[:max(2, args.limit // 4)]

    batch_size = config["batch_size"]
    workers = config["workers"]
    init_tag = "_i22k" if args.init == "timm22k" else ""
    tag = f"{manifest.stem}/{args.model}_{args.method}{args.suffix}{init_tag}_s{args.seed}"
    if args.limit:
        tag = "smoke/" + tag
    out = ROOT / "runs" / tag
    if (out / "complete.json").exists():
        raise SystemExit(f"{out} already completed; pick a different --suffix")
    out.mkdir(parents=True, exist_ok=True)

    settings = {
        "manifest": str(manifest.resolve()), "manifest_sha256": digest(manifest),
        "model": args.model, "method": args.method, "init": args.init,
        "seed": args.seed, "epochs": args.epochs,
        "batch_size": batch_size, "workers": workers, "classes": labels, "num_classes": n,
        "augmented": True, "augmentation": "combined", "train_n": len(train_rows),
        "val_n": len(val_rows), "scope": rows[0]["scope"], "protocol": rows[0]["protocol"],
        "config_sha256": digest(ROOT / "configs/protocol.json"),
        "recipe_sha256": digest(ROOT / "configs/degradations.json"),
        "combined_recipe": combined_recipe(),
        "combined_recipe_sha256": digest(ROOT / "configs/combined_augmentation.json"),
        "lambda_consistency": args.lambda_consistency if args.method == "t1" else None,
        "environment": environment(), "smoke_only": bool(args.limit),
    }
    settings["source_sha256"] = {str(p.relative_to(ROOT)): digest(p)
                                 for p in sorted((ROOT / "src").rglob("*.py"))}
    write_json(out / "config.json", settings)

    with Budget(args.phase) as budget:
        seed_all(args.seed)
        torch.set_num_threads(4)
        torch.hub.set_dir(str(ROOT / "artifacts/weights"))
        model = build_method(args.model, n, args.method)
        if args.init == "timm22k":
            settings["init"] = "timm22k"
            settings["init_transfer"] = load_imagenet22k(model)
            write_json(out / "config.json", settings)
        model = model.cuda()
        opt_config = config["optimizer"][args.model]
        kwargs = {k: v for k, v in opt_config.items() if k != "name"}
        optimizer = (torch.optim.SGD if opt_config["name"] == "sgd" else torch.optim.AdamW)(
            model.parameters(), **kwargs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        scaler = torch.amp.GradScaler("cuda")
        if args.method == "t1":
            ds = PairDataset(train_rows, seed=args.seed)
        elif args.method in ("t2", "t3", "m1"):
            ds = AugDataset(train_rows, seed=args.seed, curriculum=args.method == "t3",
                            total_epochs=args.epochs, with_mask=args.method == "m1")
        else:
            ds = ShipDataset(train_rows, train=True, augmentation="combined", seed=args.seed)
        val_ds = MaskedClean(val_rows) if args.method == "m1" else ShipDataset(val_rows)
        # T2: group DRO weights over the six degradation groups
        q = torch.full((len(KINDS),), 1.0 / len(KINDS), device="cuda")
        eta, scale = 0.05, 1.0
        best, history = -1, []
        for epoch in range(args.epochs):
            started = time.perf_counter()
            ds.epoch = epoch
            model.train()
            loss_sum = correct = 0
            for batch in loader(ds, batch_size, workers, shuffle=True, seed=args.seed + epoch):
                kind = None
                if args.method == "t1":
                    x, single, y = batch
                    single = single.cuda(non_blocking=True)
                elif args.method in ("t2", "t3", "m1"):
                    x, y, kind = batch
                    kind = kind.cuda(non_blocking=True)
                else:
                    x, y = batch[0], batch[1]
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(x)
                    if args.method == "t2":
                        per_sample = F.cross_entropy(logits.float(), y, reduction="none")
                        with torch.no_grad():
                            scale = 0.9 * scale + 0.1 * per_sample.mean().item()
                            for k in range(len(KINDS)):
                                mask = kind == k
                                if mask.any():
                                    q[k] *= math.exp(eta * per_sample[mask].mean().item()
                                                     / max(scale, 1e-3))
                            q /= q.sum()
                        loss = (per_sample * q[kind]).mean()
                    else:
                        loss = F.cross_entropy(logits, y)
                    if args.method == "t1":
                        with torch.no_grad():
                            teacher = model(single).float().softmax(1)
                        student = logits.float().log_softmax(1)
                        loss = loss + args.lambda_consistency * F.kl_div(
                            student, teacher, reduction="batchmean")
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                budget.check()
            result, _ = predict(model, val_ds, n, batch_size, workers, budget)
            record = {"epoch": epoch + 1, "train_loss": loss_sum / len(ds),
                      "train_accuracy": correct / len(ds), "val_macro_f1": result["macro_f1"],
                      "val_accuracy": result["accuracy"], "lr": optimizer.param_groups[0]["lr"],
                      "seconds": time.perf_counter() - started}
            history.append(record)
            write_json(out / "history.json", history)
            if result["macro_f1"] > best:
                best = result["macro_f1"]
                tmp = out / "best.tmp.pt"
                torch.save({"model": model.state_dict(), "settings": settings,
                            "epoch": epoch + 1, "validation": result}, tmp)
                tmp.replace(out / "best.pt")
            scheduler.step()
            print(json.dumps({"run": tag, **record}), flush=True)
        elapsed = time.monotonic() - budget.started
        plain = sum(p.numel() for p in build_method(args.model, n, "none").parameters())
        settings.update(best_val_macro_f1=best, training_wall_seconds=elapsed,
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                        parameter_count=sum(p.numel() for p in model.parameters()),
                        parameter_delta_vs_none=sum(p.numel() for p in model.parameters()) - plain,
                        checkpoint_sha256=digest(out / "best.pt"))
        write_json(out / "complete.json", settings)
    print(json.dumps({"done": tag, "best_val_macro_f1": best}))


if __name__ == "__main__":
    main()
