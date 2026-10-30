#!/usr/bin/env python
"""Train a paired supervised restorer on exact clean/degraded train-split pairs.

The restorer is a small residual U-Net trained with L1 + 0.1 edge L1 on
contained images (before letterbox padding).  Degradations come from the frozen
recipe, sampled per epoch in the same mixture family the classifiers see:
clean, single 1-5, combined 3/5, matched domain, random placement.  No
classifier, label, or test/validation image is used for training.

Validation pairs are fixed (clean, haze5, cloud5, glare5, haze+glare5,
haze+cloud5) and the checkpoint is selected by mean pixel L1 over the five
degraded conditions, not by any classification metric.
"""
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
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from maritime_robust.common import (ROOT as PROJECT, Budget, digest, environment,  # noqa: E402
                                    read_jsonl, seed_all, stable_seed, write_json)
from maritime_robust.degradations import RECIPE, corrupt, corruption_seed  # noqa: E402
from modules.pair_restorer import PairRestorer, pad_to_multiple  # noqa: E402

SINGLE_KINDS = ("haze", "cloud", "glare")
COMBINED_KINDS = ("haze+glare", "haze+cloud")
COMBINED_SEVERITIES = (3, 5)
VAL_CONDITIONS = (
    ("clean", "clean", 0),
    ("haze5", "haze", 5),
    ("cloud5", "cloud", 5),
    ("glare5", "glare", 5),
    ("haze+glare5", "haze+glare", 5),
    ("haze+cloud5", "haze+cloud", 5),
)


def content(row):
    with Image.open(PROJECT / row["path"]) as f:
        image = f.convert("RGB")
    return ImageOps.contain(image, (224, 224), Image.Resampling.BICUBIC)


def geometry(image, row, seed, epoch):
    rng = np.random.default_rng(stable_seed(seed, epoch, row["sample_id"], "restore_geometry"))
    if rng.random() < .5:
        image = ImageOps.mirror(image)
    if rng.random() < .5:
        image = ImageOps.flip(image)
    rotation = 90 * int(rng.integers(4))
    return image.rotate(rotation, expand=True)


def sample_condition(rng, clean_probability, single_share):
    if rng.random() < clean_probability:
        return "clean", 0
    if rng.random() < single_share:
        kind = SINGLE_KINDS[int(rng.integers(len(SINGLE_KINDS)))]
        severity = 1 + int(rng.integers(5))
    else:
        kind = COMBINED_KINDS[int(rng.integers(len(COMBINED_KINDS)))]
        severity = COMBINED_SEVERITIES[int(rng.integers(len(COMBINED_SEVERITIES)))]
    return kind, severity


class PairedTrainDataset(Dataset):
    def __init__(self, rows, seed, clean_probability, single_share):
        self.rows = rows
        self.seed = seed
        self.clean_probability = clean_probability
        self.single_share = single_share
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        clean = geometry(content(row), row, self.seed, self.epoch)
        rng = np.random.default_rng(stable_seed(self.seed, self.epoch, row["sample_id"],
                                                "restore_condition"))
        kind, severity = sample_condition(rng, self.clean_probability, self.single_share)
        if kind == "clean":
            degraded = clean
        else:
            seed = corruption_seed(row["sample_id"], f"restore_train_s{self.seed}_e{self.epoch}",
                                   kind, severity, "matched", 0)
            degraded, _ = corrupt(clean, seed=seed, kind=kind, severity=severity,
                                  domain="matched", placement="random")
        return degraded, clean


def val_pairs(rows):
    """Fixed validation pairs, cached once. Returns {condition: [(deg, clean)]}."""
    pairs = {name: [] for name, _, _ in VAL_CONDITIONS}
    for row in rows:
        clean = content(row)
        for name, kind, severity in VAL_CONDITIONS:
            if kind == "clean":
                degraded = clean
            else:
                seed = corruption_seed(row["sample_id"], "restore_val", kind, severity,
                                       "matched", 0)
                degraded, _ = corrupt(clean, seed=seed, kind=kind, severity=severity,
                                      domain="matched", placement="random")
            pairs[name].append((degraded, clean))
    return pairs


def collate_pairs(batch):
    height = max(image.height for image, _ in batch)
    width = max(image.width for image, _ in batch)
    height = ((height + 7) // 8) * 8
    width = ((width + 7) // 8) * 8
    degraded = torch.zeros(len(batch), 3, height, width)
    target = torch.zeros(len(batch), 3, height, width)
    mask = torch.zeros(len(batch), 1, height, width)
    for index, (image, clean) in enumerate(batch):
        d = TF.to_tensor(image)
        t = TF.to_tensor(clean)
        d = F.pad(d, (0, width - image.width, 0, height - image.height), mode="replicate")
        t = F.pad(t, (0, width - clean.width, 0, height - clean.height), mode="replicate")
        degraded[index] = d
        target[index] = t
        mask[index, :, :image.height, :image.width] = 1
    return degraded, target, mask


def masked_l1(output, target, mask):
    return ((output - target).abs() * mask).sum() / (mask.sum() * output.shape[1])


def edge_l1(output, target, mask):
    diff = output - target
    weight_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    weight_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    loss_x = (diff[:, :, :, 1:] - diff[:, :, :, :-1]).abs()
    loss_y = (diff[:, :, 1:, :] - diff[:, :, :-1, :]).abs()
    return ((loss_x * weight_x).sum() / (weight_x.sum() * output.shape[1]) +
            (loss_y * weight_y).sum() / (weight_y.sum() * output.shape[1]))


def tensor_l1(output, target, mask):
    error = (output - target).abs() * mask
    return (error.sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) * output.shape[1]))


@torch.inference_mode()
def validation_loss(model, pairs, device, batch_size):
    model.eval()
    result = {}
    for name, items in pairs.items():
        losses, edge_losses = [], []
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            degraded, target, mask = collate_pairs(batch)
            degraded, target, mask = degraded.to(device), target.to(device), mask.to(device)
            output = model(degraded).float().clamp(0, 1)
            losses.extend(tensor_l1(output, target, mask).tolist())
            edge_losses.append(float(edge_l1(output, target, mask)))
        result[name] = {"l1": float(np.mean(losses)),
                        "edge_l1": float(np.mean(edge_losses))}
    return result


def run_dir(args):
    root = PROJECT / "restorers"
    if args.limit:
        root = root / "smoke"
    return root / Path(args.manifest).stem / f"pairunet_s{args.seed}"


def train(args):
    config = json.loads((PROJECT / "configs/protocol.json").read_text())
    if json.loads((PROJECT / "configs/degradations.json").read_text()) != RECIPE:
        raise ValueError("Recipe configuration and implementation disagree")
    rows = read_jsonl(args.manifest)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = {r["source_group_id"] for r in rows if r["split"] == a} & \
                  {r["source_group_id"] for r in rows if r["split"] == b}
        if overlap and rows[0]["protocol"] == "research":
            raise ValueError(f"Leaking source groups: {a}/{b}")
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    if args.limit:
        train_rows = train_rows[:args.limit]
        val_rows = val_rows[:max(2, args.limit // 4)]
    if not train_rows or not val_rows:
        raise ValueError("Training and validation both required")
    out = run_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    settings = {
        "manifest": str(Path(args.manifest).resolve()), "manifest_sha256": digest(args.manifest),
        "module": "pair_restorer", "model": "PairRestorer",
        "model_config": {"base": args.base, "groups": 8},
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "workers": args.workers, "lr": args.lr, "weight_decay": args.weight_decay,
        "clean_probability": args.clean_probability, "single_share": args.single_share,
        "loss": "L1 + 0.1 * edge L1 (masked to content)",
        "selection": "mean validation L1 over five degraded fixed conditions",
        "train_n": len(train_rows), "val_n": len(val_rows),
        "recipe_sha256": digest(PROJECT / "configs/degradations.json"),
        "config_sha256": digest(PROJECT / "configs/protocol.json"),
        "environment": environment(), "smoke_only": bool(args.limit),
        "source_sha256": {str(p.relative_to(PROJECT)): digest(p) for p in
                          [PROJECT / "modules/pair_restorer.py", Path(__file__).resolve()]},
    }
    if (out / "complete.json").exists():
        completed = json.loads((out / "complete.json").read_text())
        for key in ("manifest_sha256", "epochs", "batch_size", "seed", "lr",
                    "clean_probability", "single_share", "model_config"):
            if completed.get(key) != settings.get(key):
                raise ValueError(f"Completed restorer has different {key}")
        print(json.dumps({"skipped": str(out), "reason": "already complete"}))
        return completed

    seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    dataset = PairedTrainDataset(train_rows, args.seed, args.clean_probability,
                                 args.single_share)
    pairs = val_pairs(val_rows)
    model = PairRestorer(base=args.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    history, best, best_record = [], -1.0, None
    with Budget("augmentation") as budget:
        for epoch in range(args.epochs):
            started = time.perf_counter()
            dataset.epoch = epoch
            loader = DataLoader(
                dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True,
                pin_memory=True, persistent_workers=False, collate_fn=collate_pairs,
                generator=torch.Generator().manual_seed(args.seed + epoch))
            model.train()
            loss_sum, count = 0.0, 0
            for degraded, target, mask in loader:
                degraded = degraded.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    output = model(degraded)
                output = output.float().clamp(0, 1)
                loss = masked_l1(output, target, mask) + 0.1 * edge_l1(output, target, mask)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss) * len(mask)
                count += len(mask)
                budget.check()
            val = validation_loss(model, pairs, device, max(8, args.batch_size * 2))
            degraded_val = [val[name]["l1"] for name, kind, _ in VAL_CONDITIONS if kind != "clean"]
            selection = float(np.mean(degraded_val))
            record = {"epoch": epoch + 1, "train_loss": loss_sum / max(1, count),
                      "val": val, "selection_l1": selection,
                      "lr": optimizer.param_groups[0]["lr"],
                      "seconds": time.perf_counter() - started}
            history.append(record)
            write_json(out / "history.json", history)
            if best < 0 or selection < best:
                best = selection
                best_record = record
                checkpoint = {"model": model.state_dict(), "settings": settings,
                              "epoch": epoch + 1, "val": val, "selection_l1": selection}
                tmp = out / "best.tmp.pt"
                torch.save(checkpoint, tmp)
                tmp.replace(out / "best.pt")
            scheduler.step()
            print(json.dumps({"run": str(out.relative_to(PROJECT)), "epoch": epoch + 1,
                              "train_loss": record["train_loss"],
                              "val_degraded_l1": selection,
                              "val_clean_l1": val["clean"]["l1"],
                              "seconds": record["seconds"]}), flush=True)
        elapsed = time.monotonic() - budget.started
        settings.update(best_epoch=best_record["epoch"], best_selection_l1=best,
                        best_val=best_record["val"], training_wall_seconds=elapsed,
                        parameter_count=sum(p.numel() for p in model.parameters()),
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                        checkpoint_sha256=digest(out / "best.pt"))
        write_json(out / "complete.json", settings)
    return settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/fgsc23_research_civilian.jsonl")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clean-probability", type=float, default=0.5)
    parser.add_argument("--single-share", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="smoke-only subset")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
