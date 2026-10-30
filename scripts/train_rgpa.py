#!/usr/bin/env python
"""Train and evaluate the RGPA variants under the frozen project recipe.

Only the model is new: dataset, augmentation, optimizer, schedule, checkpoint
selection and the 43 evaluation conditions are imported from the frozen engine
so that a `control` run is a direct parity check against the combined-augmentation
baseline. Runs live under runs_modules/ to keep the 45-run baseline accounting
and its artifacts untouched.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from maritime_robust.common import (ROOT as PROJECT, Budget, digest, environment,  # noqa: E402
                                    read_jsonl, seed_all, write_json, write_jsonl)
from maritime_robust.degradations import RECIPE, conditions  # noqa: E402
from maritime_robust.engine import (MODEL_WEIGHTS, ShipDataset, combined_recipe,  # noqa: E402
                                    build_model as build_engine, loader, predict)
from modules.rgpa import build_model as build_rgpa  # noqa: E402

SUFFIX = {"plain": "plain", "control": "control", "learned": "learned", "full": "gate"}


def run_dir(manifest, backbone, mode, seed, limit):
    tag = f"{Path(manifest).stem}/{backbone}_rgpa_{SUFFIX[mode]}_s{seed}"
    if limit:
        tag = "smoke/" + tag
    return PROJECT / "runs_modules" / tag


def build(backbone, num_classes, mode, pretrained):
    """`plain` is the frozen engine's own model: harness parity check."""
    if mode == "plain":
        return build_engine(backbone, num_classes, pretrained)
    return build_rgpa(backbone, num_classes, mode, pretrained)


def train(args):
    config = json.loads((PROJECT / "configs/protocol.json").read_text())
    if json.loads((PROJECT / "configs/degradations.json").read_text()) != RECIPE:
        raise ValueError("Recipe configuration and implementation disagree")
    epoch_count = args.epochs or config["epochs"]
    batch_size = args.batch_size or config["batch_size"]
    workers = config["workers"]
    rows = read_jsonl(args.manifest)
    if not rows:
        raise ValueError("Empty manifest")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = {r["source_group_id"] for r in rows if r["split"] == a} & \
                  {r["source_group_id"] for r in rows if r["split"] == b}
        if overlap and rows[0]["protocol"] == "research":
            raise ValueError(f"Leaking source groups: {a}/{b}")
    n = max(r["label"] for r in rows) + 1
    labels = {r["label"]: r["label_name"] for r in rows}
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    if args.limit:
        train_rows = train_rows[:args.limit]
        val_rows = val_rows[:max(2, args.limit // 4)]
    out = run_dir(args.manifest, args.backbone, args.mode, args.seed, args.limit)
    out.mkdir(parents=True, exist_ok=True)

    settings = {"manifest": str(Path(args.manifest).resolve()),
                "manifest_sha256": digest(args.manifest),
                "model": args.backbone, "module": "rgpa", "module_mode": args.mode,
                "augmentation": "combined", "seed": args.seed, "epochs": epoch_count,
                "batch_size": batch_size, "workers": workers, "classes": labels,
                "num_classes": n, "train_n": len(train_rows), "val_n": len(val_rows),
                "scope": rows[0]["scope"], "protocol": rows[0]["protocol"],
                "config_sha256": digest(PROJECT / "configs/protocol.json"),
                "recipe_sha256": digest(PROJECT / "configs/degradations.json"),
                "combined_recipe": combined_recipe(),
                "combined_recipe_sha256": digest(PROJECT / "configs/combined_augmentation.json"),
                "environment": environment(), "smoke_only": bool(args.limit),
                "source_sha256": {str(p.relative_to(PROJECT)): digest(p)
                                  for p in sorted((PROJECT / "src").rglob("*.py"))},
                "module_sha256": {str(p.relative_to(PROJECT)): digest(p)
                                  for p in sorted((PROJECT / "modules").rglob("*.py"))
                                  if "__pycache__" not in str(p)},
                "script_sha256": digest(Path(__file__).resolve())}
    if (out / "complete.json").exists():
        done = json.loads((out / "complete.json").read_text())
        if (done["manifest_sha256"], done["epochs"], done["module_mode"]) != \
                (settings["manifest_sha256"], epoch_count, args.mode):
            raise ValueError("Completed run has different settings; use another run dir")
        print(json.dumps({"skip": str(out.relative_to(PROJECT))}), flush=True)
        return done
    write_json(out / "config.json", settings)

    with Budget("augmentation") as budget:
        seed_all(args.seed)
        torch.set_num_threads(4)
        torch.hub.set_dir(str(PROJECT / "artifacts/weights"))
        model = build(args.backbone, n, args.mode, True).cuda()
        optimizer_config = config["optimizer"][args.backbone]
        kwargs = {k: v for k, v in optimizer_config.items() if k != "name"}
        optimizer = (torch.optim.SGD if optimizer_config["name"] == "sgd"
                     else torch.optim.AdamW)(model.parameters(), **kwargs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch_count)
        scaler = torch.amp.GradScaler("cuda")
        ds = ShipDataset(train_rows, train=True, augmentation="combined", seed=args.seed)
        history, best = [], -1
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(epoch_count):
            start = time.perf_counter()
            ds.epoch = epoch
            model.train()
            loss_sum = correct = 0
            for x, y, _, _ in loader(ds, batch_size, workers, shuffle=True, seed=args.seed + epoch):
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    output = model(x)
                    loss = F.cross_entropy(output, y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item() * len(y)
                correct += (output.argmax(1) == y).sum().item()
                budget.check()
            result, _ = predict(model, ShipDataset(val_rows), n, batch_size, workers, budget)
            record = {"epoch": epoch + 1, "train_loss": loss_sum / len(ds),
                      "train_accuracy": correct / len(ds), "val_macro_f1": result["macro_f1"],
                      "val_accuracy": result["accuracy"], "lr": optimizer.param_groups[0]["lr"],
                      "seconds": time.perf_counter() - start}
            history.append(record)
            write_json(out / "history.json", history)
            if result["macro_f1"] > best:
                best = result["macro_f1"]
                checkpoint = {"model": model.state_dict(), "settings": settings,
                              "epoch": epoch + 1, "validation": result}
                tmp = out / "best.tmp.pt"
                torch.save(checkpoint, tmp)
                tmp.replace(out / "best.pt")
            scheduler.step()
            print(json.dumps({"run": str(out.relative_to(PROJECT)), **record}), flush=True)
        gate = sum(p.numel() for name, p in model.named_parameters()
                   if name.startswith("rgpa.reliability") or name.startswith("rgpa.cue"))
        settings.update(best_val_macro_f1=best,
                        training_wall_seconds=time.monotonic() - budget.started,
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                        parameter_count=sum(p.numel() for p in model.parameters()),
                        gate_parameter_count=gate, last_epoch=history[-1],
                        checkpoint_sha256=digest(out / "best.pt"))
        try:
            weights = PROJECT / "artifacts/weights/checkpoints" / Path(MODEL_WEIGHTS[args.backbone].url).name
            settings["pretrained_sha256"] = digest(weights)
        except FileNotFoundError:
            settings["pretrained_sha256"] = None
        write_json(out / "complete.json", settings)
    return settings


def load(run):
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    settings = checkpoint["settings"]
    if digest(settings["manifest"]) != settings["manifest_sha256"]:
        raise ValueError("Manifest changed after training")
    if digest(PROJECT / "configs/degradations.json") != settings["recipe_sha256"]:
        raise ValueError("Degradation recipe changed after training")
    if digest(PROJECT / "configs/combined_augmentation.json") != settings["combined_recipe_sha256"]:
        raise ValueError("Combined augmentation recipe changed after training")
    model = build(settings["model"], settings["num_classes"], settings["module_mode"], False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.cuda().eval(), settings, read_jsonl(settings["manifest"])


def evaluate(run, split="test", validation_only=False):
    torch.set_num_threads(4)
    output = run / ("validation_robust" if split == "val" else "evaluation")
    output.mkdir(parents=True, exist_ok=True)
    with Budget("evaluation") as budget:
        model, settings, rows = load(run)
        rows = [r for r in rows if r["split"] == split]
        if not rows:
            raise ValueError("Empty evaluation split")
        conds = conditions(include_shifted=not validation_only)
        if validation_only:
            conds = [c for c in conds if c["kind"] == "clean" or
                     c["kind"] in ("haze", "cloud", "glare") and c["placement"] == "random"]
        summaries = {}
        for c in conds:
            path = output / (c["name"] + ".json")
            if path.exists():
                summaries[c["name"]] = json.loads(path.read_text())
                continue
            ds = ShipDataset(rows, condition=c)
            score, preds = predict(model, ds, settings["num_classes"], workers=4, budget=budget)
            score.update(condition=c, checkpoint_sha256=digest(run / "best.pt"),
                         manifest_sha256=settings["manifest_sha256"],
                         recipe_sha256=settings["recipe_sha256"], restoration=None,
                         module="rgpa", module_mode=settings["module_mode"])
            write_jsonl(output / (c["name"] + ".jsonl"), preds)
            write_json(path, score)
            summaries[c["name"]] = score
            print(json.dumps({"run": str(run.relative_to(PROJECT)), "condition": c["name"],
                              "macro_f1": score["macro_f1"], "accuracy": score["accuracy"]}),
                  flush=True)
        write_json(output / "summary.json", summaries)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--backbone", default="swin_t", choices=["swin_t", "convnext_tiny"])
    parser.add_argument("--mode", default="full",
                        choices=["plain", "control", "learned", "full"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", type=int, help="smoke subset of train samples")
    parser.add_argument("--eval", dest="do_eval", action="store_true", default=True)
    parser.add_argument("--no-eval", dest="do_eval", action="store_false")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    out = run_dir(args.manifest, args.backbone, args.mode, args.seed, args.limit)
    if not args.eval_only:
        train(args)
    if args.do_eval:
        evaluate(out, validation_only=args.validation_only)


if __name__ == "__main__":
    main()
