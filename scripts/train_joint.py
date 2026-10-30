#!/usr/bin/env python
"""Joint-training ablations C1/C2 under the frozen recipe.

C1  localized augmentation: same combined recipe, but degraded samples get
    placement sampled from configs/localized_augmentation.json.
C2  clean/degraded prediction consistency: CE on the degraded view plus a
    temperature-scaled KL from a detached clean-view teacher. The clean view
    uses the same geometry (geometry rng is independent of augmentation).
plain  frozen engine model and combined recipe, harness parity check.

Runs live under runs_joint/ and do not touch the 45 baseline runs.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maritime_robust.common import (ROOT as PROJECT, Budget, digest, environment,  # noqa: E402
                                    read_jsonl, seed_all, write_json, write_jsonl)
from maritime_robust.degradations import RECIPE, conditions  # noqa: E402
from maritime_robust.engine import (MODEL_WEIGHTS, ShipDataset, combined_recipe,  # noqa: E402
                                    build_model, loader, predict)

MODES = {"plain": "plain", "c1": "localized", "c2": "consistency", "c3": "feature"}
PLACEMENT_PATH = PROJECT / "configs/localized_augmentation.json"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class LocalizedDataset(ShipDataset):
    def __init__(self, *args, placement_mixture=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not placement_mixture:
            raise ValueError("placement mixture required")
        self.placement_mixture = placement_mixture

    def augmentation_condition(self, rng):
        c = super().augmentation_condition(rng)
        names = list(self.placement_mixture)
        p = np.array([self.placement_mixture[k] for k in names], dtype=float)
        c["placement"] = str(rng.choice(names, p=p / p.sum()))
        return c


def run_dir(manifest, backbone, mode, seed, limit):
    tag = f"{Path(manifest).stem}/{backbone}_joint_{MODES[mode]}_s{seed}"
    if limit:
        tag = "smoke/" + tag
    return PROJECT / "runs_joint" / tag


def make_dataset(rows, mode, seed, placement, train=True, clean_view=False):
    if mode == "c1" and not clean_view:
        return LocalizedDataset(rows, train=train, augmentation="combined", seed=seed,
                                placement_mixture=placement)
    if clean_view:
        return ShipDataset(rows, train=train, augmentation="clean", seed=seed)
    return ShipDataset(rows, train=train, augmentation="combined", seed=seed)


def val_diagnostics(records, n):
    probs = np.array([r["probabilities"] for r in records])
    ent = float(np.mean(-(probs * np.log(np.clip(probs, 1e-12, 1))).sum(1)))
    hist = np.bincount([r["prediction"] for r in records], minlength=n).tolist()
    return {"val_mean_entropy": ent, "val_mean_max_prob": float(probs.max(1).mean()),
            "val_pred_hist": hist}


def train(args):
    config = json.loads((PROJECT / "configs/protocol.json").read_text())
    if json.loads((PROJECT / "configs/degradations.json").read_text()) != RECIPE:
        raise ValueError("Recipe configuration and implementation disagree")
    placement = json.loads(PLACEMENT_PATH.read_text())["placement_mixture"] if args.mode == "c1" else None
    epoch_count = args.epochs or config["epochs"]
    batch_size = args.batch_size or config["batch_size"]
    workers = config["workers"]
    rows = read_jsonl(args.manifest)
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
    settings = {
        "manifest": str(Path(args.manifest).resolve()), "manifest_sha256": digest(args.manifest),
        "model": args.backbone, "module": "joint", "module_mode": args.mode,
        "augmentation": "combined", "seed": args.seed, "epochs": epoch_count,
        "batch_size": batch_size, "workers": workers, "classes": labels, "num_classes": n,
        "train_n": len(train_rows), "val_n": len(val_rows), "scope": rows[0]["scope"],
        "protocol": rows[0]["protocol"],
        "config_sha256": digest(PROJECT / "configs/protocol.json"),
        "recipe_sha256": digest(PROJECT / "configs/degradations.json"),
        "combined_recipe": combined_recipe(),
        "combined_recipe_sha256": digest(PROJECT / "configs/combined_augmentation.json"),
        "placement_mixture": placement,
        "placement_config_sha256": digest(PLACEMENT_PATH) if placement else None,
        "consistency_lambda": args.lambda_consistency if args.mode == "c2" else None,
        "consistency_temperature": args.temperature if args.mode == "c2" else None,
        "feature_lambda": args.lambda_feature if args.mode == "c3" else None,
        "feature_layer": args.feature_layer if args.mode == "c3" else None,
        "environment": environment(), "smoke_only": bool(args.limit),
        "source_sha256": {str(p.relative_to(PROJECT)): digest(p)
                          for p in sorted((PROJECT / "src").rglob("*.py"))},
        "script_sha256": digest(Path(__file__).resolve()),
    }
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
        model = build_model(args.backbone, n, True).cuda()
        opt_config = config["optimizer"][args.backbone]
        kwargs = {k: v for k, v in opt_config.items() if k != "name"}
        optimizer = (torch.optim.SGD if opt_config["name"] == "sgd"
                     else torch.optim.AdamW)(model.parameters(), **kwargs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch_count)
        scaler = torch.amp.GradScaler("cuda")
        ds = make_dataset(train_rows, args.mode, args.seed, placement)
        args_clean = args.mode in ("c2", "c3")
        ds_clean = (make_dataset(train_rows, args.mode, args.seed, placement, clean_view=True)
                    if args_clean else None)
        capture, state = {}, {"view": "deg"}
        if args.mode == "c3":
            layer = model.features[args.feature_layer]
            layer.register_forward_hook(
                lambda module, inputs, output: capture.__setitem__(state["view"], output))
        mean_t = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
        std_t = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)
        history, best, consistency_seen, feature_seen = [], -1, 0.0, 0.0
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(epoch_count):
            start = time.perf_counter()
            ds.epoch = epoch
            if ds_clean is not None:
                ds_clean.epoch = epoch
            model.train()
            loss_sum = correct = cons_sum = feat_sum = 0
            load_deg = loader(ds, batch_size, workers, shuffle=True, seed=args.seed + epoch)
            load_clean = (loader(ds_clean, batch_size, workers, shuffle=True,
                                 seed=args.seed + epoch) if ds_clean is not None else None)
            iter_clean = iter(load_clean) if load_clean is not None else None
            for batch in load_deg:
                x, y = batch[0].cuda(non_blocking=True), batch[1].cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                state["view"] = "deg"
                with torch.autocast("cuda", dtype=torch.float16):
                    output = model(x)
                    loss = F.cross_entropy(output, y)
                if iter_clean is not None:
                    (xc, yc) = next(iter_clean)[:2]
                    if not torch.equal(y, yc.cuda()):
                        raise RuntimeError("clean/degraded label order mismatch")
                    state["view"] = "clean"
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                        teacher = model(xc.cuda(non_blocking=True))
                    if args.mode == "c2":
                        t = args.temperature
                        cons = F.kl_div(F.log_softmax(output.float() / t, 1),
                                        F.softmax(teacher.float() / t, 1).detach(),
                                        reduction="batchmean") * t * t
                        loss = loss + args.lambda_consistency * cons
                        cons_sum += cons.item() * len(y)
                    else:
                        fd, fc = capture["deg"], capture["clean"]
                        if fd.shape[-1] == fc.shape[-1] and fd.shape[1] != fd.shape[-1]:
                            fd, fc = fd.permute(0, 3, 1, 2), fc.permute(0, 3, 1, 2)
                        rgb = x.float() * std_t + mean_t
                        sat = (rgb.max(1).values >= 0.98).float().unsqueeze(1)
                        weight = 1 - F.adaptive_avg_pool2d(sat, fd.shape[-2:])
                        cos = (F.normalize(fd.float(), dim=1) *
                               F.normalize(fc.float(), dim=1)).sum(1)
                        feat_loss = (weight[:, 0] * (1 - cos)).sum() / \
                            weight.sum().clamp_min(1e-6)
                        loss = loss + args.lambda_feature * feat_loss
                        feat_sum += feat_loss.item() * len(y)
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
            result, records = predict(model, ShipDataset(val_rows), n, batch_size, workers, budget)
            record = {"epoch": epoch + 1, "train_loss": loss_sum / len(ds),
                      "train_accuracy": correct / len(ds),
                      "val_macro_f1": result["macro_f1"], "val_accuracy": result["accuracy"],
                      "lr": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter() - start}
            if iter_clean is not None:
                if args.mode == "c2":
                    record["consistency_loss"] = cons_sum / len(ds)
                    consistency_seen = record["consistency_loss"]
                else:
                    record["feature_loss"] = feat_sum / len(ds)
                    feature_seen = record["feature_loss"]
                record.update(val_diagnostics(records, n))
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
        settings.update(best_val_macro_f1=best,
                        training_wall_seconds=time.monotonic() - budget.started,
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                        parameter_count=sum(p.numel() for p in model.parameters()),
                        final_consistency_loss=consistency_seen,
                        final_feature_loss=feature_seen, last_epoch=history[-1],
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
    model = build_model(settings["model"], settings["num_classes"], False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.cuda().eval(), settings, read_jsonl(settings["manifest"])


def evaluate(run, split="test", validation_only=False):
    torch.set_num_threads(4)
    output = run / ("validation_robust" if split == "val" else "evaluation")
    output.mkdir(parents=True, exist_ok=True)
    with Budget("evaluation") as budget:
        model, settings, rows = load(run)
        rows = [r for r in rows if r["split"] == split]
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
                         module="joint", module_mode=settings["module_mode"])
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
    parser.add_argument("--mode", default="c1", choices=list(MODES))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--lambda-consistency", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--lambda-feature", type=float, default=0.5)
    parser.add_argument("--feature-layer", type=int, default=5)
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
