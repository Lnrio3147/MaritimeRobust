"""Additivity-constrained training: an objective that never has to identify the degradation.

Every intervention that failed before needed to know *which* degradation it was
looking at, either at test time (correction, abstention) or per image (drift
weighting). This arm does not. It takes the one structural fact the interaction
study established — accuracy-space interaction is additive over families — and
asks the network to satisfy its feature-space counterpart:

    phi(a+b) - phi(a) - phi(b) + phi(clean)  ->  0

The probe in ``scripts/additivity_probe.py`` shows this residual is large on the
baseline (relative size ~0.87 for haze+glare at severity 5), so the constraint
has something to act on. It does *not* predict per-image failure, so this is a
training-time inductive bias and nothing more.

Both arms see identical data; only ``lambda`` differs:

    a0   control: CE(combined-aug view) + CE(haze+glare_5 view)
    a1   + lambda * || residual ||^2 / (||shift_a|| + ||shift_b||)^2

The a0 arm exists so that a gain cannot be attributed to mere extra exposure to
the compound condition — that confound sank the sampling route (commit 71d6b69).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from maritime_robust.common import (ROOT, Budget, digest, environment, read_jsonl,
                                    seed_all, stable_seed, write_json)  # noqa: E402
from maritime_robust.engine import ShipDataset, loader, predict  # noqa: E402
from maritime_robust.methods import build as build_method  # noqa: E402

PAIR = ("haze", "glare")


def quad_datasets(rows):
    """clean / a / b / a+b over the same rows with the same (absent) geometry aug.

    ``ShipDataset`` only honours a fixed ``condition`` when ``train=False``; that
    is exactly what we want here, because the four views must differ in the
    degradation and in nothing else.
    """
    def cond(kind):
        return {"kind": kind, "severity": 5, "domain": "matched", "placement": "random"}
    return [ShipDataset(rows, train=False, condition=cond(k)) for k in
            ("clean", PAIR[0], PAIR[1], f"{PAIR[0]}+{PAIR[1]}")]


def feature_and_logits(model, x):
    f = model.features_of(x)
    return f, model.head(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--model", default="convnext_tiny")
    ap.add_argument("--arm", choices=("a0", "a1"), default="a1")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lam", type=float, default=0.1, help="a1 only: weight on the residual term")
    ap.add_argument("--suffix", default="_e40")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--phase", default="augmentation")
    ap.add_argument("--skip-budget", action="store_true",
                    help="smoke only: run without taking the GPU lock")
    args = ap.parse_args()

    manifest = PROJECT / f"data/manifests/folds/fgsc23_civ8_fold{args.fold}.jsonl"
    config = json.loads((ROOT / "configs/protocol.json").read_text())
    rows = read_jsonl(manifest)
    n = max(r["label"] for r in rows) + 1
    labels = {r["label"]: r["label_name"] for r in rows}
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    if args.limit:
        train_rows = train_rows[:args.limit]
        val_rows = val_rows[:max(2, args.limit // 4)]

    batch_size, workers = config["batch_size"], config["workers"]
    tag = f"{manifest.stem}/{args.model}_{args.arm}{args.suffix}_s{args.seed}"
    out = ROOT / "runs" / tag
    if (out / "complete.json").exists():
        raise SystemExit(f"{out} already completed; pick a different --suffix")
    out.mkdir(parents=True, exist_ok=True)

    settings = {
        "manifest": str(manifest.resolve()), "manifest_sha256": digest(manifest),
        # Recorded as the plain backbone so every existing eval script loads it
        # unchanged; the arm is carried separately.
        "model": args.model, "method": "none", "arm": args.arm,
        "lambda_additivity": args.lam if args.arm == "a1" else 0.0,
        "quad_pair": list(PAIR), "quad_severity": 5,
        "seed": args.seed, "epochs": args.epochs, "batch_size": batch_size,
        "workers": workers, "classes": labels, "num_classes": n,
        "augmented": True, "augmentation": "combined",
        "train_n": len(train_rows), "val_n": len(val_rows),
        "scope": rows[0]["scope"], "protocol": rows[0]["protocol"],
        "config_sha256": digest(ROOT / "configs/protocol.json"),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p)
                          for p in sorted((ROOT / "src").rglob("*.py"))},
        "environment": environment(),
    }
    write_json(out / "config.json", settings)

    if args.skip_budget and not args.limit:
        raise SystemExit("--skip-budget is only allowed with --limit (smoke runs)")

    class NoBudget:
        def __enter__(self):
            self.started = time.monotonic()
            return self

        def __exit__(self, *exc):
            return False

        def check(self, force=False):
            return None

    with (NoBudget() if args.skip_budget else Budget(args.phase)) as budget:
        seed_all(args.seed)
        torch.set_num_threads(4)
        torch.hub.set_dir(str(ROOT / "artifacts/weights"))
        model = build_method(args.model, n, "none").cuda()
        opt_config = config["optimizer"][args.model]
        kwargs = {k: v for k, v in opt_config.items() if k != "name"}
        optimizer = (torch.optim.SGD if opt_config["name"] == "sgd" else torch.optim.AdamW)(
            model.parameters(), **kwargs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        scaler = torch.amp.GradScaler("cuda")

        aug_ds = ShipDataset(train_rows, train=True, augmentation="combined", seed=args.seed)
        quads = quad_datasets(train_rows)
        val_ds = ShipDataset(val_rows)

        def permuted(epoch):
            """Rebuild every stream over the same row order.

            The five streams are zipped, so they must be index-aligned: shuffling
            one loader and not the others silently pairs an image with another
            image's label. Permuting the row list *before* constructing the
            datasets keeps the shuffle without breaking alignment.
            """
            order = np.random.default_rng(stable_seed(args.seed, epoch, "row_order"))
            rows_e = [train_rows[i] for i in order.permutation(len(train_rows))]
            return (ShipDataset(rows_e, train=True, augmentation="combined", seed=args.seed),
                    quad_datasets(rows_e))

        best, history = -1, []
        checked = False
        for epoch in range(args.epochs):
            started = time.perf_counter()
            aug_ds, quads = permuted(epoch)
            aug_ds.epoch = epoch
            model.train()
            loss_sum = ce_sum = int_sum = correct = 0
            quad_loaders = [loader(d, batch_size, workers, shuffle=False) for d in quads]
            for batch, *quad_batches in zip(loader(aug_ds, batch_size, workers, shuffle=False),
                                            *quad_loaders):
                x, y = batch[0].cuda(non_blocking=True), batch[1].cuda(non_blocking=True)
                f_clean, f_a, f_b, f_ab = [qb[0].cuda(non_blocking=True) for qb in quad_batches]
                if epoch == 0 and not checked:
                    # The quad CE terms reuse y from the augmented stream, so the
                    # five streams must be index-aligned. Fail loudly if not.
                    for qb in quad_batches:
                        if not torch.equal(qb[1], batch[1]):
                            raise SystemExit("stream misalignment: quad labels != aug labels")
                    checked = True
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(x)
                    ce_aug = F.cross_entropy(logits, y)
                    fa, la = feature_and_logits(model, f_a)
                    fb, lb = feature_and_logits(model, f_b)
                    fab, lab = feature_and_logits(model, f_ab)
                    f0, l0 = feature_and_logits(model, f_clean)
                    ce_ab = (F.cross_entropy(la, y) + F.cross_entropy(lb, y)
                             + F.cross_entropy(lab.float(), y)) / 3
                    loss = ce_aug + ce_ab
                    int_val = torch.zeros((), device=x.device)
                    if args.arm == "a1":
                        # Work in float32: the residual is a difference of large,
                        # nearly equal vectors and fp16 rounding alone would put
                        # a floor under it.
                        fa, fb, fab, f0 = fa.float(), fb.float(), fab.float(), f0.float()
                        da, db = fb - f0, fa - f0
                        residual = fab - fa - fb + f0
                        scale = (da.norm(dim=1) + db.norm(dim=1)).clamp_min(1e-6)
                        int_val = (residual.norm(dim=1) / scale).pow(2).mean()
                        loss = loss + args.lam * int_val
                if not torch.isfinite(loss):
                    raise SystemExit(f"non-finite loss at {tag} epoch {epoch}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item() * len(y)
                ce_sum += (ce_aug.item() + ce_ab.item()) * len(y)
                int_sum += float(int_val) * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                budget.check()
            result, _ = predict(model, val_ds, n, batch_size, workers, budget)
            record = {"epoch": epoch + 1, "train_loss": loss_sum / len(aug_ds),
                      "train_accuracy": correct / len(aug_ds),
                      "ce": ce_sum / len(aug_ds), "interaction": int_sum / len(aug_ds),
                      "val_macro_f1": result["macro_f1"], "val_accuracy": result["accuracy"],
                      "lr": optimizer.param_groups[0]["lr"],
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
        settings.update(best_val_macro_f1=best,
                        training_wall_seconds=time.monotonic() - budget.started,
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                        parameter_count=sum(p.numel() for p in model.parameters()),
                        checkpoint_sha256=digest(out / "best.pt"))
        write_json(out / "complete.json", settings)
    print(json.dumps({"done": tag, "best_val_macro_f1": best}))


if __name__ == "__main__":
    main()
