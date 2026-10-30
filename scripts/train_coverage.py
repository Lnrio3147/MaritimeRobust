"""Coverage training: the same augmentation budget, a different support over pairs.

Everything tried before this point assumed the compound deficit needs a
*structural* answer (identify the degradation, constrain the features, weight
per image). This arm tests the competing explanation: the deficit is what
happens when training never visits the operating point.

Nothing else changes. Same backbone, epochs, optimiser, schedule, batch size,
and — critically — the same *amount* of compound exposure: every sample is
clean with probability 0.5 and compound otherwise. The only variable is how many
distinct pairs the compound sampling is drawn from (``--k``). Three lattice pairs
are held out from every pool (``lattice.HELD_OUT``), and the paper's haze+glare
and haze+cloud compounds are never in the lattice at all, so any gain there is
transfer rather than fitting the tested condition.

Pre-registered in ``reports/PREREG_COVERAGE.md``; do not evaluate against a
different condition set than the one fixed there.
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
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from maritime_robust.common import (ROOT, Budget, digest, environment, read_jsonl,
                                    seed_all, stable_seed, write_json)  # noqa: E402
from maritime_robust.engine import ShipDataset, loader, predict  # noqa: E402
from maritime_robust.lattice import HELD_OUT, apply_family, apply_pair, pool  # noqa: E402
from maritime_robust.methods import build as build_method  # noqa: E402


class LatticeDataset(ShipDataset):
    """``ShipDataset`` with the augmentation support replaced by a fixed pair pool.

    ``ShipDataset.image`` is overridden whole rather than parameterised because
    the parent chooses a *severity* inside a fixed kind and then routes the draw
    through ``corrupt``; here the pair is chosen first and the two families are
    applied back to back. Geometry is unchanged, so the difference against the
    baseline recipe is confined to the degradation distribution.
    """

    def __init__(self, rows, pairs, seed=17, clean_probability=0.5, singles=False,
                 mix_base=False):
        # mix_base keeps the project's own combined recipe in the mix instead of
        # replacing it, so haze/glare/cloud stay trained and the haze+glare
        # comparison against the baseline becomes like-for-like.
        super().__init__(rows, train=True, augmented=False,
                         augmentation="combined" if mix_base else None,
                         seed=seed)
        self.pairs = list(pairs)
        self.mix_base = mix_base
        if not mix_base:
            self.clean_probability = clean_probability
        # s6 control: same six families, same compound/clean rate, one family per
        # draw instead of two. If this matches the pair arms, the gain is exposure
        # to new families and says nothing about pairing.
        self.singles = singles
        if singles:
            self.families = sorted({f for p in self.pairs for f in p})

    def image(self, index):
        row = self.rows[index]
        with Image.open(ROOT / row["path"]) as f:
            image = f.convert("RGB")
        image = ImageOps.contain(image, (224, 224), Image.Resampling.BICUBIC)
        rng = np.random.default_rng(stable_seed(self.seed, self.epoch, row["sample_id"], "geometry"))
        if rng.random() < .5:
            image = ImageOps.mirror(image)
        if rng.random() < .5:
            image = ImageOps.flip(image)
        image = image.rotate(90 * int(rng.integers(4)), expand=True)

        if self.mix_base:
            split = np.random.default_rng(
                stable_seed(self.seed, self.epoch, row["sample_id"], "lattice_mix"))
            if split.random() < 0.5:
                # Parent path: same geometry rng, so the two branches differ only
                # in the degradation drawn, never in the crop/flip.
                return super().image(index)

        aug = np.random.default_rng(stable_seed(self.seed, self.epoch, row["sample_id"], "lattice"))
        meta = {"kind": "clean", "severity": 0}
        if aug.random() >= self.clean_probability:
            if self.singles:
                family = self.families[int(aug.integers(len(self.families)))]
                image = apply_family(image, family, int(aug.integers(2**31 - 1)))
                meta = {"kind": family, "family_a": family, "severity_a": _level(family)}
            else:
                pair = self.pairs[int(aug.integers(len(self.pairs)))]
                image = apply_pair(image, pair, int(aug.integers(2**31 - 1)))
                meta = {"kind": f"{pair[0]}+{pair[1]}",
                        "family_a": pair[0], "family_b": pair[1],
                        "severity_a": _level(pair[0]), "severity_b": _level(pair[1])}
        image = ImageOps.pad(image, (224, 224), method=Image.Resampling.BICUBIC,
                             color=(124, 116, 104))
        return image, meta


def _level(family):
    from maritime_robust.lattice import MATCHED_LEVEL
    return MATCHED_LEVEL[family]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--model", default="convnext_tiny")
    ap.add_argument("--k", type=int, required=True, help="number of distinct pairs in the pool")
    ap.add_argument("--singles", action="store_true",
                    help="s6 control: same families, one per draw instead of a pair")
    ap.add_argument("--mix-base", action="store_true",
                    help="half the draws use the project combined recipe, half the lattice")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--suffix", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--phase", default="augmentation")
    ap.add_argument("--skip-budget", action="store_true")
    args = ap.parse_args()

    # The control always uses the full 12-pair family set so that it sees the
    # same six families as the widest arm; only the pairing is removed.
    pairs = pool(12 if args.singles else args.k)
    if len(pairs) != (12 if args.singles else args.k):
        raise SystemExit(f"pool has {len(pairs)} pairs, requested {args.k}")

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
    suffix = args.suffix or (
        f"_s6b_e40" if (args.singles and args.mix_base) else
        f"_s6_e40" if args.singles else
        f"_c{args.k}b_e40" if args.mix_base else f"_c{args.k}_e40")
    tag = f"{manifest.stem}/{args.model}_none{suffix}_s{args.seed}"
    out = ROOT / "runs" / tag
    if (out / "complete.json").exists():
        raise SystemExit(f"{out} already completed; pick a different --suffix")
    out.mkdir(parents=True, exist_ok=True)

    settings = {
        "manifest": str(manifest.resolve()), "manifest_sha256": digest(manifest),
        "model": args.model, "method": "none",
        "arm": (f"singles_6{'_mixbase' if args.mix_base else ''}" if args.singles else
                f"coverage_{args.k}_mixbase" if args.mix_base else f"coverage_{args.k}"),
        "mix_base": bool(args.mix_base),
        "coverage_k": 0 if args.singles else args.k,
        "coverage_singles": bool(args.singles),
        "coverage_pool": ["+".join(p) for p in pairs],
        "coverage_excluded": ["+".join(p) for p in HELD_OUT],
        "seed": args.seed, "epochs": args.epochs, "batch_size": batch_size,
        "workers": workers, "classes": labels, "num_classes": n,
        "augmented": True, "augmentation": "lattice",
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

        train_ds = LatticeDataset(train_rows, pairs, seed=args.seed,
                                  singles=args.singles, mix_base=args.mix_base)
        val_ds = ShipDataset(val_rows)

        best, history = -1, []
        for epoch in range(args.epochs):
            started = time.perf_counter()
            train_ds.epoch = epoch
            model.train()
            loss_sum = correct = 0
            for batch in loader(train_ds, batch_size, workers, shuffle=True, seed=args.seed + epoch):
                x, y = batch[0].cuda(non_blocking=True), batch[1].cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    raise SystemExit(f"non-finite loss at {tag} epoch {epoch}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                budget.check()
            result, _ = predict(model, val_ds, n, batch_size, workers, budget)
            record = {"epoch": epoch + 1, "train_loss": loss_sum / len(train_ds),
                      "train_accuracy": correct / len(train_ds),
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
