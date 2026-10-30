"""Evaluate one CIFAR checkpoint on the full single+pair grid.

Conditions are the 15 canonical corruptions at severities 1-5 (75 singles) and
every unordered pair of corruptions applied back-to-back at the same severity
(525 compounds), which is what the union decomposition needs: for each compound
the paired singles are already on the grid, so ``e(a+b) - |E_a u E_b|`` costs no
extra forward passes.

Only per-image correctness is stored (bit-packed), so the decomposition can be
recomputed or re-binned without touching the GPU again.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional as TF

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from maritime_robust.common import ROOT, Budget, stable_seed, write_json  # noqa: E402
from maritime_robust.external import cifar_c  # noqa: E402

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


class Conditioned(torch.utils.data.Dataset):
    """CIFAR test images under one fixed condition, corruption seeded per image.

    Takes the arrays rather than a dataset object: the evaluator walks 600
    conditions, and re-loading the 170 MB pickle per condition per worker
    dominates the cost otherwise. Forked workers share this memory.
    """

    def __init__(self, data, targets, condition, seed=0):
        self.data = data
        self.targets = targets
        self.kind, self.args = condition
        self.key = name(condition)
        self.seed = seed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        np.random.seed(stable_seed(self.seed, self.key, index) % (2**32))
        image = Image.fromarray(self.data[index])
        if self.kind == "clean":
            out = image
        elif self.kind == "single":
            out = Image.fromarray(cifar_c.apply(image, self.args[0], self.args[1]))
        else:
            out = Image.fromarray(cifar_c.apply_pair(image, self.args[:2], self.args[2]))
        tensor = TF.normalize(TF.to_tensor(out), MEAN, STD)
        return tensor, int(self.targets[index])


def conditions(which):
    """'pairs' also returns the singles, because every compound needs its two
    marginal error sets for the union baseline."""
    out = [("clean", ())]
    severities = cifar_c.SEVERITIES
    if which in ("all", "singles", "pairs"):
        out += [("single", (n, s)) for n in cifar_c.NAMES for s in severities]
    if which in ("all", "pairs"):
        out += [("pair", (*p, s)) for p in cifar_c.all_pairs() for s in severities]
    return out


def name(condition):
    kind, args = condition
    if kind == "clean":
        return "clean"
    if kind == "single":
        return f"{args[0]}_{args[1]}"
    return f"{args[0]}+{args[1]}_{args[2]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--which", default="all", choices=["all", "singles", "pairs", "clean"])
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-budget", action="store_true")
    args = ap.parse_args()

    run = ROOT / args.run_dir
    checkpoint = torch.load(run / "best.pt", map_location="cuda", weights_only=False)
    arch = checkpoint["settings"]["arch"]
    model = getattr(torchvision.models, arch)(num_classes=10).cuda()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    out_dir = Path(args.out) if args.out else PROJECT / "reports/cifar_grid" / run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = ROOT / "data/cifar"
    test = torchvision.datasets.CIFAR10(data_root, train=False, download=True)
    test_data, test_targets = np.asarray(test.data), np.asarray(test.targets)

    correct, accuracy = {}, {}

    class NoBudget:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def check(self, force=False):
            return None

    with (NoBudget() if args.skip_budget else Budget("cifar")) as budget:
        for condition in conditions(args.which):
            started = time.perf_counter()
            ds = Conditioned(test_data, test_targets, condition)
            if args.limit:
                ds = torch.utils.data.Subset(ds, range(args.limit))
            loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                                 num_workers=args.workers,
                                                 pin_memory=True)
            hits, total = [], 0
            with torch.no_grad():
                for x, y in loader:
                    pred = model(x.cuda(non_blocking=True)).argmax(1).cpu()
                    hits.append((pred == y).numpy())
                    total += len(y)
                    budget.check()
            flags = np.concatenate(hits)
            key = name(condition)
            correct[key] = np.packbits(flags)
            accuracy[key] = float(flags.mean())
            print(json.dumps({"run": run.name, "condition": key,
                              "accuracy": accuracy[key],
                              "seconds": time.perf_counter() - started}), flush=True)
            if len(correct) % 10 == 0:
                np.savez_compressed(out_dir / "correct.npz", **correct)
                write_json(out_dir / "summary.json",
                           {"run": run.name, "arch": arch, "accuracy": accuracy,
                            "n": total, "conditions": len(correct)})
    np.savez_compressed(out_dir / "correct.npz", **correct)
    write_json(out_dir / "summary.json",
               {"run": run.name, "arch": arch, "accuracy": accuracy,
                "n": len(test_data), "conditions": len(correct)})
    print(json.dumps({"done": run.name, "conditions": len(correct)}))


if __name__ == "__main__":
    main()
