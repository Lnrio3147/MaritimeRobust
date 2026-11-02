"""CIFAR-10 training with a selectable degradation recipe.

Purpose is the same decomposition as the maritime study, on the standard
benchmark: ``--aug clean`` gives the reference model whose 600-condition
single+pair grid is the observational study; ``--aug lattice`` gives the
intervention arms that test whether covering pairs removes the compound excess.

The clean model follows the usual CIFAR recipe (SGD, momentum 0.9, weight decay
5e-4, cosine schedule, 4-pixel pad-and-crop, horizontal flip) so its accuracy is
comparable to published ResNet-18 numbers rather than to a deliberately weak or
strong variant. Corruption arms add the degradation to the *training* draws at a
fixed probability, in place of nothing; the geometry transform is unchanged.

    --aug otheraug  applies photometric and geometric degradations drawn from
      families *disjoint* from the fifteen CIFAR-10-C corruptions, at the same
      rate and cost. It is the falsification control for the coverage claim: if
      augmentation strength alone explained the compound gain, this arm would
      close the same gap as the coverage arms.
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
import torchvision
from PIL import Image
from torchvision.transforms import functional as TF

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from maritime_robust.common import (ROOT, Budget, digest, environment, seed_all,
                                    stable_seed, write_json)  # noqa: E402
from maritime_robust.external import cifar_c  # noqa: E402

# Degradations disjoint from CIFAR-10-C's fifteen families: no blur, no noise,
# no brightness/contrast, no elastic, no pixelate, no jpeg. What is left is
# photometric and geometric -- rotation, shear, saturation, hue, posterize,
# solarize, sharpness, cutout -- so the arm can carry the same augmentation
# rate and strength without ever showing the model a tested family.
OTHER_OPS = ("rotate", "shear", "saturation", "hue", "posterize", "solarize",
             "sharpness", "cutout")


def apply_other(image, op, rng):
    """One disjoint degradation, applied to a PIL image; returns a PIL image."""
    if op == "rotate":
        return TF.rotate(image, float(rng.uniform(-15, 15)))
    if op == "shear":
        return TF.affine(image, angle=0.0, translate=[0, 0], scale=1.0,
                         shear=[float(rng.uniform(-10, 10)), 0.0])
    if op == "saturation":
        return TF.adjust_saturation(image, float(rng.uniform(0.3, 1.7)))
    if op == "hue":
        return TF.adjust_hue(image, float(rng.uniform(-0.08, 0.08)))
    if op == "posterize":
        return TF.posterize(image, int(rng.integers(3, 6)))
    if op == "solarize":
        return TF.solarize(image, float(rng.choice([64, 128, 192])))
    if op == "sharpness":
        return TF.adjust_sharpness(image, float(rng.uniform(0.3, 2.0)))
    # cutout: an occluder, the only spatially structured op in this set
    w, h = image.size
    size = int(rng.integers(8, 13))
    x, y = int(rng.integers(0, w - size)), int(rng.integers(0, h - size))
    arr = np.asarray(image).copy()
    arr[y:y + size, x:x + size] = 128
    return Image.fromarray(arr)

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class CifarTrain(torch.utils.data.Dataset):
    """CIFAR-10 with an epoch-dependent random crop, flip and optional corruption.

    ``epoch`` is read through a shared ``Value`` so that it reaches persistent
    DataLoader workers: those keep the dataset object they were forked with, and
    a plain attribute would stay at whatever it held when the workers started.
    Every epoch would then re-use the epoch-0 crop and flip, which trains the
    model on a frozen augmented set (training accuracy 100%, test near 90%).
    """

    def __init__(self, root, aug="clean", pairs=(), severity=3, corrupt_probability=0.5,
                 seed=17, epoch_ref=None, field_bank=None, compound=False,
                 field_repeats=1):
        self.data = torchvision.datasets.CIFAR10(root, train=True, download=True)
        self.aug, self.pairs = aug, list(pairs)
        self.compound = compound
        self.field_repeats = int(field_repeats)
        # NCF: the training degradations are fields estimated from unlabeled
        # deployment images, upsampled once from the estimator's 8x8 grid.
        if aug == "ncf":
            if not field_bank:
                raise SystemExit("--aug ncf needs --field-bank")
            bank = np.load(field_bank)
            n_arr = bank["N"].astype(np.float32)
            if n_arr.shape[-1] == 3:                     # (n, h, w, 3) -> (n, 3, h, w)
                n_arr = n_arr.transpose(0, 3, 1, 2)
            if n_arr.shape[1] != 3:
                raise SystemExit(f"{field_bank}: N has shape {bank['N'].shape}, expected channels")
            self.bank_k = F.interpolate(torch.from_numpy(bank["K"].astype(np.float32)),
                                        size=(32, 32), mode="bilinear", align_corners=False)
            self.bank_n = F.interpolate(torch.from_numpy(n_arr),
                                        size=(32, 32), mode="bilinear", align_corners=False)
            self.bank_s = torch.from_numpy(bank["sigma"].astype(np.float32))
            # per-image kernels, built once: blur is a convolution and cannot be
            # stored as a pixel-wise field (docs/MODULE_NCF.md 2.1, 8.5)
            from maritime_robust.cifar_field_estimator import gaussian_kernels
            if "blur" in bank:
                self.bank_blur = gaussian_kernels(
                    torch.from_numpy(bank["blur"].astype(np.float32))).squeeze(1)
            else:
                self.bank_blur = None
        self.severity, self.corrupt_probability, self.seed = severity, corrupt_probability, seed
        self.epoch_ref = epoch_ref
        self._epoch = 0
        # Canonical AugMix (Hendrycks et al. 2020) with torchvision's own operator set
        # and mixing; it returns an unnormalised tensor, normalised below to match the
        # clean view's scale.
        self.augmix = torchvision.transforms.AugMix(severity=3)

    @property
    def epoch(self):
        return int(self.epoch_ref.value) if self.epoch_ref is not None else self._epoch

    @epoch.setter
    def epoch(self, value):
        if self.epoch_ref is not None:
            self.epoch_ref.value = int(value)
        else:
            self._epoch = int(value)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raw, label = self.data[index]
        image = TF.pad(raw, 4)
        rng = np.random.default_rng(stable_seed(self.seed, self.epoch, index, "cifar_geom"))
        i, j = int(rng.integers(9)), int(rng.integers(9))
        image = TF.crop(image, i, j, 32, 32)
        if rng.random() < 0.5:
            image = TF.hflip(image)
        if self.aug in ("lattice", "singles"):
            np.random.seed(stable_seed(self.seed, self.epoch, index, "cifar_corrupt") % (2**32))
            aug = np.random.default_rng(stable_seed(self.seed, self.epoch, index, "cifar_pick"))
            if aug.random() >= self.corrupt_probability:
                if self.aug == "singles":
                    name = cifar_c.NAMES[int(aug.integers(len(cifar_c.NAMES)))]
                    image = Image.fromarray(cifar_c.apply(image, name, self.severity))
                else:
                    pair = self.pairs[int(aug.integers(len(self.pairs)))]
                    image = Image.fromarray(cifar_c.apply_pair(image, pair, self.severity))
        if self.aug == "otheraug":
            # Same 0.5 rate as the corruption arms; two ops per corrupted draw,
            # so the augmentation strength is comparable rather than a token
            # perturbation.
            aug = np.random.default_rng(stable_seed(self.seed, self.epoch, index, "cifar_other"))
            if aug.random() < self.corrupt_probability:
                for op in aug.choice(OTHER_OPS, size=2, replace=False):
                    image = apply_other(image, str(op), aug)
        if self.aug == "ncf":
            # Two sampled fields composed in *field* space, not two operators
            # stacked in pixel space: that is the module's coverage claim, and
            # --field-compound off is its ablation.
            aug = np.random.default_rng(stable_seed(self.seed, self.epoch, index, "cifar_ncf"))
            if aug.random() < self.corrupt_probability:
                image = TF.to_tensor(image)
                picks = aug.integers(0, len(self.bank_s), size=2 if self.compound else 1)
                for p in picks:
                    # Each estimated field is shallower than the degradation it
                    # describes (an 8x8 prediction averages away pixel-scale
                    # structure, and the kernel head is compressed the same way),
                    # so the whole estimated operation -- convolution, gain,
                    # additive -- is applied `field_repeats` times. The count is
                    # calibrated in scripts/cifar_field_calibrate.py by matching
                    # the deployment pool's own PSNR, not tuned on accuracy.
                    for _ in range(self.field_repeats):
                        if self.bank_blur is not None:
                            kk = self.bank_blur[p].expand(3, 1, *self.bank_blur.shape[-2:])
                            image = F.conv2d(F.pad(image[None], (5, 5, 5, 5), mode="replicate"),
                                             kk, groups=3)[0].clamp(0.0, 1.0)
                        image = (self.bank_k[p] * image + self.bank_n[p]).clamp(0.0, 1.0)
                    image = (image + torch.randn_like(image)
                             * float(self.bank_s[p])).clamp(0.0, 1.0)
        if self.aug == "augmix":
            clean = TF.normalize(TF.to_tensor(image), MEAN, STD)
            # AugMix takes uint8 images: its operators use the uint8 bounds.
            raw_u8 = TF.pil_to_tensor(image)
            to_float = lambda aug: TF.normalize(TF.convert_image_dtype(aug, torch.float32), MEAN, STD)
            return (clean, to_float(self.augmix(raw_u8.clone())),
                    to_float(self.augmix(raw_u8.clone())), label)
        tensor = image if isinstance(image, torch.Tensor) else TF.to_tensor(image)
        return TF.normalize(tensor, MEAN, STD), label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--stem", default="cifar", choices=("cifar", "torchvision"),
                    help="CIFAR-style 3x3 conv1 without max-pool, or torchvision's stem")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--aug", default="clean",
                    choices=["clean", "lattice", "singles", "augmix", "otheraug", "ncf"])
    ap.add_argument("--field-bank", default=None, help="NCF field bank (.npz)")
    ap.add_argument("--field-compound", action="store_true",
                    help="NCF: compose two sampled fields (the coverage arm)")
    ap.add_argument("--field-repeats", type=int, default=1,
                    help="NCF: apply each sampled field this many times; calibrated "
                         "against the deployment pool by scripts/cifar_field_calibrate.py "
                         "(measured: 1 is near-identity at 40 dB, the pool's severity 3 is 22 dB)")
    ap.add_argument("--pairs", default="", help="comma-separated a:b pairs; empty means all")
    ap.add_argument("--pool-k", type=int, default=None,
                    help="use the first k pairs of the balanced non-held-out pool")
    ap.add_argument("--severity", type=int, default=3)
    ap.add_argument("--corrupt-probability", type=float, default=0.5)
    ap.add_argument("--suffix", default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke runs: subset of the train split")
    ap.add_argument("--skip-budget", action="store_true",
                    help="smoke runs only: no GPU lock, no ledger")
    args = ap.parse_args()

    data_root = ROOT / "data/cifar"
    np.random.seed(args.seed)
    seed_all(args.seed)

    pairs = []
    if args.aug in ("lattice", "singles"):
        pairs = ([tuple(p.split(":")) for p in args.pairs.split(",") if p]
                 or (cifar_c.pool(args.pool_k) if args.pool_k else cifar_c.all_pairs()))
        if args.aug == "singles":
            pairs = [(n, n) for n in cifar_c.NAMES]
    suffix = args.suffix or (f"_{args.aug}"
                             + (f"_k{args.pool_k}" if args.pool_k else "")
                             + (f"_{'c' if args.field_compound else 's'}"
                                if args.aug == "ncf" else "")
                             + (f"_r{args.field_repeats}"
                                if args.aug == "ncf" and args.field_repeats > 1 else ""))
    stem = "" if args.stem == "torchvision" else f"_{args.stem}"
    tag = f"cifar10_{args.arch}{stem}{suffix}_s{args.seed}"
    out = ROOT / "runs_cifar" / tag
    if (out / "complete.json").exists():
        raise SystemExit(f"{out} already completed")
    out.mkdir(parents=True, exist_ok=True)

    settings = {"dataset": "cifar10", "arch": args.arch, "stem": args.stem,
                "aug": args.aug, "field_bank": args.field_bank,
                "field_compound": bool(args.field_compound),
                "field_repeats": int(args.field_repeats),
                "pairs": ["+".join(p) for p in pairs], "severity": args.severity,
                "held_out": ["+".join(p) for p in cifar_c.HELD_OUT],
                "corrupt_probability": args.corrupt_probability, "seed": args.seed,
                "epochs": args.epochs, "batch_size": args.batch_size,
                "optimizer": {"name": "sgd", "lr": 0.1, "momentum": 0.9,
                              "weight_decay": 5e-4, "nesterov": True},
                "environment": environment()}
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

    with (NoBudget() if args.skip_budget else Budget("cifar")) as budget:
        epoch_ref = torch.multiprocessing.get_context("fork").Value("i", 0)
        train_ds = CifarTrain(data_root, args.aug, pairs, args.severity,
                              args.corrupt_probability, args.seed, epoch_ref=epoch_ref,
                              field_bank=args.field_bank, compound=args.field_compound,
                              field_repeats=args.field_repeats)
        base = train_ds
        if args.limit:
            train_ds = torch.utils.data.Subset(train_ds, range(args.limit))
        test_ds = torchvision.datasets.CIFAR10(data_root, train=False, download=True)
        test_x = torch.from_numpy(np.asarray(test_ds.data).astype(np.float32))
        test_x = test_x.permute(0, 3, 1, 2) / 255.
        test_x = TF.normalize(test_x, MEAN, STD).cuda()
        test_y = torch.tensor(np.asarray(test_ds.targets)).cuda()

        model = cifar_c.build_resnet(args.arch, args.stem, 10).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                    weight_decay=5e-4, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        # persistent_workers is safe now: the epoch reaches the workers through
        # the shared epoch_ref rather than through the pickled dataset copy.
        loader = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
            pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)

        best, history = -1.0, []
        for epoch in range(args.epochs):
            started = time.perf_counter()
            train_ds.epoch = epoch
            base.epoch = epoch
            model.train()
            loss_sum = correct = 0
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                if args.aug == "augmix":
                    # AugMix with the Jensen-Shannon term of the original recipe:
                    # two mixed views and the clean view, averaged cross-entropy plus
                    # lambda * JSD, so the baseline is the published method and not a
                    # stripped version of it.
                    xc, xm1, xm2, y = batch
                    xc, xm1, xm2, y = (v.cuda(non_blocking=True) for v in (xc, xm1, xm2, y))
                    oc, o1, o2 = model(xc), model(xm1), model(xm2)
                    loss = (F.cross_entropy(o1, y) + F.cross_entropy(o2, y)) / 2
                    pc, p1, p2 = F.softmax(oc, 1), F.softmax(o1, 1), F.softmax(o2, 1)
                    pm = (pc + p1 + p2) / 3
                    kl = lambda p, q: (p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))).sum(1).mean()
                    loss = loss + 12 * (kl(pc, pm) + kl(p1, pm) + kl(p2, pm)) / 3
                    logits = oc
                else:
                    x, y = batch
                    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                    logits = model(x)
                    loss = F.cross_entropy(logits, y, label_smoothing=0.0)
                if not torch.isfinite(loss):
                    raise SystemExit(f"non-finite loss at {tag} epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                loss_sum += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                budget.check()
            model.eval()
            acc = 0
            with torch.no_grad():
                for k in range(0, len(test_x), 500):
                    acc += (model(test_x[k:k + 500]).argmax(1) == test_y[k:k + 500]).sum().item()
            acc /= len(test_y)
            record = {"epoch": epoch + 1, "train_loss": loss_sum / len(train_ds),
                      "train_accuracy": correct / len(train_ds), "test_accuracy": acc,
                      "lr": optimizer.param_groups[0]["lr"],
                      "seconds": time.perf_counter() - started}
            history.append(record)
            write_json(out / "history.json", history)
            if acc > best:
                best = acc
                tmp = out / "best.tmp.pt"
                torch.save({"model": model.state_dict(), "settings": settings,
                            "epoch": epoch + 1, "test_accuracy": acc}, tmp)
                tmp.replace(out / "best.pt")
            scheduler.step()
            print(json.dumps({"run": tag, **record}), flush=True)

        settings.update(best_test_accuracy=best,
                        training_wall_seconds=time.monotonic() - budget.started,
                        checkpoint_sha256=digest(out / "best.pt"))
        write_json(out / "complete.json", settings)
    print(json.dumps({"done": tag, "best_test_accuracy": best}))


if __name__ == "__main__":
    main()
