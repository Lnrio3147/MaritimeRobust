"""Train the field estimator on the continuous prior and test it on real families.

Training uses only ``cifar_fields.sample_field`` — physical parameters, no family
names. Evaluation has two halves:

    synthetic   estimate the field from a corrupted image, apply it to a
                *different* clean image, compare with that image's true
                corruption. This is what the phase-0 per-image fit failed.
    real        the same transfer test on the fifteen CIFAR-10-C families and a
                few compound conditions, which the estimator has never seen by
                name. A family the representation cannot express will show up
                here, not in training.

    python scripts/cifar_field_train.py --steps 3000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from maritime_robust.cifar_fields import sample_degradation  # noqa: E402
from maritime_robust.cifar_field_estimator import (FieldEstimator, apply_blur_torch,  # noqa: E402
                                                   apply_field_torch, identifiability,
                                                   supervised_loss)
from maritime_robust.external import cifar_c  # noqa: E402


def clean_batch(root, batch, device, rng):
    import torchvision
    if not hasattr(clean_batch, "data"):
        clean_batch.data = torchvision.datasets.CIFAR10(root, train=True, download=False).data
    d = clean_batch.data
    idx = rng.integers(0, len(d), size=batch)
    x = torch.from_numpy(np.asarray(d)[idx]).permute(0, 3, 1, 2).float().div_(255.)
    return x.to(device)


def synth_batch(root, batch, device, rng):
    """Two contents, one shared field — the construction the estimator is trained on.

    Sharing the field is what makes the decomposition identifiable: the network
    cannot put content into ``N`` when ``N`` must explain two different pictures.
    """
    x1 = clean_batch(root, batch, device, rng)
    x2 = clean_batch(root, batch, device, rng)
    ks, ns, ss, bs = zip(*(sample_degradation(rng, 32, 32) for _ in range(batch)))
    k = torch.from_numpy(np.stack(ks)).unsqueeze(1).to(device)
    n = torch.from_numpy(np.stack(ns)).permute(0, 3, 1, 2).to(device)
    s = torch.tensor(ss, dtype=torch.float32, device=device)
    b = torch.from_numpy(np.stack(bs)).to(device)

    def degrade(x):
        # order matches the physics: the scene is convolved first, then the
        # field attenuates and scatters it, then sensor noise is added
        y = apply_field_torch(apply_blur_torch(x, b), k, n)
        return (y + torch.randn_like(y) * s[:, None, None, None]).clamp_(0.0, 1.0)

    return x1, x2, degrade(x1), degrade(x2), k, n, s, b


def psnr(mse):
    return float(10 * torch.log10(1.0 / mse.clamp_min(1e-10)).mean())


@torch.no_grad()
def transfer_eval(model, x_clean_u8, name, severity, pair=None):
    """Estimate on image i; apply to image j; compare with j's real corruption."""
    device = next(model.parameters()).device
    if pair is None:
        target = np.stack([cifar_c.apply(im, name, severity) for im in x_clean_u8])
    else:
        target = np.stack([cifar_c.apply_pair(im, pair, severity) for im in x_clean_u8])
    target = torch.from_numpy(target).permute(0, 3, 1, 2).float().div_(255.).to(device)
    clean = torch.from_numpy(x_clean_u8).permute(0, 3, 1, 2).float().div_(255.).to(device)
    perm = torch.roll(torch.arange(clean.shape[0], device=device), -1)
    k, n = model(target)
    sigma = model.coarse(target)[2]
    blur = model.blur_params(target)
    # a fresh noise realisation at the *estimated magnitude*: the realisation of
    # white noise is not transferable, its scale is
    noise = torch.randn_like(clean) * sigma[:, None, None, None]
    pred = (apply_field_torch(apply_blur_torch(clean[perm], blur), k, n) + noise).clamp_(0.0, 1.0)
    tp = target[perm]
    mse_pred = (pred - tp).pow(2).mean(dim=(1, 2, 3))
    mse_id = (clean[perm] - tp).pow(2).mean(dim=(1, 2, 3))
    return {"psnr_transfer": psnr(mse_pred), "psnr_identity": psnr(mse_id),
            "mse_removed": float((1 - mse_pred / mse_id.clamp_min(1e-10)).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--smooth", type=float, default=1e-2)
    ap.add_argument("--equiv", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--images", type=int, default=64, help="evaluation images")
    ap.add_argument("--out", default="reports/FIELD_ESTIMATOR.md")
    ap.add_argument("--json", default="reports/field_estimator.json")
    ap.add_argument("--ckpt", default="artifacts/field_estimator.pt")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FieldEstimator().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    started = time.time()
    model.train()
    for step in range(args.steps):
        x1, _, d1, _, k1, n1, s1, b1 = synth_batch(PROJECT / "data/cifar", args.batch, device, rng)
        loss, met = supervised_loss(model, d1, x1, k1, n1, s1, b_true=b1, smooth=args.smooth)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()
        sched.step()
        if step % 200 == 0 or step == args.steps - 1:
            # identifiability: on a clean image the field must vanish
            idf = identifiability(model, x1, d1)
            print(f"step {step:5d} rec {met['rec']:.4f} field {met['field']:.4f} "
                  f"sig {met.get('sigma_err', float('nan')):.4f} "
                  f"blur {met.get('blur_err', float('nan')):.3f} "
                  f"K(clean) {idf['k_mean']:.3f} N(clean) {idf['n_abs']:.4f} "
                  f"spread {idf['k_spread']:.4f}/{idf['n_spread']:.4f} "
                  f"{time.time() - started:6.1f}s", flush=True)

    torch.save(model.state_dict(), PROJECT / args.ckpt)

    import torchvision
    ds = torchvision.datasets.CIFAR10(PROJECT / "data/cifar", train=False, download=False)
    idx = rng.choice(len(ds), size=args.images, replace=False)
    x_u8 = np.asarray(ds.data)[idx]

    model.eval()
    rows, payload, removed = [], {}, []
    for name in cifar_c.NAMES:
        out = transfer_eval(model, x_u8, name, 3)
        payload[name] = out
        removed.append(out["mse_removed"])
        rows.append((name, out))
        print(f"{name:20s} identity {out['psnr_identity']:5.2f} transfer {out['psnr_transfer']:5.2f} "
              f"removed {out['mse_removed']:+.3f}", flush=True)
    compounds = [("frost", "snow"), ("fog", "contrast"), ("gaussian_noise", "shot_noise")]
    for a, b in compounds:
        out = transfer_eval(model, x_u8, None, 3, pair=(a, b))
        payload[f"{a}+{b}"] = out
        rows.append((f"{a}+{b}", out))
        print(f"{a}+{b:20s} identity {out['psnr_identity']:5.2f} transfer {out['psnr_transfer']:5.2f} "
              f"removed {out['mse_removed']:+.3f}", flush=True)

    idf = identifiability(model, clean_batch(PROJECT / "data/cifar", 64, device, rng))
    lines = ["# Field estimator: cross-content transfer (NCF phase 1)", "",
             f"Estimator trained {args.steps} steps on the continuous prior of "
             "`cifar_fields.sample_field` — physical parameters only, no family names. "
             f"Training objective is equivariant (one field, two contents, estimates "
             f"must agree; weight {args.equiv}). Identifiability on clean input: "
             f"mean K {idf['k_mean']:.3f} (target 1), mean |N| {idf['n_abs']:.4f} "
             f"(target 0), corr(N, x) {idf['corr_n_x']:+.3f} (target 0). "
             f"Transfer measured on {args.images} held-out CIFAR-10 test images: estimate the "
             "field from image *i*'s degraded version, apply it to image *j*, compare with "
             "*j*'s true degradation.", "",
             "| condition | PSNR identity | PSNR transfer | MSE removed |",
             "|---|---:|---:|---:|"]
    for name, out in rows:
        lines.append(f"| {name} | {out['psnr_identity']:.2f} | {out['psnr_transfer']:.2f} | "
                     f"{out['mse_removed']:+.3f} |")
    lines += ["", f"Families with positive transfer: "
                  f"{sum(1 for r in removed if r > 0)} of {len(removed)}."]
    (PROJECT / args.out).write_text("\n".join(lines) + "\n")
    (PROJECT / args.json).write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
