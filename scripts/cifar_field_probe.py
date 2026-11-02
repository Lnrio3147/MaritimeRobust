"""Phase 0 of docs/MODULE_NCF.md: does a fitted field transfer across content?

The module's premise is that a degradation can be estimated from a deployment
image and then applied to *different* clean images to synthesise training pairs.
The first thing to settle is whether the (K, N) representation captures the
degradation rather than the picture it was fitted on.

That matters because the representation is, on its own, universal: with K = 1 the
additive field N can be any image at all, so a per-image fit always "works" and
measures nothing. The informative test is a content transfer: fit (K, N) on one
clean image and its degradation, then apply that field to a *different* clean
image and compare with that image's true degraded version.

    transfer ≈ identity   the field explains none of the degradation
    transfer >> identity  the field carries degradable structure across content

Reported per condition: PSNR of identity, of the in-sample fit (overfit
reference), and of the cross-content transfer, plus the fraction of the
degradation's MSE the transferred field removes.

    python scripts/cifar_field_probe.py --images 32 --steps 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from maritime_robust.external import cifar_c  # noqa: E402


def load_clean(n: int, seed: int) -> np.ndarray:
    import torchvision
    ds = torchvision.datasets.CIFAR10(PROJECT / "data/cifar", train=False, download=False)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=n, replace=False)
    # uint8: the corruption code is the canonical one and expects uint8 or PIL,
    # exactly as scripts/cifar_train.py calls it. Feeding it floats silently
    # casts every image to a constant.
    return np.asarray(ds.data)[idx]


def psnr(mse: torch.Tensor) -> float:
    return float(10 * torch.log10(1.0 / mse.clamp_min(1e-10)).mean())


def fit_fields(clean, target, steps, lr, smooth_k, smooth_n):
    """Fit one (K, N) per image with the clean image known."""
    K = torch.ones(clean.shape[0], 1, *clean.shape[2:], device=clean.device,
                   requires_grad=True)
    N = torch.zeros_like(clean, requires_grad=True)
    opt = torch.optim.Adam([K, N], lr=lr)
    tv = lambda t: ((t[..., 1:] - t[..., :-1]).abs().mean()
                    + (t[..., :, 1:] - t[..., :, :-1]).abs().mean())
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = ((K * clean + N).clamp(0, 1) - target).abs().mean()
        loss = loss + smooth_k * tv(K) + smooth_n * tv(N)
        loss.backward()
        opt.step()
        with torch.no_grad():
            K.clamp_(0, 2)
    return K.detach(), N.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=32)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--severity", type=int, default=3)
    ap.add_argument("--smooth-k", type=float, default=1e-2)
    ap.add_argument("--smooth-n", type=float, default=1e-2)
    ap.add_argument("--pairs", default="frost+snow,gaussian_noise+shot_noise,jpeg_compression+pixelate",
                    help="compound conditions to probe, in addition to all singles")
    ap.add_argument("--out", default="reports/FIELD_CAPACITY.md")
    ap.add_argument("--json", default="reports/field_capacity.json")
    args = ap.parse_args()

    clean_u8 = load_clean(args.images, seed=17)
    clean_np = clean_u8.astype(np.float32) / 255.0
    clean = torch.from_numpy(clean_np).permute(0, 3, 1, 2).contiguous().cuda()
    # Partner image for the transfer test: a different picture, same condition.
    perm = torch.roll(torch.arange(clean.shape[0], device=clean.device), -1)
    clean_partner = clean[perm]

    rows, payload = [], {}

    def probe(name, target_np):
        target = torch.from_numpy(target_np).permute(0, 3, 1, 2).contiguous().cuda()
        target_partner = target[perm]
        K, N = fit_fields(clean, target, args.steps, args.lr, args.smooth_k, args.smooth_n)
        with torch.no_grad():
            fit = ((K * clean + N).clamp(0, 1) - target).pow(2).mean(dim=(1, 2, 3))
            trans = ((K * clean_partner + N).clamp(0, 1) - target_partner).pow(2).mean(dim=(1, 2, 3))
            ident = (clean_partner - target_partner).pow(2).mean(dim=(1, 2, 3))
        out = {"psnr_identity": psnr(ident), "psnr_fit": psnr(fit),
               "psnr_transfer": psnr(trans),
               "mse_removed": float((1 - trans / ident.clamp_min(1e-10)).mean())}
        payload[name] = out
        rows.append((name, out))
        print(f"{name:32s} identity {out['psnr_identity']:5.2f}  fit {out['psnr_fit']:5.2f}  "
              f"transfer {out['psnr_transfer']:5.2f} dB  removed {out['mse_removed']:+.3f}",
              flush=True)

    for name in cifar_c.NAMES:
        target = np.stack([cifar_c.apply(im, name, args.severity) for im in clean_u8])
        probe(name, target.astype(np.float32) / 255.0)

    for pair in [p for p in args.pairs.split(",") if p]:
        a, b = pair.split("+")
        target = np.stack([cifar_c.apply_pair(im, (a, b), args.severity) for im in clean_u8])
        probe(pair, target.astype(np.float32) / 255.0)

    lines = ["# Field capacity probe (NCF phase 0)", "",
             f"{args.images} clean CIFAR-10 test images, severity {args.severity}. "
             f"(K, N) fitted per image on image *i* ({args.steps} Adam steps, "
             f"TV {args.smooth_k}/{args.smooth_n}) and evaluated on a different image. "
             "`fit` is the in-sample reference and is expected to look perfect — with "
             "K = 1 the additive field can copy any target — so only the transfer column "
             "says anything.", "",
             "| condition | PSNR identity | PSNR in-sample | PSNR transfer | MSE removed |",
             "|---|---:|---:|---:|---:|"]
    for name, out in rows:
        lines.append(f"| {name} | {out['psnr_identity']:.2f} | {out['psnr_fit']:.2f} | "
                     f"{out['psnr_transfer']:.2f} | {out['mse_removed']:+.3f} |")
    (PROJECT / args.out).write_text("\n".join(lines) + "\n")
    (PROJECT / args.json).write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
