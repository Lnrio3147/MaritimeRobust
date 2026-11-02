"""Does the blur head identify anything? Correlate its prediction with truth.

A head that outputs the conditional mean of the prior has zero correlation with
the true kernel and an MAE close to the prior's own spread. That is what
``blur 2.3`` at the end of training looked like, so this separates "learned a
little" from "learned nothing" before any arm is spent on the bank it feeds.

    python scripts/cifar_field_blurdiff.py --ckpt artifacts/field_estimator_blur.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from maritime_robust.cifar_field_estimator import (FieldEstimator, apply_blur_torch,  # noqa: E402
                                                   apply_field_torch)
from maritime_robust.cifar_fields import sample_field  # noqa: E402
from maritime_robust.external import cifar_c  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/field_estimator_blur.pt")
    ap.add_argument("--n-synth", type=int, default=512)
    ap.add_argument("--n-real", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FieldEstimator().to(device)
    model.load_state_dict(torch.load(PROJECT / args.ckpt, map_location=device))
    model.eval()

    import torchvision
    ds = torchvision.datasets.CIFAR10(PROJECT / "data/cifar", train=False, download=False)
    rng = np.random.default_rng(5)

    # --- synthetic: blur + field from the training prior, truth known
    idx = rng.choice(len(ds), size=args.n_synth, replace=False)
    x = torch.from_numpy(np.asarray(ds.data)[idx]).permute(0, 3, 1, 2).float().div_(255.).to(device)
    from maritime_robust.cifar_fields import sample_blur
    B = np.stack([sample_blur(rng, p=0.5) for _ in range(args.n_synth)])
    bt = torch.from_numpy(B).to(device)
    with torch.no_grad():
        y = apply_blur_torch(x, bt)
        pred = model.blur_params(y).cpu().numpy()
    touched = (B[:, 0] + B[:, 1]) > 0
    print(f"synthetic: {touched.sum()}/{len(B)} samples carry blur")
    for j, nm in enumerate(("sx", "sy")):
        err = np.abs(pred[:, j] - B[:, j])
        base = np.abs(B[:, j] - B[:, j].mean())          # predict-the-mean baseline
        corr = np.corrcoef(pred[:, j], B[:, j])[0, 1] if B[:, j].std() > 0 else float("nan")
        print(f"  {nm}: MAE {err.mean():.3f} (mean-baseline {base.mean():.3f}) "
              f"corr {corr:+.3f}  pred mean {pred[:, j].mean():.3f} "
              f"true mean {B[:, j].mean():.3f}")
    return  # real-image half below is printed only when the synthetic half is alive

    # --- real: mean predicted blur per family, is it ordered the way physics says?
    clean = np.asarray(ds.data)[rng.choice(len(ds), size=args.n_real, replace=False)]
    rows = []
    for fam in cifar_c.NAMES:
        t = np.stack([cifar_c.apply(im, fam, 3) for im in clean])
        t = torch.from_numpy(t).permute(0, 3, 1, 2).float().div_(255.).to(device)
        with torch.no_grad():
            p = model.blur_params(t).cpu().numpy()
        rows.append((fam, p[:, 0].mean(), p[:, 1].mean()))
    for fam, sx, sy in rows:
        print(f"  {fam:20s} sx {sx:.3f} sy {sy:.3f}")


if __name__ == "__main__":
    main()
