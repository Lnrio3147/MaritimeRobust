"""Diagnose *why* the phase-1 estimator transfers nothing.

The transfer table is unambiguous: every family gets worse than doing nothing.
That has a mechanical explanation, and this script measures it rather than
asserting it. The field representation is underdetermined — one equation per
pixel, four unknowns — so (K=0, N=x_deg) reconstructs any target exactly. If
the estimator found that solution it is not estimating degradation at all, it is
copying the input into the additive field. Three measurements decide it:

    clean        feed an *undegraded* image. A field estimator should say
                 K = 1, N = 0. A copier says N ≈ x.
    identity     correlation between the estimated N and the input image.
    fit-error    reconstruct with the estimated field from a *different* image's
                 content. A field is content-free only if this stays low.

    python scripts/cifar_field_diagnose.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from maritime_robust.cifar_field_estimator import FieldEstimator, apply_field_torch  # noqa: E402
from maritime_robust.cifar_fields import sample_field  # noqa: E402


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FieldEstimator().to(device)
    model.load_state_dict(torch.load(PROJECT / "artifacts/field_estimator.pt", map_location=device))
    model.eval()

    import torchvision
    ds = torchvision.datasets.CIFAR10(PROJECT / "data/cifar", train=False, download=False)
    rng = np.random.default_rng(17)
    idx = rng.choice(len(ds), size=32, replace=False)
    x = torch.from_numpy(np.asarray(ds.data)[idx]).permute(0, 3, 1, 2).float().div_(255.).to(device)

    with torch.no_grad():
        # 1. clean input: an honest estimator is the identity field here
        k, n = model(x)
        corr = np.corrcoef(n.mean(1).flatten().cpu().numpy(),
                           x.mean(1).flatten().cpu().numpy())[0, 1]
        print(f"clean input   mean K {k.mean():.3f}  mean|N| {n.abs().mean():.4f}  "
              f"std N {n.std():.4f}")
        print(f"              corr(N, x) {corr:+.3f}")
        print(f"              |clip(K*x+N) - x| {((k * x + n).clamp(0, 1) - x).abs().mean():.4f}")

        # 2. degradation built from a *known* field: does the estimate recover it?
        ks, ns, _ss = zip(*(sample_field(rng, 32, 32) for _ in range(32)))
        kt = torch.from_numpy(np.stack(ks)).unsqueeze(1).to(device)
        nt = torch.from_numpy(np.stack(ns)).permute(0, 3, 1, 2).to(device)
        xd = apply_field_torch(x, kt, nt)
        k, n = model(xd)
        print(f"known field   K err {((k - kt).abs().mean()):.4f}  "
              f"N err {(n - nt).abs().mean():.4f}")

        # 3. is N carrying the picture? correlate with the degraded input
        corr_deg = np.corrcoef(n.mean(1).flatten().cpu().numpy(),
                               xd.mean(1).flatten().cpu().numpy())[0, 1]
        print(f"              corr(N, x_deg) {corr_deg:+.3f}")

        # 4. transfer: estimate on image i, apply to image j (the phase-1 metric)
        perm = torch.roll(torch.arange(x.shape[0], device=device), -1)
        pred = apply_field_torch(x[perm], k, n)
        mse_pred = (pred - xd[perm]).pow(2).mean(dim=(1, 2, 3))
        mse_id = (x[perm] - xd[perm]).pow(2).mean(dim=(1, 2, 3))
        print(f"transfer      psnr {10 * torch.log10(1 / mse_pred.mean()):.2f} dB  "
              f"identity {10 * torch.log10(1 / mse_id.mean()):.2f} dB")


if __name__ == "__main__":
    main()
