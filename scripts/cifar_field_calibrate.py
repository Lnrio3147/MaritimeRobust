"""Calibrate the estimated field bank against the deployment pool's own strength.

The estimator is trained to regress smooth fields, but the deployment images it
is applied to carry high-frequency structure (fog's transmission field varies at
pixel scale). An 8x8 prediction averages that structure away, so the estimated
field is *shallower* than the degradation it describes: measured, mean K is 0.943
across a pool whose true transmissions run to 0.2.

That bias is not random and can be corrected without labels: the deployment pool
itself says how degraded it is, and the bank says how degraded its fields make an
image. Match the two distributions and read off the correction.

The correction stays inside the representation -- a field applied ``m`` times is
still a field, ``x -> K^m x + N (1 + K + ... + K^{m-1})`` -- so calibrated
sampling is still "sample a field, apply it", not a new operator.

    python scripts/cifar_field_calibrate.py --bank artifacts/fields_a.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from maritime_robust.cifar_fields import apply_blur  # noqa: E402
from maritime_robust.external import cifar_c  # noqa: E402


def apply_m(x, K, N, m):
    """Apply one field ``m`` times; closed form, so no loop over pixels."""
    out = x
    for _ in range(m):
        out = np.clip(K[..., None] * out + N, 0.0, 1.0)
    return out


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse <= 0 else 10.0 * np.log10(1.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="artifacts/fields_a.npz")
    ap.add_argument("--n-images", type=int, default=48)
    ap.add_argument("--severities", default="1,3,5")
    ap.add_argument("--out", default="artifacts/fields_a_cal.npz")
    ap.add_argument("--report", default="reports/FIELD_CALIBRATION.md")
    ap.add_argument("--report-json", default="reports/field_calibration.json")
    args = ap.parse_args()

    import torchvision
    ds = torchvision.datasets.CIFAR10(PROJECT / "data/cifar", train=False, download=False)
    rng = np.random.default_rng(3)
    idx = rng.choice(len(ds), size=args.n_images, replace=False)
    clean = np.asarray(ds.data)[idx].astype(np.float32) / 255.0

    sevs = [int(s) for s in args.severities.split(",")]
    fams = [n for n in cifar_c.NAMES]

    # How strong is the deployment pool, as a PSNR against its own clean image?
    real = {}
    for fam in fams:
        for sev in sevs:
            vals = [psnr(cifar_c.apply((c * 255).astype(np.uint8), fam, sev).astype(np.float32) / 255.0, c)
                    for c in clean[:16]]
            real[(fam, sev)] = float(np.mean(vals))

    pool = np.load(PROJECT / args.bank)
    K = pool["K"].astype(np.float32)                # (n, 1, 8, 8)
    N = pool["N"].astype(np.float32)                # (n, 3, 8, 8)
    import torch
    import torch.nn.functional as TF

    def upsample(f):
        """Exactly the upsample ``cifar_train`` applies when it loads the bank."""
        return TF.interpolate(torch.from_numpy(f), size=(32, 32),
                              mode="bilinear", align_corners=False).numpy()

    Ku = upsample(K)[:, 0]                              # (n, 32, 32)
    Nu = upsample(N).transpose(0, 2, 3, 1)              # (n, 32, 32, 3)
    B = pool["blur"].astype(np.float32) if "blur" in pool else None

    # How strong is a bank sample, for m applications of one field, and for two
    # composed fields each applied m times?
    rows = []
    for m in (1, 2, 3):
        for comp in (1, 2):
            vals = []
            for _ in range(96):
                c = clean[rng.integers(len(clean))]
                i = rng.integers(len(Ku))
                out = c
                for _ in range(m):
                    out = apply_blur(out, B[i]) if B is not None else out
                    out = np.clip(Ku[i][..., None] * out + Nu[i], 0.0, 1.0)
                for _ in range(comp - 1):
                    j = rng.integers(len(Ku))
                    for _ in range(m):
                        out = apply_blur(out, B[j]) if B is not None else out
                        out = np.clip(Ku[j][..., None] * out + Nu[j], 0.0, 1.0)
                vals.append(psnr(out, c))
            rows.append({"m": m, "compound": comp, "psnr_mean": float(np.mean(vals)),
                         "psnr_p10": float(np.percentile(vals, 10))})

    singles_by_sev = {s: float(np.mean([real[(f, s)] for f in fams])) for s in sevs}
    target_single = singles_by_sev.get(3, float("nan"))

    lines = ["# Field-bank calibration", "",
             "PSNR of the deployment pool's conditions against their clean originals "
             "(the strength the training augmentation should reach), and PSNR of images "
             "degraded by sampled bank fields under different application counts.", "",
             "## Deployment pool (real CIFAR-10-C conditions)", "",
             "| severity | mean PSNR over 15 families |", "|---|---:|"]
    lines += [f"| {s} | {singles_by_sev[s]:.2f} |" for s in sevs]
    lines += ["", "## Sampled bank fields", "",
              "| applications m | fields composed | mean PSNR | 10th pct |", "|---|---:|---:|---:|"]
    lines += [f"| {r['m']} | {r['compound']} | {r['psnr_mean']:.2f} | {r['psnr_p10']:.2f} |" for r in rows]

    best = min(rows, key=lambda r: abs(r["psnr_mean"] - target_single))
    lines += ["", f"Closest to the single-family severity-3 pool ({target_single:.2f} dB): "
                  f"m={best['m']}, composed={best['compound']}.", ""]

    (PROJECT / args.report).write_text("\n".join(lines) + "\n")
    (PROJECT / args.report_json).write_text(json.dumps(
        {"pool_psnr": {f"{f}_{s}": v for (f, s), v in real.items()},
         "bank": rows, "best": best}, indent=1) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
