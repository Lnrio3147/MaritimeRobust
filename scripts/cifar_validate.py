"""Does our regeneration of the canonical corruptions behave like CIFAR-10-C?

The published archive cannot be reproduced bit-exactly: ``make_cifar_c.py`` does
not seed NumPy, so each generation draws fresh noise. The portability of the
replication therefore rests on a weaker but sufficient check -- for a fixed
model, per-condition accuracy on the regenerated corruption must match accuracy
on the published images. If the two agree within noise, the corruption *family*
and its severity calibration are the same thing the literature calls severity s.

Reports mean absolute accuracy difference per corruption across the five
severities, and flags any whose difference exceeds ``--tolerance`` points.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as TF

PROJECT = Path(__file__).resolve().parents[1]
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="reports/cifar_grid/<run> with correct.npz")
    ap.add_argument("--archive", default="/home/uestc/mount_2T/uestc/lnrio/cifar_c/CIFAR-10-C")
    ap.add_argument("--tolerance", type=float, default=1.0, help="pp")
    ap.add_argument("--out", default="reports/CIFAR_VALIDATION.md")
    args = ap.parse_args()

    run = PROJECT / args.run_dir
    summary = json.loads((run / "summary.json").read_text())
    checkpoint = torch.load(PROJECT / "runs_cifar" / run.name / "best.pt",
                            map_location="cuda", weights_only=False)
    model = getattr(torchvision.models, checkpoint["settings"]["arch"])(num_classes=10).cuda()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    archive = Path(args.archive)
    labels = np.load(archive / "labels.npy")
    test = torchvision.datasets.CIFAR10(str(PROJECT / "data/cifar"), train=False, download=True)
    assert np.array_equal(labels, np.asarray(test.targets)), "labels.npy does not match CIFAR-10 test order"
    n = len(labels)

    rows, worst = [], 0.0
    for path in sorted(archive.glob("*.npy")):
        corruption = path.stem
        if corruption == "labels":
            continue
        images = np.load(path)
        deltas = []
        for severity in range(1, 6):
            chunk = images[(severity - 1) * n: severity * n]
            x = TF.normalize(torch.from_numpy(chunk).permute(0, 3, 1, 2) / 255., MEAN, STD).cuda()
            y = torch.from_numpy(labels).cuda()
            hits = 0
            with torch.no_grad():
                for k in range(0, len(x), 500):
                    hits += (model(x[k:k + 500]).argmax(1) == y[k:k + 500]).sum().item()
            ours = summary["accuracy"].get(f"{corruption}_{severity}")
            deltas.append(float("nan") if ours is None
                          else 100 * (hits / len(x) - ours))
        usable = [d for d in deltas if not np.isnan(d)]
        if not usable:
            continue
        rows.append((corruption, deltas))
        worst = max(worst, max(abs(d) for d in usable))

    lines = ["# Port fidelity against the published CIFAR-10-C archive", "",
             "Model accuracy on published vs regenerated images, per severity "
             "(percentage points; regenerated minus published). The archive is not "
             "bit-reproducible upstream, so agreement in accuracy is the check that "
             "matters: the same corruption family at the same calibrated severity.",
             "",
             f"Worst absolute difference: **{worst:.2f} pp** "
             f"(tolerance {args.tolerance:.1f} pp) — "
             f"{'PASS' if worst <= args.tolerance else 'REVIEW'}.",
             "",
             "| corruption | s1 | s2 | s3 | s4 | s5 | mean | verdict |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for corruption, deltas in rows:
        cells = ["—" if np.isnan(d) else f"{d:+.2f}" for d in deltas]
        usable = [d for d in deltas if not np.isnan(d)]
        mean = float(np.mean(usable))
        verdict = "ok" if abs(mean) <= args.tolerance else "mismatch"
        lines.append(f"| {corruption} | " + " | ".join(cells) +
                     f" | {mean:+.2f} | {verdict} |")
    (PROJECT / args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
