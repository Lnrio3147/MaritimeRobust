#!/usr/bin/env python
"""Validation curve for the reproduced EFM-Net run.

Reads <run>/record.txt (one JSON object per epoch, written by EFM-Net's own
logger) and prints a compact table plus saves a single-series PNG.

    python scripts/efmnet_curve.py --run-dir runs_external/fgsc23_civ8/<run>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID, ACCENT = "#1f2328", "#6e7781", "#e6e8eb", "#2f6db3"


def load(run_dir: Path):
    rows = []
    for line in (run_dir / "record.txt").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if "test_acc1" in record:
            rows.append(record)
    rows.sort(key=lambda r: r["epoch"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--step", type=int, default=100, help="print every N epochs")
    ap.add_argument("--output", default="reports/figures/efmnet_civ8_s17_val.png")
    args = ap.parse_args()

    rows = load(ROOT / args.run_dir)
    if not rows:
        raise SystemExit("no epochs recorded yet")
    epochs = [r["epoch"] + 1 for r in rows]
    # EFM-Net logs acc1 as a percentage (35.03); accept a 0-1 fraction too.
    acc = [r["test_acc1"] if r["test_acc1"] > 1 else 100 * r["test_acc1"] for r in rows]
    loss = [r["train_loss"] for r in rows]
    best = max(range(len(acc)), key=lambda i: acc[i])

    print(f"epochs={len(rows)} best=epoch {epochs[best]} acc={acc[best]:.2f}% "
          f"last={acc[-1]:.2f}%")
    print("epoch\tval_acc%\ttrain_loss")
    for i, e in enumerate(epochs):
        if e % args.step == 0 or i == best or i == len(epochs) - 1:
            print(f"{e}\t{acc[i]:.2f}\t{loss[i]:.3f}")

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=160)
    ax.plot(epochs, acc, color=ACCENT, linewidth=2, solid_capstyle="round")
    ax.scatter([epochs[best]], [acc[best]], s=40, color=ACCENT, zorder=3,
               edgecolor="white", linewidth=2)
    ax.annotate(f"best {acc[best]:.2f}% @ epoch {epochs[best]}", (epochs[best], acc[best]),
                textcoords="offset points", xytext=(10, 8), fontsize=9, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.axhline(12.5, color=MUTED, linewidth=1, linestyle=(0, (4, 4)))
    ax.text(0.005, 12.5, " chance 12.5%", transform=ax.get_yaxis_transform(),
            va="bottom", fontsize=8, color=MUTED,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_xlabel("epoch", fontsize=9, color=MUTED)
    ax.set_ylabel("val accuracy (%)", fontsize=9, color=MUTED)
    ax.set_title("EFM-Net (ConvNeXt-B), 8-class civilian split, seed 17", fontsize=10,
                 color=INK, loc="left")
    ax.set_xlim(0, max(epochs))
    ax.set_ylim(0, max(acc) + 8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
