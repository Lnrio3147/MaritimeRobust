"""Figure: the neural-corruption-field module.

Drawn at true print size (one column, 3.4 in wide, 2.3 in tall) so the labels
survive the page. Three moves, top to bottom: estimate one field per image;
bank the fields from unlabeled deployment images; sample and compose them as
training augmentation for an otherwise unchanged classifier.

    python scripts/fig_ncf_architecture.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT = Path(__file__).resolve().parents[1]

INK = "#1e2430"
MUTED = "#5c6675"
FIELD = "#2f6f9f"
FIELD_BG = "#e8f0f7"
NET = "#6f4f9c"
NET_BG = "#efeaf7"
AUG = "#a85724"
AUG_BG = "#faeee4"

W, H = 3.4, 2.06          # inches, real print size


def box(ax, x, y, w, h, label, fc="white", ec=INK, fs=5.2, lw=0.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, color=INK, zorder=3, linespacing=1.20)


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=0.7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=5, linewidth=lw, color=color,
                                 zorder=1, shrinkA=0.6, shrinkB=0.6))


def main():
    fig = plt.figure(figsize=(W, H), dpi=600)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ------------------------------------------------------- 1  estimate
    ax.text(0.03, 1.98, "1  estimate one field per image", fontsize=5.9,
            color=NET, weight="bold")
    ax.text(0.03, 1.86, "one image at a time, no family name",
            fontsize=4.7, color=MUTED, style="italic")

    box(ax, 0.05, 1.40, 0.66, 0.32, "deployment\nimage")
    box(ax, 0.79, 1.40, 0.76, 0.32, "encoder\n3 conv blocks", fc=NET_BG, ec=NET)
    box(ax, 1.65, 1.40, 1.70, 0.32,
        "K, N on 8×8  ·  σ\nkernel  σx, σy, θ", fc=FIELD_BG, ec=FIELD)
    arrow(ax, 0.71, 1.56, 0.79, 1.56)
    arrow(ax, 1.55, 1.56, 1.65, 1.56)

    # ------------------------------------------------------- 2  bank
    ax.text(0.03, 1.16, "2  bank them", fontsize=5.9, color=FIELD, weight="bold")
    for i, alpha in enumerate((0.30, 0.55, 1.0)):
        ax.add_patch(FancyBboxPatch((1.45 + i * 0.05, 0.78 + i * 0.035), 1.15, 0.24,
                                    boxstyle="round,pad=0.004,rounding_size=0.012",
                                    linewidth=0.6, edgecolor=FIELD, facecolor=FIELD_BG,
                                    alpha=alpha, zorder=2 + i))
    ax.text(2.02, 0.66, "fields from the unlabeled pool", ha="center",
            fontsize=4.7, color=MUTED)
    arrow(ax, 2.50, 1.39, 2.16, 1.06, color=FIELD)

    # ------------------------------------------------------- 3  resample
    ax.text(0.03, 0.46, "3  resample as augmentation", fontsize=5.9,
            color=AUG, weight="bold")
    box(ax, 0.05, 0.06, 0.62, 0.28, "sample\n1–2 fields", fc=AUG_BG, ec=AUG)
    box(ax, 0.73, 0.06, 0.78, 0.28, "compose in\nfield space", fc=AUG_BG, ec=AUG)
    box(ax, 1.57, 0.06, 0.72, 0.28, "apply m×\nto clean", fc=AUG_BG, ec=AUG)
    box(ax, 2.35, 0.06, 1.00, 0.28, "ResNet-18\n(unchanged)", fc=NET_BG, ec=NET)
    arrow(ax, 0.67, 0.20, 0.73, 0.20, color=AUG)
    arrow(ax, 1.51, 0.20, 1.57, 0.20, color=AUG)
    arrow(ax, 2.29, 0.20, 2.35, 0.20)

    out = PROJECT / "paper/figures/fig6_ncf_architecture.png"
    fig.savefig(out, dpi=600, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
