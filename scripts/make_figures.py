"""Figures for the paper draft. Light-mode, two-color categorical palette.

    python scripts/make_figures.py    -> paper/figures/*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"   # validated pair (CVD dE 24.7 protan)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def fig_severity_curve():
    d = json.loads((ROOT / "reports/severity_curve_results.json").read_text())
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    for pair, color, label in (("haze+glare", BLUE, "haze + glare"),
                               ("haze+cloud", ORANGE, "haze + cloud")):
        cells = sorted([c for c in d["cells"] if c["pair"] == pair], key=lambda c: c["severity"])
        xs = [c["severity"] for c in cells]
        ys = [c["excess_pp"][0] for c in cells]
        lo = [c["excess_pp"][0] - c["excess_pp"][1] for c in cells]
        hi = [c["excess_pp"][2] - c["excess_pp"][0] for c in cells]
        for run in cells[0]["per_run"]:
            pass
        # per-run thin lines (same hue, low alpha)
        runs = {}
        for c in cells:
            for r in c["per_run"]:
                runs.setdefault((r["fold"], r["seed"]), []).append(r)
        for key, series in runs.items():
            ax.plot(xs, [s["excess_pp"] for s in series], color=color, lw=0.8,
                    alpha=0.28, zorder=2)
        ax.errorbar(xs, ys, yerr=[lo, hi], color=color, marker="o", ms=5,
                    capsize=3, elinewidth=1.2, label=label, zorder=3)
        ax.annotate(f"{ys[-1]:+.1f}", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(7, 0), color=color, fontsize=9, va="center")
    ax.set_xlabel("degradation severity")
    ax.set_ylabel("superadditivity  (pp, accuracy space)")
    ax.set_xticks([2, 3, 4, 5])
    ax.set_xlim(1.6, 5.7)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    ax.set_ylim(-6, 19)
    fig.savefig(OUT / "fig1_severity_curve.png")
    plt.close(fig)


def fig_compound_vs_single():
    d = json.loads((ROOT / "reports/severity_curve_results.json").read_text())
    pairs = (("haze+glare", "haze + glare"), ("haze+cloud", "haze + cloud"))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    for ax, (pair, label) in zip(axes, pairs):
        cells = sorted([c for c in d["cells"] if c["pair"] == pair], key=lambda c: c["severity"])
        xs = [c["severity"] for c in cells]
        w = 0.36
        ax.bar([x - w / 2 for x in xs], [c["drop_pp_mean"] for c in cells], width=w,
               color=BLUE, label="compound drop")
        ax.bar([x + w / 2 for x in xs], [c["max_single_pp_mean"] for c in cells], width=w,
               color=ORANGE, label="largest single drop")
        ax.set_xticks(xs)
        ax.set_title(label, color=INK, fontsize=9, pad=6)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("accuracy drop (pp)")
    axes[1].set_xlabel("severity")
    axes[1].legend(frameon=False, loc="upper left", fontsize=8)
    fig.savefig(OUT / "fig2_compound_vs_single.png")
    plt.close(fig)


CANDIDATES = [
    ("T1 consistency (40 ep)", -0.18, -1.13, 0.76),
    ("I1 conditional norm", 0.94, -0.09, 1.98),
    ("R3 feature denoise", -0.69, -1.76, 0.35),
    ("R1 multi-scale", -1.10, -2.23, 0.05),
    ("I2 Retinex", -2.56, -3.86, -1.22),
    ("T2 group DRO", -5.04, -6.52, -3.54),
    ("T3 curriculum", -8.76, -10.14, -7.40),
    ("T4 22K init", -0.71, -2.07, 0.52),
]


def fig_candidates():
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.axvline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
    for i, (name, mean, lo, hi) in enumerate(CANDIDATES):
        ax.plot([lo, hi], [i, i], color=BLUE, lw=1.4, solid_capstyle="round")
        ax.plot(mean, i, "o", color=BLUE, ms=6)
        ax.annotate(f"{mean:+.2f}", (mean, i), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7.5, color=MUTED)
    ax.set_yticks(range(len(CANDIDATES)))
    ax.set_yticklabels([c[0] for c in CANDIDATES])
    ax.invert_yaxis()
    ax.set_xlabel("mean change vs baseline, combined conditions (pp)")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "fig3_candidates.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_severity_curve()
    fig_compound_vs_single()
    fig_candidates()
    print("wrote", *[p.name for p in sorted(OUT.glob('*.png'))])
