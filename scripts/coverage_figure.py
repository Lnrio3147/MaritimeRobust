"""Figure for the coverage study: does pair diversity remove the compound excess?

Two panels, both on the held-out lattice pairs (three pairs absent from every
training pool, so nothing here can be fitted):

  (a) per-arm held-out delta, one dot per (fold, seed) cell, arm mean as a bar
  (b) the same-pool comparison that removes "mere exposure" as an explanation:
      paired minus singles, cell by cell, for the two recipe-matched arm pairs

Reads only the JSON written by ``lattice_eval.py``; no model is loaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]

ARMS = [("baseline\n(k=0)", "reports/lattice_baseline"),
        ("c3", "reports/lattice_c3"),
        ("c6", "reports/lattice_c6"),
        ("c12", "reports/lattice_c12"),
        ("c12b\n(+recipe)", "reports/lattice_c12b"),
        ("s6\n(singles)", "reports/lattice_s6"),
        ("s6b\n(singles+recipe)", "reports/lattice_s6b"),
        ("s6h\n(damage-matched)", "reports/lattice_s6h")]

CONTRASTS = [("c12", "reports/lattice_c12", "s6", "reports/lattice_s6",
              "pair pool vs singles"),
             ("c12b", "reports/lattice_c12b", "s6b", "reports/lattice_s6b",
              "pair pool vs singles, base recipe kept")]


def load(directory):
    path = PROJECT / directory
    if not path.is_dir():
        return {}
    return {p.stem: json.loads(p.read_text())
            for p in sorted(path.glob("fold*_s*.json"))}


def held(cells):
    return {k: float(np.mean([r["delta_pp"] for r in v["pairs"] if r["held_out"]]))
            for k, v in cells.items() if any(r["held_out"] for r in v["pairs"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figures/fig5_coverage.png")
    args = ap.parse_args()

    arms = [(label, held(load(d))) for label, d in ARMS]
    arms = [(label, h) for label, h in arms if h]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.6),
                                 gridspec_kw={"width_ratios": [1.35, 1]})
    for i, (label, h) in enumerate(arms):
        vals = np.array(list(h.values()))
        ax.barh(i, vals.mean(), height=.55, color="#c6dbef", edgecolor="#2171b5", lw=.8)
        jitter = np.linspace(-.16, .16, len(vals))
        ax.scatter(vals, i + jitter, s=13, color="#08519c", zorder=3, lw=0)
        # label on whichever side the dots leave free
        if vals.max() <= 11:
            ax.text(vals.max() + 0.7, i, f"{vals.mean():+.1f}", va="center",
                    ha="left", fontsize=7)
        else:
            ax.text(vals.min() - 0.7, i, f"{vals.mean():+.1f}", va="center",
                    ha="right", fontsize=7)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([a[0] for a in arms], fontsize=7)
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("held-out pair delta (pp)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title("(a) excess on pairs never trained", fontsize=8)
    ax.set_xlim(ax.get_xlim()[0] - 2.2, ax.get_xlim()[1] + 2.2)

    for i, (a_label, a_dir, b_label, b_dir, note) in enumerate(CONTRASTS):
        a, b = held(load(a_dir)), held(load(b_dir))
        shared = sorted(set(a) & set(b))
        if not shared:
            continue
        diff = np.array([a[k] - b[k] for k in shared])
        colour, edge = ("#fdd0a2", "#a63603") if i == 0 else ("#c7e9c0", "#238b45")
        bx.bar(i, diff.mean(), width=.5, color=colour, edgecolor=edge, lw=.8, zorder=1)
        bx.scatter(i + np.linspace(-.14, .14, len(diff)), diff, s=13, color=edge,
                   zorder=3, lw=0)
        bx.text(i, diff.max() + 0.7,
                f"{diff.mean():+.1f} pp\n{int((diff < 0).sum())}/{len(diff)} cells",
                ha="center", va="bottom", fontsize=7)
        print(f"{a_label} - {b_label}: {diff.mean():+.2f} pp over {len(diff)} cells "
              f"({diff.min():+.1f} .. {diff.max():+.1f}), {note}")
    bx.axhline(0, color="k", lw=.8)
    bx.set_xticks(range(len(CONTRASTS)))
    bx.set_xticklabels(["no base\nrecipe", "base recipe\nkept"], fontsize=7)
    bx.set_xlim(-.6, len(CONTRASTS) - .4)
    lo, hi = bx.get_ylim()
    bx.set_ylim(lo - .6, hi + 2.4)
    bx.set_ylabel("paired $-$ singles (pp)", fontsize=8)
    bx.tick_params(axis="y", labelsize=7)
    bx.set_title("(b) pairing, exposure held fixed", fontsize=8)

    for axis in (ax, bx):
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = PROJECT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
