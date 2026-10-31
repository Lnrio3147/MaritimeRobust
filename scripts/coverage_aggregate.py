"""Coverage study: does a wider pair pool shrink the compound excess on held-out pairs?

Two numbers per arm, both already defined elsewhere:

  held-out delta   mean delta over the three lattice pairs missing from every
                   training pool, from ``lattice_eval``
  haze+glare_5     the paper's transfer target, from ``overlap_main_grid``

The arm comparison is seeded/folded, so paired differences are computed within a
cell (same fold, same seed) before averaging. The pre-registration in
``reports/PREREG_COVERAGE.md`` fixes what counts as support for the hypothesis.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]

ARMS = [("baseline (k=0)", "reports/lattice_baseline", "reports/overlap"),
        ("c3", "reports/lattice_c3", "reports/overlap_c3"),
        ("c6", "reports/lattice_c6", "reports/overlap_c6"),
        ("c12", "reports/lattice_c12", "reports/overlap_c12"),
        ("c12b (lattice+recipe)", "reports/lattice_c12b", "reports/overlap_c12b"),
        ("s6 (singles control)", "reports/lattice_s6", "reports/overlap_s6"),
        ("s6b (singles+recipe)", "reports/lattice_s6b", "reports/overlap_s6b"),
        ("s6h (damage-matched singles)", "reports/lattice_s6h", "reports/overlap_s6h")]


def load(directory):
    return {p.stem: json.loads(p.read_text())
            for p in sorted((PROJECT / directory).glob("fold*_s*.json"))}


def held_out_delta(cells):
    out = {}
    for cell, data in cells.items():
        vals = [r["delta_pp"] for r in data["pairs"] if r["held_out"]]
        if vals:
            out[cell] = float(np.mean(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/COVERAGE.md")
    args = ap.parse_args()

    lat = {name: load(d) for name, d, _ in ARMS}
    grid = {name: load(d) for name, _, d in ARMS}
    lat = {n: c for n, c in lat.items() if c}
    if not lat:
        raise SystemExit("no lattice results found")

    lines = ["# Coverage study", "",
             "Nested pair pools over the six calibrated families; three pairs held out of "
             "every pool. All arms apply a compound to half the batch and a clean view to "
             "the other half, so compound exposure is constant and only pair diversity "
             "varies.", ""]

    # --- held-out lattice pairs
    held = {n: held_out_delta(c) for n, c in lat.items()}
    base = held.get("baseline (k=0)", {})
    lines += ["## Held-out lattice pairs", "",
              "| arm | held-out delta (pp) | clean acc | vs baseline (paired, pp) |",
              "|---|---:|---:|---:|"]
    for name in lat:
        clean = np.mean([100 * c["clean_accuracy"] for c in lat[name].values()])
        mean = np.mean(list(held[name].values())) if held[name] else float("nan")
        shared = sorted(set(held[name]) & set(base))
        diff = (np.mean([held[name][k] - base[k] for k in shared])
                if shared and name != "baseline (k=0)" else float("nan"))
        lines.append(f"| {name} | {mean:+.1f} | {clean:.2f} | "
                     f"{'—' if np.isnan(diff) else f'{diff:+.1f}'} |")

    # --- the load-bearing comparison: pairing removed, everything else equal
    for singles_arm, pair_arm in (("s6 (singles control)", "c12"),
                                  ("s6b (singles+recipe)", "c12b (lattice+recipe)"),
                                  ("s6h (damage-matched singles)", "c3")):
        if singles_arm not in lat or pair_arm not in lat:
            continue
        if singles_arm not in held or pair_arm not in held:
            continue
        a_cells, b_cells = held[singles_arm], held[pair_arm]
        shared = sorted(set(a_cells) & set(b_cells))
        diff = [b_cells[k] - a_cells[k] for k in shared]
        clean_a = np.mean([100 * c["clean_accuracy"] for c in lat[singles_arm].values()])
        clean_b = np.mean([100 * c["clean_accuracy"] for c in lat[pair_arm].values()])
        lines += ["", f"## Pairing versus mere exposure ({pair_arm} - {singles_arm})", "",
                  f"- held-out delta: **{np.mean(diff):+.2f} pp** "
                  f"({min(diff):+.1f} .. {max(diff):+.1f}), "
                  f"{sum(d < 0 for d in diff)}/{len(diff)} cells favour {pair_arm}",
                  f"- clean accuracy: {clean_b - clean_a:+.2f} pp "
                  f"({pair_arm} {clean_b:.2f} vs {singles_arm} {clean_a:.2f})"]
        for label, want_held in (("held-out pairs", True), ("pool pairs", False)):
            vals = {}
            for arm in (singles_arm, pair_arm):
                vals[arm] = {cell: float(np.mean([r["delta_pp"] for r in data["pairs"]
                                                  if r["held_out"] == want_held]))
                             for cell, data in lat[arm].items()}
            cells = sorted(set(vals[singles_arm]) & set(vals[pair_arm]))
            d = [vals[pair_arm][k] - vals[singles_arm][k] for k in cells]
            lines.append(f"- {label} delta difference: **{np.mean(d):+.2f} pp** "
                         f"over {len(d)} cells")

    # --- per-pair, so a single held-out pair cannot carry the mean
    lines += ["", "## Per-pair held-out delta", "",
              "| pair | " + " | ".join(lat) + " |",
              "|---|" + "---:|" * len(lat)]
    pairs = sorted({r["pair"] for cells in lat.values() for c in cells.values()
                    for r in c["pairs"] if r["held_out"]})
    for pair in pairs:
        row = [pair]
        for name in lat:
            vals = [r["delta_pp"] for c in lat[name].values() for r in c["pairs"]
                    if r["pair"] == pair]
            row.append(f"{np.mean(vals):+.1f}" if vals else "—")
        lines.append("| " + " | ".join(row) + " |")

    # --- transfer to the paper grid
    lines += ["", "## Transfer: paper grid (never in any lattice)", "",
              "| arm | haze+glare_5 acc | delta | haze+cloud_5 acc | delta | clean acc |",
              "|---|---:|---:|---:|---:|---:|"]
    transfer = {}
    for name, cells in grid.items():
        if not cells:
            continue
        first = next(iter(cells.values()))
        needed = {"clean", "matched_haze+glare_5", "matched_haze+cloud_5"}
        if not needed <= set(first["accuracy"]):
            continue
        acc = {k: np.mean([100 * c["accuracy"][k] for c in cells.values()]) for k in needed}
        d = {}
        for pair in ("haze+glare", "haze+cloud"):
            vals = [r["delta_pp"] for c in cells.values() for r in c["pairs"]
                    if r["pair"] == pair and r["severity"] == 5]
            d[pair] = float(np.mean(vals)) if vals else float("nan")
        transfer[name] = {"clean": acc["clean"], **d}
        lines.append(f"| {name} | {acc['matched_haze+glare_5']:.1f} | {d['haze+glare']:+.1f} | "
                     f"{acc['matched_haze+cloud_5']:.1f} | {d['haze+cloud']:+.1f} | "
                     f"{acc['clean']:.1f} |")

    (PROJECT / args.out).write_text("\n".join(lines) + "\n")
    (PROJECT / "reports/coverage.json").write_text(json.dumps(
        {"held_out_delta": held, "transfer": transfer}, indent=1) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
