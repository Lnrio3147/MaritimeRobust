"""Compare the two additivity arms on the compound conditions.

a0 is the control, so the only meaningful comparison is a1 - a0 *within a cell*
(same fold, same seed). Anything a1 gains over the plain baseline could equally
come from the extra compound supervision that a0 also receives.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]

ARMS = [("baseline", "reports/overlap"),
        ("a0 (control)", "reports/overlap_a0"),
        ("a1 (additivity)", "reports/overlap_a1")]

FOCUS = [("haze+glare", 5), ("haze+cloud", 5), ("haze+glare", 4)]


def load(out_dir):
    cells = {}
    for path in sorted((PROJECT / out_dir).glob("fold*_s*.json")):
        cells[path.stem] = json.loads(path.read_text())
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/ADDITIVE.md")
    args = ap.parse_args()

    data = {name: load(d) for name, d in ARMS}
    data = {n: c for n, c in data.items() if c}
    if "a0 (control)" not in data or "a1 (additivity)" not in data:
        raise SystemExit("both a0 and a1 must be evaluated first")

    lines = ["# Additivity-constrained training", "",
             "delta = e(a+b) - |E_a ∪ E_b|; acc = accuracy under the named condition.",
             "Means over the cells present for each arm (fold0/1/2 × seed17/29).", ""]

    # Clean accuracy first: a gain there would confound every compound number.
    lines += ["## Clean accuracy", "", "| arm | clean acc | matched_haze_5 | matched_glare_5 |",
              "|---|---:|---:|---:|"]
    for name, cells in data.items():
        clean = np.mean([c["accuracy"]["clean"] for c in cells.values()])
        h5 = np.mean([c["accuracy"]["matched_haze_5"] for c in cells.values()])
        g5 = np.mean([c["accuracy"]["matched_glare_5"] for c in cells.values()])
        lines.append(f"| {name} | {clean:.2f} | {h5:.2f} | {g5:.2f} |")

    lines += ["", "## Compound accuracy and excess", "",
              "| arm | " + " | ".join(f"{p}_{s} acc | delta" for p, s in FOCUS) + " |",
              "|---|" + "---:|---:|" * len(FOCUS)]

    summary = {}
    for name, cells in data.items():
        row = [name]
        for pair, sev in FOCUS:
            key = f"matched_{pair}_{sev}"
            acc = np.mean([c["accuracy"][key] for c in cells.values()])
            d = [r["delta_pp"] for c in cells.values() for r in c["pairs"]
                 if r["pair"] == pair and r["severity"] == sev]
            summary.setdefault(f"{pair}_{sev}", {})[name] = {
                "accuracy": float(acc), "delta_mean": float(np.mean(d)) if d else None,
                "delta_values": [round(x, 2) for x in d]}
            row += [f"{acc:.2f}", f"{np.mean(d):+.1f}" if d else "—"]
        lines.append("| " + " | ".join(row) + " |")

    # Paired comparison against the control, cell by cell.
    lines += ["", "## Paired a1 - a0 per cell", ""]
    a0, a1 = data["a0 (control)"], data["a1 (additivity)"]
    for key_label, getter in (
            ("matched_haze+glare_5 acc", lambda c: c["accuracy"]["matched_haze+glare_5"]),
            ("matched_haze+cloud_5 acc", lambda c: c["accuracy"]["matched_haze+cloud_5"]),
            ("clean acc", lambda c: c["accuracy"]["clean"])):
        diffs = [100 * (getter(a1[k]) - getter(a0[k])) for k in sorted(set(a0) & set(a1))]
        if diffs:
            lines.append(f"- **{key_label}**: mean {np.mean(diffs):+.2f} pp "
                         f"({min(diffs):+.1f} .. {max(diffs):+.1f}), "
                         f"{sum(d > 0 for d in diffs)}/{len(diffs)} positive")

    for pair, sev in FOCUS:
        d0 = [r["delta_pp"] for c in a0.values() for r in c["pairs"]
              if r["pair"] == pair and r["severity"] == sev]
        d1 = [r["delta_pp"] for c in a1.values() for r in c["pairs"]
              if r["pair"] == pair and r["severity"] == sev]
        if d0 and d1:
            lines.append(f"- **{pair}_{sev} delta**: a0 {np.mean(d0):+.1f} -> "
                         f"a1 {np.mean(d1):+.1f} ({np.mean(d1) - np.mean(d0):+.1f})")

    (PROJECT / args.out).write_text("\n".join(lines) + "\n")
    (PROJECT / "reports/additive.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
