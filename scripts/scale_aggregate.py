"""Capacity axis: does the compound excess shrink as the backbone grows?

Reads the 47-condition overlap decompositions produced for each backbone and
reports delta = e(a+b) - |E_a u E_b| per (pair, severity), averaged over the
cells available for that backbone.

The question is not "is accuracy higher" — a bigger backbone will be better on
single degradations by construction. It is whether the *excess*, which is
measured against the model's own marginal damage, shrinks with capacity.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]

BACKBONES = [
    ("convnext_tiny", "reports/overlap"),
    ("convnext_small", "reports/overlap_convnext_small"),
    ("convnext_base", "reports/overlap_convnext_base"),
    ("convnext_large", "reports/overlap_convnext_large"),
]

# Frozen-feature probes. Same JSON schema, produced by linear_probe_grid.py.
# Kept in a separate table because a probe and a fine-tuned net are not points
# on the same axis: the probe's clean accuracy is much lower by construction.
PROBES = [
    ("convnext_tiny", "reports/probe_convnext_tiny"),
    ("convnext_small", "reports/probe_convnext_small"),
    ("convnext_base", "reports/probe_convnext_base"),
]


def load(out_dir):
    cells = defaultdict(list)
    for path in sorted((PROJECT / out_dir).glob("fold*_s*.json")):
        data = json.loads(path.read_text())
        for r in data["pairs"]:
            cells[(r["pair"], r["severity"])].append(r["delta_pp"])
    return cells


def load_raw(directory):
    """Per-cell JSON, for the fields the delta table does not need."""
    return {p.stem: json.loads(p.read_text())
            for p in sorted((PROJECT / directory).glob("fold*_s*.json"))}


def _mean_err_union(directory, pair, severity):
    vals = [r["error_union_pp"] for p in (PROJECT / directory).glob("fold*_s*.json")
            for r in json.loads(p.read_text())["pairs"]
            if r["pair"] == pair and r["severity"] == severity]
    return float(np.mean(vals)) if vals else float("nan")


def _mean_delta(directory, pair, severity):
    vals = [r["delta_pp"] for p in (PROJECT / directory).glob("fold*_s*.json")
            for r in json.loads(p.read_text())["pairs"]
            if r["pair"] == pair and r["severity"] == severity]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/SCALE.md")
    ap.add_argument("--json", default="reports/scale.json")
    args = ap.parse_args()

    present = [(name, load(d)) for name, d in BACKBONES]
    present = [(n, c) for n, c in present if c]
    if not present:
        raise SystemExit("no overlap results found")

    lines = ["# Capacity axis: compound excess versus backbone size", "",
             "delta = e(a+b) - |E_a ∪ E_b|, matched-marginal protocol, mean over cells "
             "(fold0/1/2 × seed17/29).", ""]
    header = "| pair | sev | " + " | ".join(f"{n} (n={len(c[(list(c)[0][0], list(c)[0][1])])})"
                                            for n, c in present) + " |"
    keys = sorted({k for _, c in present for k in c})
    lines.append("| pair | sev | " + " | ".join(n for n, _ in present) + " |")
    lines.append("|---|---:|" + "---:|" * len(present))
    payload = {}
    for key in keys:
        row = [f"{key[0]}", str(key[1])]
        entry = {}
        for name, cells in present:
            vals = cells.get(key, [])
            if vals:
                row.append(f"{np.mean(vals):+.1f} ({len(vals)})")
                entry[name] = {"mean": float(np.mean(vals)), "n": len(vals),
                               "values": [round(v, 2) for v in vals]}
            else:
                row.append("—")
        payload[f"{key[0]}_{key[1]}"] = entry
        lines.append("| " + " | ".join(row) + " |")

    sev5 = [k for k in keys if k[1] == 5]
    lines += ["", "## Severity-5 excess only", ""]
    for key in sev5:
        vals = {n: payload[f"{key[0]}_{key[1]}"].get(n) for n, _ in present}
        series = [(n, v["mean"]) for n, v in vals.items() if v]
        if len(series) >= 2:
            trend = ", ".join(f"{n.split('_')[-1]} {v:+.1f}" for n, v in series)
            lines.append(f"- **{key[0]}**: {trend}")

    # --- frozen features: capacity without the 412-image bottleneck
    probes = [(name, d, load(d), load_raw(d)) for name, d in PROBES]
    probes = [(n, d, c, raw) for n, d, c, raw in probes if c]
    if probes:
        fine_clean = {}
        for name, directory in BACKBONES:
            raw = load_raw(directory)
            if raw:
                fine_clean[name] = np.mean([100 * v["clean_accuracy"] for v in raw.values()])
        lines += ["", "## Frozen-feature probe", "",
                  "A linear head cannot overfit 412 images the way a 28M-parameter "
                  "fine-tune can, so a probe isolates whether the excess lives in the "
                  "representation itself.", "",
                  "**delta is not comparable across models of different accuracy.** Its "
                  "ceiling is the remaining headroom ``100 - err_union``, and a weaker "
                  "model exhausts more of it on the single corruptions. The normalised "
                  "column divides by that headroom; read the capacity claim off *that*, "
                  "not off the raw delta.", "",
                  "| backbone | probe clean | fine-tuned clean | hg5 delta | hg5 headroom | "
                  "hg5 normalised | hc5 delta | cells |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, directory, cells, raw in probes:
            hg = cells.get(("haze+glare", 5), [])
            hc = cells.get(("haze+cloud", 5), [])
            err_union = _mean_err_union(directory, "haze+glare", 5)
            norm = np.mean(hg) / (100 - err_union) if err_union and hg else float("nan")
            lines.append(
                f"| {name} | {np.mean([100 * v['clean_accuracy'] for v in raw.values()]):.1f} | "
                f"{fine_clean.get(name, float('nan')):.1f} | {np.mean(hg):+.1f} | "
                f"{100 - err_union:.1f} | {100 * norm:.1f}% | {np.mean(hc):+.1f} | {len(raw)} |")
        lines += ["", "Fine-tuned tiny, same normalisation: "
                  f"{100 * _mean_delta('reports/overlap', 'haze+glare', 5) / (100 - _mean_err_union('reports/overlap', 'haze+glare', 5)):.1f}%."]

    (PROJECT / args.out).write_text("\n".join(lines) + "\n")
    (PROJECT / args.json).write_text(json.dumps(payload, indent=1) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
