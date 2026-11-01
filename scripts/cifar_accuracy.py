"""Absolute accuracy per condition family for every CIFAR-10-C arm.

The excess measures (delta, super) are relative: they say how a compound error
compares with baselines built from the same model's single-corruption errors.
They do not say how well the arm actually classifies a compound image, which is
the number a reader compares against published robustness results. This script
reports that number directly from the stored per-image flags:

    clean       the 10,000-image clean test set
    singles     mean over the 75 single-corruption conditions
    pairs       mean over the 525 pair conditions
    heldout s3  the five pairs no training pool contains, at severity 3

and the matched control is the clean reference model on the same cells.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from maritime_robust.external.cifar_c import HELD_OUT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label=reports/cifar_grid/<run>")
    ap.add_argument("--out", default="reports/CIFAR_ACCURACY.md")
    ap.add_argument("--json", default="reports/cifar_accuracy.json")
    args = ap.parse_args()
    held = {frozenset(p) for p in HELD_OUT}

    rows, payload = [], {}
    for spec in args.runs:
        label, directory = spec.split("=", 1)
        d = PROJECT / directory
        npz = np.load(d / "correct.npz")
        n = json.loads((d / "summary.json").read_text())["n"]
        ok = {k: np.unpackbits(npz[k], count=n).astype(bool) for k in npz.files}

        def mean(key_pred):
            vals = [v.mean() for k, v in ok.items() if key_pred(k)]
            return 100 * float(np.mean(vals)) if vals else float("nan")

        entry = {
            "clean": mean(lambda k: "+" not in k and k.startswith("clean")),
            "singles": mean(lambda k: "+" not in k and not k.startswith("clean")),
            "pairs": mean(lambda k: "+" in k),
            "heldout_s3": mean(lambda k: "+" in k and k.endswith("_3")
                               and frozenset(k.rsplit("_", 1)[0].split("+")) in held),
        }
        payload[label] = entry
        rows.append((label, entry))

    lines = ["# CIFAR-10-C: absolute accuracy per condition family", "",
             "Percent correct, averaged over the conditions in each family. The excess "
             "measures elsewhere in this project are relative to the same model's own "
             "singles; this table is the absolute number.", "",
             "| arm | clean | singles | pairs | held-out pairs s3 |", "|---|---:|---:|---:|---:|"]
    base = next((e for lbl, e in rows if lbl.startswith("clean")), None)
    for label, e in rows:
        lines.append(f"| {label} | {e['clean']:.2f} | {e['singles']:.2f} | "
                     f"{e['pairs']:.2f} | {e['heldout_s3']:.2f} |")
    if base:
        lines += ["", "Relative to the clean reference model, on the same cells:", "",
                  "| arm | Δ pairs | Δ held-out pairs s3 | Δ clean |", "|---|---:|---:|---:|"]
        for label, e in rows:
            if label.startswith("clean"):
                continue
            lines.append(f"| {label} | {e['pairs'] - base['pairs']:+.2f} | "
                         f"{e['heldout_s3'] - base['heldout_s3']:+.2f} | "
                         f"{e['clean'] - base['clean']:+.2f} |")
    (PROJECT / args.out).write_text("\n".join(lines) + "\n")
    (PROJECT / args.json).write_text(json.dumps(payload, indent=1) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
