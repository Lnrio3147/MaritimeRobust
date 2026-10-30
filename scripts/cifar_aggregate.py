"""The union decomposition on the standard CIFAR-10-C corruption grid.

For each compound ``a+b`` at severity ``s``:

    e_ab     = error rate on a+b
    union    = |E_a u E_b| / N      (paired singles, same images, same severity)
    delta    = 100 * (e_ab - union)  sub-additive if negative

``delta`` is the part of the compound error that neither single explains: it is
not a function of the two marginals, only of how they combine. The maritime
study found this excess vanishes under pair coverage on its six calibrated
families; whether that is a property of the study or of compound corruption in
general is what this grid decides.

Reports the distribution of ``delta`` over all 525 compound cells, its relation
to the composite damage (which decides whether the excess is a damage-only
effect), and the same table for the accuracy-space superadditivity measure
``drop_ab - (drop_a + drop_b)`` that the compound-corruption literature usually
reports, so the two conventions can be compared directly.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def load(directory):
    npz = np.load(PROJECT / directory / "correct.npz")
    summary = json.loads((PROJECT / directory / "summary.json").read_text())
    return {k: np.unpackbits(npz[k], count=summary["n"]).astype(bool) for k in npz.files}


def split(condition):
    """'a+b_s' -> (('a', 'b'), s); 'a_s' -> (('a',), s)."""
    stem, severity = condition.rsplit("_", 1)
    return tuple(stem.split("+")), int(severity)


def decompose(flags):
    rows = []
    for condition in flags:
        if "+" not in condition:
            continue
        (a, b), severity = split(condition)
        key_a, key_b = f"{a}_{severity}", f"{b}_{severity}"
        if key_a not in flags or key_b not in flags:
            continue
        ok_ab, ok_a, ok_b = flags[condition], flags[key_a], flags[key_b]
        n = len(ok_ab)
        err_ab = 1 - ok_ab.mean()
        err_a, err_b = 1 - ok_a.mean(), 1 - ok_b.mean()
        union = 1 - (ok_a & ok_b).mean()
        rows.append({
            "pair": f"{a}+{b}", "a": a, "b": b, "severity": severity,
            "error_ab": float(err_ab), "error_a": float(err_a), "error_b": float(err_b),
            "error_union": float(union),
            # the two conventions, in percentage points
            "delta_pp": float(100 * (err_ab - union)),
            "super_pp": float(100 * (err_ab - err_a - err_b)),
            "damage_pp": float(100 * err_ab),
            "headroom_pp": float(100 * (1 - union)),
            "n": int(n),
        })
    return rows


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


def describe(values):
    v = np.asarray(values)
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
            "min": float(v.min()), "max": float(v.max()),
            "fraction_positive": float((v > 0).mean()),
            "fraction_above_2pp": float((v > 2).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="label=reports/cifar_grid/<run> entries")
    ap.add_argument("--out", default="reports/CIFAR_GRID.md")
    ap.add_argument("--json", default="reports/cifar_grid.json")
    args = ap.parse_args()

    arms = {}
    for spec in args.runs:
        label, directory = spec.split("=", 1)
        arms[label] = decompose(load(directory))
    payload = {}

    lines = ["# CIFAR-10-C: compound excess over the union baseline", "",
             "`delta = e(a+b) - |E_a u E_b|`, 105 unordered pairs x 5 severities, "
             "15000 test images per condition. `super` is the accuracy-space "
             "superadditivity `e(a+b) - e(a) - e(b)` that the compound-corruption "
             "literature reports; the two disagree, and the disagreement is the point.",
             "",
             "| arm | cells | delta mean | median | p10 .. p90 | > 0 | > +2pp | super mean |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]

    for label, rows in arms.items():
        if not rows:
            continue
        d = describe([r["delta_pp"] for r in rows])
        s = np.mean([r["super_pp"] for r in rows])
        payload[label] = {"delta": d, "super_mean": float(s),
                          "severity": defaultdict(list), "rows": rows}
        lines.append(f"| {label} | {len(rows)} | {d['mean']:+.2f} | {d['median']:+.2f} | "
                     f"{d['p10']:+.1f} .. {d['p90']:+.1f} | "
                     f"{100 * d['fraction_positive']:.0f}% | "
                     f"{100 * d['fraction_above_2pp']:.0f}% | {s:+.2f} |")

    # --- is the excess explained by how damaged the compound image is?
    lines += ["", "## Damage law", "",
              "If delta were a function of how much damage the compound carries, a "
              "damage-only predictor would be enough. Spearman correlations over all "
              "cells, with the composite error as the damage measure.", "",
              "| arm | spearman(delta, compound error) | p | spearman(super, compound error) | p |",
              "|---|---:|---:|---:|---:|"]
    for label, rows in arms.items():
        if not rows:
            continue
        r1, p1 = spearman([r["damage_pp"] for r in rows], [r["delta_pp"] for r in rows])
        r2, p2 = spearman([r["damage_pp"] for r in rows], [r["super_pp"] for r in rows])
        payload[label]["damage_law"] = {"delta": [r1, p1], "super": [r2, p2]}
        lines.append(f"| {label} | {r1:+.2f} | {p1:.1e} | {r2:+.2f} | {p2:.1e} |")

    # --- severity structure
    lines += ["", "## By severity", "", "| arm | " + " | ".join(f"s{s}" for s in range(1, 6)) + " |",
              "|---|" + "---:|" * 5]
    for label, rows in arms.items():
        if not rows:
            continue
        means = {s: np.mean([r["delta_pp"] for r in rows if r["severity"] == s])
                 for s in range(1, 6)}
        payload[label]["severity"] = {str(k): float(v) for k, v in means.items()}
        lines.append(f"| {label} | " + " | ".join(f"{means[s]:+.2f}" for s in range(1, 6)) + " |")

    # --- which families carry the excess
    lines += ["", "## By family", "",
              "Mean delta over the ten compounds each family appears in, per severity.",
              "", "| family | " + " | ".join(f"s{s}" for s in range(1, 6)) + " | mean |",
              "|---|" + "---:|" * 6]
    first = next((r for r in arms.values() if r), [])
    families = sorted({r["a"] for r in first} | {r["b"] for r in first})
    for label, rows in arms.items():
        payload[label]["family"] = {}
    for family in families:
        row = [family]
        entry = {}
        for s in range(1, 6):
            vals = [r["delta_pp"] for r in first if r["severity"] == s
                    and family in (r["a"], r["b"])]
            entry[str(s)] = float(np.mean(vals)) if vals else float("nan")
        payload[[k for k in arms][0]]["family"][family] = entry
        row += [f"{entry[str(s)]:+.1f}" for s in range(1, 6)]
        row.append(f"{np.mean(list(entry.values())):+.1f}")
        if len(arms) == 1:
            lines.append("| " + " | ".join(row) + " |")
    if len(arms) > 1:
        lines.append("")
        lines.append("(per-family table written for the first arm only; the arms share "
                     "the condition grid)")

    (PROJECT / args.out).write_text("\n".join(lines) + "\n")
    (PROJECT / args.json).write_text(json.dumps(payload, indent=1, default=float) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
