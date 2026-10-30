"""Reanalysis of stored predictions: is the compound-degradation effect just severity?

Uses only the 43-condition evaluation JSONs already on disk (no training, no new
forward passes). Three questions, all on the baseline arm (convnext_tiny, 40 ep):

  A. Severity coverage  - does any single degradation reach the compound drop?
  B. Accuracy-space additivity - is the E2 margin-space result an artifact of
     the margin metric? (recomputed here in pp of accuracy)
  C. Failure novelty - are compound errors the same errors a severe single
     degradation already makes? Share of compound errors that are (i) also
     produced by the strongest single condition, (ii) not produced by any
     single condition of either kind at any severity.

    python scripts/compound_reanalysis.py
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S2B = ROOT / "reports/s2b"
FOLDS = (0, 1, 2)
SEEDS = (17, 29)
SINGLE_KINDS = ("haze", "cloud", "glare")
SEVERITIES = (1, 2, 3, 4, 5)
PAIRS = (("haze", "glare"), ("haze", "cloud"))
PAIR_SEVERITIES = (3, 5)
REPS = 4000
RNG = random.Random(20261030)


def load_eval(fold: int, seed: int) -> dict:
    path = S2B / f"fgsc23_civ8_fold{fold}__convnext_tiny_none_e40_s{seed}.json"
    return json.loads(path.read_text())


def val_labels(fold: int) -> list[int]:
    rows = [json.loads(line) for line in
            (ROOT / f"data/manifests/folds/fgsc23_civ8_fold{fold}.jsonl").read_text().splitlines()]
    val = [r for r in rows if r["split"] == "val"]
    return val


def boot_mean(values: list[float], clusters: list[str], reps: int = REPS) -> tuple[float, float, float]:
    """Cluster bootstrap over source_group_id; seeds of the same image share a cluster."""
    by = {}
    for v, c in zip(values, clusters):
        by.setdefault(c, []).append(v)
    keys = list(by)
    means = []
    for _ in range(reps):
        sample = [by[RNG.choice(keys)] for _ in keys]
        flat = [x for s in sample for x in s]
        means.append(sum(flat) / len(flat))
    means.sort()
    point = sum(values) / len(values)
    return point, means[int(0.025 * reps)], means[int(0.975 * reps)]


def main() -> None:
    # Per (fold, seed): correctness arrays for every condition.
    data = {}
    for fold in FOLDS:
        rows = val_labels(fold)
        labels = [r["label"] for r in rows]
        groups = [r["source_group_id"] for r in rows]
        for seed in SEEDS:
            ev = load_eval(fold, seed)
            conds = ev["conditions"]
            assert ev["n"] == len(labels), (fold, seed, ev["n"], len(labels))
            correct = {k: [int(p == y) for p, y in zip(v["predictions"], labels)] for k, v in conds.items()}
            data[(fold, seed)] = {"correct": correct, "groups": groups, "labels": labels}

    report = {"n_runs": len(data), "n_val": len(labels)}

    # ---- A. severity coverage -------------------------------------------------
    coverage = []
    for pair in PAIRS:
        for sev in PAIR_SEVERITIES:
            cell = f"matched_{pair[0]}+{pair[1]}_{sev}"
            for (fold, seed), d in data.items():
                c = d["correct"]
                n = len(c[cell])
                base_acc = sum(c["clean"]) / n
                combo_drop = 100 * (base_acc - sum(c[cell]) / n)
                singles = [f"matched_{k}_{s}" for k in pair for s in SEVERITIES]
                max_single = max(100 * (base_acc - sum(c[k]) / n) for k in singles)
                coverage.append({"cell": cell, "fold": fold, "seed": seed,
                                 "combo_drop_pp": combo_drop, "max_single_drop_pp": max_single,
                                 "exceeds": combo_drop > max_single})
    report["A_severity_coverage"] = coverage
    report["A_summary"] = {
        "cells_where_compound_exceeds_all_single": sum(r["exceeds"] for r in coverage),
        "cells_total": len(coverage),
    }

    # ---- B. accuracy-space additivity ----------------------------------------
    add_rows = []
    for pair in PAIRS:
        for sev in PAIR_SEVERITIES:
            cell = f"matched_{pair[0]}+{pair[1]}_{sev}"
            for (fold, seed), d in data.items():
                c = d["correct"]
                n = len(labels)
                base = 100 * sum(c["clean"]) / n
                drop_a = base - 100 * sum(c[f"matched_{pair[0]}_{sev}"]) / n
                drop_b = base - 100 * sum(c[f"matched_{pair[1]}_{sev}"]) / n
                observed = base - 100 * sum(c[cell]) / n
                add_rows.append({"cell": cell, "fold": fold, "seed": seed,
                                 "observed_drop_pp": observed, "additive_pred_pp": drop_a + drop_b,
                                 "excess_pp": observed - (drop_a + drop_b)})
    report["B_additivity_accuracy_space"] = add_rows
    agg = {}
    for cell in sorted({r["cell"] for r in add_rows}):
        vals = [r["excess_pp"] for r in add_rows if r["cell"] == cell]
        agg[cell] = {"mean_excess_pp": sum(vals) / len(vals), "per_run": vals}
    report["B_summary"] = agg

    # ---- C. failure novelty ---------------------------------------------------
    novelty = []
    for pair in PAIRS:
        for sev in PAIR_SEVERITIES:
            cell = f"matched_{pair[0]}+{pair[1]}_{sev}"
            strongest = max((f"matched_{k}_{s}" for k in pair for s in SEVERITIES),
                            key=lambda k: -sum(sum(dd["correct"][k]) for dd in data.values()))
            # per (fold, seed): a compound error counts as novel only if no single
            # condition of either kind, at any severity, was wrong on that image
            # under the same seed.
            singles = [f"matched_{kk}_{s}" for kk in pair for s in SEVERITIES]
            for fold in FOLDS:
                n = len(labels)
                n_err = novel = also_strong = 0
                for seed in SEEDS:
                    c = data[(fold, seed)]["correct"]
                    for i in range(n):
                        if c[cell][i]:
                            continue
                        n_err += 1
                        if all(c[k][i] for k in singles):
                            novel += 1
                        if not c[strongest][i]:
                            also_strong += 1
                if n_err == 0:
                    continue
                novelty.append({"cell": cell, "fold": fold, "strongest_single": strongest,
                                "n_combo_errors": n_err,
                                "novel_share": novel / n_err,
                                "explained_by_strongest_share": also_strong / n_err,
                                "novel_n": novel, "explained_n": also_strong})
    report["C_novelty"] = novelty
    summary_c = {}
    for cell in sorted({r["cell"] for r in novelty}):
        rows = [r for r in novelty if r["cell"] == cell]
        n_err = sum(r["n_combo_errors"] for r in rows)
        summary_c[cell] = {
            "n_combo_errors": n_err,
            "novel_share": round(sum(r["novel_n"] for r in rows) / n_err, 3),
            "explained_by_strongest_share": round(sum(r["explained_n"] for r in rows) / n_err, 3),
        }
    report["C_summary"] = summary_c

    # ---- D. cluster-bootstrap intervals -------------------------------------
    # Per-sample excess (accuracy space) and novelty indicators, grouped by
    # (fold, source_group_id) and averaged over seeds before resampling.
    boot = {}
    for pair in PAIRS:
        for sev in PAIR_SEVERITIES:
            cell = f"matched_{pair[0]}+{pair[1]}_{sev}"
            excess_vals, excess_cl = [], []
            novel_vals, novel_cl = [], []
            for fold in FOLDS:
                d = data[(fold, SEEDS[0])]
                n = len(labels)
                groups = d["groups"]
                singles = [f"matched_{k}_{s}" for k in pair for s in SEVERITIES]
                for i in range(n):
                    ex = 0.0
                    for seed in SEEDS:
                        c = data[(fold, seed)]["correct"]
                        e_c = 1 - c[cell][i]
                        e_a = 1 - c[f"matched_{pair[0]}_{sev}"][i]
                        e_b = 1 - c[f"matched_{pair[1]}_{sev}"][i]
                        e_clean = 1 - c["clean"][i]
                        ex += 100 * (e_c - e_a - e_b + e_clean) / len(SEEDS)
                    excess_vals.append(ex)
                    excess_cl.append((fold, groups[i]))
                    nov = 0.0
                    for seed in SEEDS:
                        c = data[(fold, seed)]["correct"]
                        if c[cell][i]:
                            continue
                        if all(c[k][i] for k in singles):
                            nov += 1 / len(SEEDS)
                    novel_vals.append(nov)
                    novel_cl.append((fold, groups[i]))
            # excess: per-sample value already scaled so its mean is pp
            p, lo, hi = boot_mean(excess_vals, excess_cl)
            n_err = sum(novel_vals)
            boot[cell] = {
                "excess_pp": [round(p, 2), round(lo, 2), round(hi, 2)],
                "novel_errors_per_eval_set": round(n_err, 1),
            }
    report["D_bootstrap"] = boot

    out = ROOT / "reports/compound_analysis.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"A": report["A_summary"], "B": {k: round(v["mean_excess_pp"], 2) for k, v in agg.items()},
                      "C": summary_c}, indent=1))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
