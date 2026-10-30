"""Four-point severity curve for the compound conditions (pre-registered analysis).

Merges the extra severities 2/4 (reports/severity_curve/) with the existing
43-condition evaluations (reports/s2b/) for the baseline arm, then computes the
quantities defined in reports/PREREG_SEVERITY_2_4.md:

    D(s) observed drop, M(s) max single drop, E(s) accuracy-space excess,
    N(s) share of compound errors no single condition produces.

    python scripts/severity_curve.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDS = (0, 1, 2)
SEEDS = (17, 29)
SINGLES = ("haze", "cloud", "glare")
SEVERITIES = (1, 2, 3, 4, 5)
PAIRS = (("haze", "glare"), ("haze", "cloud"))
CURVE_SEV = (2, 3, 4, 5)
REPS = 4000
RNG = random.Random(20261030)


def boot_ci(values, clusters, reps=REPS):
    by = {}
    for v, c in zip(values, clusters):
        by.setdefault(c, []).append(v)
    keys = list(by)
    means = []
    for _ in range(reps):
        flat = [x for k in (RNG.choice(keys) for _ in keys) for x in by[k]]
        means.append(sum(flat) / len(flat))
    means.sort()
    return sum(values) / len(values), means[int(0.025 * reps)], means[int(0.975 * reps)]


def bh(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg q-values."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    q = [0.0] * n
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, pvals[i] * n / (rank + 1))
        q[i] = prev
    return q


def load_fold(fold):
    rows = [json.loads(l) for l in
            (ROOT / f"data/manifests/folds/fgsc23_civ8_fold{fold}.jsonl").read_text().splitlines()]
    val = [r for r in rows if r["split"] == "val"]
    labels = [r["label"] for r in val]
    groups = [r["source_group_id"] for r in val]
    per_seed = {}
    for seed in SEEDS:
        base = json.loads((ROOT / f"reports/s2b/fgsc23_civ8_fold{fold}__convnext_tiny_none_e40_s{seed}.json").read_text())
        extra = json.loads((ROOT / f"reports/severity_curve/fgsc23_civ8_fold{fold}__convnext_tiny_none_e40_s{seed}.json").read_text())
        conds = dict(base["conditions"])
        for k, v in extra["conditions"].items():
            if k in conds:
                assert abs(conds[k]["accuracy"] - v["accuracy"]) < 1e-12, f"clean/dup mismatch {fold} {seed} {k}"
                continue
            conds[k] = v
        correct = {k: [int(p == y) for p, y in zip(v["predictions"], labels)]
                   for k, v in conds.items()}
        per_seed[seed] = correct
    return labels, groups, per_seed


def main():
    data = {fold: load_fold(fold) for fold in FOLDS}

    rows_out = []
    for pair in PAIRS:
        for sev in CURVE_SEV:
            cell = f"matched_{pair[0]}+{pair[1]}_{sev}"
            ex_vals, ex_cl, novel_num, novel_den = [], [], 0, 0
            per_run = []
            for fold in FOLDS:
                labels, groups, per_seed = data[fold]
                n = len(labels)
                # D, M, E per (fold, seed)
                for seed in SEEDS:
                    c = per_seed[seed]
                    base = sum(c["clean"]) / n
                    d_cell = 100 * (base - sum(c[cell]) / n)
                    max_single = max(100 * (base - sum(c[f"matched_{k}_{s}"]) / n)
                                     for k in pair for s in SEVERITIES)
                    per_run.append({"fold": fold, "seed": seed, "drop_pp": d_cell,
                                    "max_single_pp": max_single})
                # per-sample arrays averaged over seeds
                for i in range(n):
                    ex = 0.0
                    for seed in SEEDS:
                        c = per_seed[seed]
                        e_c = 1 - c[cell][i]
                        e_a = 1 - c[f"matched_{pair[0]}_{sev}"][i]
                        e_b = 1 - c[f"matched_{pair[1]}_{sev}"][i]
                        e_clean = 1 - c["clean"][i]
                        ex += 100 * (e_c - e_a - e_b + e_clean) / len(SEEDS)
                    ex_vals.append(ex)
                    ex_cl.append((fold, groups[i]))
                singles = [f"matched_{k}_{s}" for k in pair for s in SEVERITIES]
                for seed in SEEDS:
                    c = per_seed[seed]
                    for i in range(n):
                        if c[cell][i]:
                            continue
                        novel_den += 1
                        if all(c[k][i] for k in singles):
                            novel_num += 1
            mean, lo, hi = boot_ci(ex_vals, ex_cl)
            rows_out.append({
                "cell": cell, "pair": "+".join(pair), "severity": sev,
                "drop_pp_mean": sum(r["drop_pp"] for r in per_run) / len(per_run),
                "max_single_pp_mean": sum(r["max_single_pp"] for r in per_run) / len(per_run),
                "exceeds_max_single": sum(r["drop_pp"] > r["max_single_pp"] for r in per_run),
                "excess_pp": [round(mean, 2), round(lo, 2), round(hi, 2)],
                "novel_share": round(novel_num / novel_den, 3) if novel_den else None,
                "novel_n": novel_num, "errors_n": novel_den,
                "per_run": per_run,
            })

    # BH over the 8 excess estimates; two-sided normal approx as p-value
    import math
    ps = []
    for r in rows_out:
        m, lo, hi = r["excess_pp"]
        se = (hi - lo) / (2 * 1.96)
        z = m / se if se > 0 else 0.0
        ps.append(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))
    for r, q in zip(rows_out, bh(ps)):
        r["q_value"] = round(q, 4)

    # pre-registered predictions
    checks = {}
    for pair in PAIRS:
        name = "+".join(pair)
        series = [r for r in rows_out if r["pair"] == name]
        series.sort(key=lambda r: r["severity"])
        points = [r["excess_pp"][0] for r in series]
        checks[f"{name}_monotone_pointwise"] = all(
            points[i] <= points[i + 1] + 1e-9 for i in range(len(points) - 1))
        # per-run monotonicity of the excess (P1 uses the 6 individual runs)
        per_run_ok = 0
        run_series = {}
        for fold in FOLDS:
            for seed in SEEDS:
                vals = []
                for r in series:
                    m = [x for x in r["per_run"] if x["fold"] == fold and x["seed"] == seed][0]
                    labels, groups, per_seed = data[fold]
                    c = per_seed[seed]
                    n = len(labels)
                    base = sum(c["clean"]) / n
                    cell = r["cell"]
                    d_cell = 100 * (base - sum(c[cell]) / n)
                    d_a = 100 * (base - sum(c[f"matched_{r['pair'].split('+')[0]}_{r['severity']}"]) / n)
                    d_b = 100 * (base - sum(c[f"matched_{r['pair'].split('+')[1]}_{r['severity']}"]) / n)
                    vals.append(d_cell - d_a - d_b)
                run_series[f"{fold}/{seed}"] = vals
                per_run_ok += all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))
        checks[f"{name}_monotone_per_run"] = f"{per_run_ok}/6"
        checks[f"{name}_run_series"] = run_series
        checks[f"{name}_excess"] = {r["severity"]: r["excess_pp"] for r in series}
        checks[f"{name}_novel"] = {r["severity"]: r["novel_share"] for r in series}

    out = {"prereg": "reports/PREREG_SEVERITY_2_4.md", "cells": rows_out, "checks": checks}
    (ROOT / "reports/severity_curve_results.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for r in rows_out:
        print(f"{r['cell']:26s} drop {r['drop_pp_mean']:5.1f} maxsingle {r['max_single_pp_mean']:5.1f} "
              f"exceeds {r['exceeds_max_single']}/6 excess {r['excess_pp']} novel {r['novel_share']} q={r['q_value']}")
    print(json.dumps(checks["haze+glare_monotone_pointwise"]), json.dumps(checks["haze+cloud_monotone_pointwise"]))


if __name__ == "__main__":
    main()
