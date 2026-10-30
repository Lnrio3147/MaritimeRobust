#!/usr/bin/env python
"""Paired evaluation for the M2 drift-weighting candidate.

Scores two runs on the identical 47 conditions of one fold's val rows: the 43
registered conditions plus the four matched compound conditions at severity 2
and 4 that ``severity_curve_eval.EXTRA`` defines. Every number is paired per
image, so the treatment can be differenced against the baseline directly.

    python scripts/m2_eval.py --baseline runs/fgsc23_civ8_fold0/convnext_tiny_none_e40_s17 \
        --treatment runs/fgsc23_civ8_fold0/convnext_tiny_d1_e40_s17 --seed-tag d1_s17

The primary metric is ``combo_low``: mean accuracy over the six matched
compound conditions at severity <= 4 (see reports/PREREG_DRIFT_WEIGHTING.md).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT, ROOT / "scripts"):
    sys.path.insert(0, str(path))

import explore_eval as ee  # noqa: E402
import m1_eval as me  # noqa: E402
import severity_curve_eval as sce  # noqa: E402
from maritime_robust.common import ROOT as PROJECT, read_jsonl, write_json  # noqa: E402
from maritime_robust.engine import metrics  # noqa: E402

OUT = "reports/m2"


def conditions_47():
    """The 43 registered conditions plus the four matched s=2/s=4 compounds."""
    conds = ee.all_conditions()
    seen = {c["name"] for c in conds}
    conds += [dict(c) for c in sce.EXTRA if c["name"] not in seen]
    return conds


def cluster_bootstrap(labels, base, treat, groups, index, resamples=2000, seed=17):
    """95% CI for the ``index`` mean accuracy difference, resampling source groups.

    ``index`` is a list of condition names; the statistic is the mean over those
    conditions of (treatment accuracy − baseline accuracy).
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    deltas = []
    for _ in range(resamples):
        take = np.concatenate([members[g] for g in rng.choice(unique, len(unique),
                                                              replace=True)])
        total = 0.0
        for name in index:
            total += (treat[name][take] == labels[take]).mean() \
                     - (base[name][take] == labels[take]).mean()
        deltas.append(total / len(index))
    deltas = np.array(deltas) * 100
    return {"mean_pp": float(deltas.mean()),
            "ci95_pp": [float(np.percentile(deltas, 2.5)),
                        float(np.percentile(deltas, 97.5))],
            "p_two_sided": float(2 * min((deltas <= 0).mean(), (deltas >= 0).mean())),
            "resamples": resamples}


def mcnemar(labels, base, treat, names):
    """Exact McNemar over the pooled (image, condition) pairs of ``names``."""
    b = c = 0
    for name in names:
        base_ok = base[name] == labels
        treat_ok = treat[name] == labels
        b += int(np.sum(base_ok & ~treat_ok))
        c += int(np.sum(~base_ok & treat_ok))
    discordant = b + c
    p = 1.0 if discordant == 0 else float(
        stats.binomtest(min(b, c), discordant, 0.5).pvalue)
    return {"baseline_only_correct": b, "treatment_only_correct": c,
            "pairs": int(len(labels) * len(names)), "p_exact": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--treatment", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--seed-tag", required=True, help="output stem, e.g. d1_s17")
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--dump-preds", default=None)
    ap.add_argument("--resamples", type=int, default=2000)
    args = ap.parse_args()

    base_run, treat_run = PROJECT / args.baseline, PROJECT / args.treatment
    base_model, base_settings = ee.load_any(base_run)
    treat_model, treat_settings = ee.load_any(treat_run)
    if base_settings.get("method") != "none":
        raise SystemExit(f"baseline arm is method={base_settings.get('method')}")
    if treat_settings.get("method") not in ("d1", "d1inv"):
        raise SystemExit(f"treatment arm is method={treat_settings.get('method')}")
    if treat_settings["manifest_sha256"] != base_settings["manifest_sha256"]:
        raise SystemExit("the two arms were trained on different manifests")

    manifest = PROJECT / args.manifest if args.manifest else Path(base_settings["manifest"])
    rows = [r for r in read_jsonl(manifest) if r["split"] == "val"]
    labels = np.array([r["label"] for r in rows])
    groups = np.array([r["source_group_id"] for r in rows])
    contents = [ee.content(r) for r in rows]
    n = base_settings["num_classes"]
    conds = conditions_47()

    base = me.run_predictions(base_model, rows, contents, conds, with_mask=False)
    treat = me.run_predictions(treat_model, rows, contents, conds, with_mask=False)

    def acc(preds, name):
        return metrics(labels.tolist(), preds[name].tolist(), n)["accuracy"]

    cells = {}
    for cond in conds:
        name = cond["name"]
        a, b = acc(base, name), acc(treat, name)
        cells[name] = {"kind": cond["kind"], "severity": cond["severity"],
                       "domain": cond["domain"], "layer": cond["layer"],
                       "baseline_accuracy": a, "treatment_accuracy": b,
                       "delta_pp": 100 * (b - a),
                       "prediction_agreement": float(np.mean(base[name] == treat[name])),
                       **mcnemar(labels, base, treat, [name])}

    def group(names):
        sel = [n_ for n_ in names if n_ in cells]
        return {
            "n": len(sel), "conditions": sorted(sel),
            "baseline_mean_accuracy": float(np.mean([cells[n_]["baseline_accuracy"] for n_ in sel])),
            "treatment_mean_accuracy": float(np.mean([cells[n_]["treatment_accuracy"] for n_ in sel])),
            "delta_pp": float(np.mean([cells[n_]["delta_pp"] for n_ in sel])),
            "mcnemar_pooled": mcnemar(labels, base, treat, sel),
            "cluster_bootstrap": cluster_bootstrap(labels, base, treat, groups,
                                                   sel, resamples=args.resamples),
        }

    combos = [k for k, v in cells.items() if v["layer"] == "combo"]
    combo_low = [k for k in combos if cells[k]["severity"] <= 4 and cells[k]["domain"] == "matched"]
    combo_low_all = [k for k in combos if cells[k]["severity"] <= 4]
    singles = [k for k, v in cells.items() if v["layer"] == "single"]

    out = {
        "baseline_run": args.baseline, "treatment_run": args.treatment,
        "method": treat_settings.get("method"), "alpha": treat_settings.get("alpha"),
        "teacher_run": treat_settings.get("teacher_run"),
        "manifest": str(manifest), "n_val": len(rows),
        "seed": treat_settings.get("seed"), "epochs": treat_settings.get("epochs"),
        "best_val_macro_f1": treat_settings.get("best_val_macro_f1"),
        "cells": cells,
        "summary": {
            "combo_low": group(combo_low),
            "combo_low_all": group(combo_low_all),
            "combo_all": group(combos),
            "combo_sev5_matched": group([k for k in combos if cells[k]["severity"] == 5
                                         and cells[k]["domain"] == "matched"]),
            "single": group(singles),
            "clean": group(["clean"]),
            "all": group(list(cells)),
        },
    }
    out_dir = PROJECT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"{args.seed_tag}.json", out)

    if args.dump_preds:
        np.savez_compressed(
            PROJECT / args.dump_preds, labels=labels, groups=groups,
            conditions=np.array([c["name"] for c in conds]),
            baseline=np.stack([base[c["name"]] for c in conds]),
            treatment=np.stack([treat[c["name"]] for c in conds]))

    print(f"{'condition':24s} {'base':>6s} {'treat':>6s} {'delta':>7s} {'p':>8s}")
    for name in sorted(cells):
        c = cells[name]
        print(f"{name:24s} {100*c['baseline_accuracy']:6.2f} {100*c['treatment_accuracy']:6.2f} "
              f"{100*(c['treatment_accuracy']-c['baseline_accuracy']):+7.1f} {c['p_exact']:8.4f}")
    print()
    for key, s in out["summary"].items():
        lo, hi = s["cluster_bootstrap"]["ci95_pp"]
        print(f"{key:20s} n={s['n']:2d} base {100*s['baseline_mean_accuracy']:6.2f} "
              f"treat {100*s['treatment_mean_accuracy']:6.2f} delta {s['delta_pp']:+6.2f} "
              f"CI[{lo:+.2f},{hi:+.2f}] p_boot={s['cluster_bootstrap']['p_two_sided']:.4f} "
              f"p_mcnemar={s['mcnemar_pooled']['p_exact']:.4f}")


if __name__ == "__main__":
    main()
