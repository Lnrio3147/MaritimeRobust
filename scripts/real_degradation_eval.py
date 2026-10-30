#!/usr/bin/env python
"""Stratified evaluation of frozen classifiers on reviewed real degradations.

Reads confirmed per-image labels and cached per-sample predictions.  Every
metric is reported per stratum, per classifier, and averaged over classifier
seeds per backbone.  Confidence intervals are group bootstraps over source
images, not over samples; the strata are observational, not randomized.

Metric definitions (also written into the report):
  accuracy      = correct / N_stratum
  balanced acc. = mean over labels present in the stratum of recall_c
  macro-F1      = unweighted mean of F1_c over labels present in the stratum
  delta vs clear = 100 * (metric_stratum - metric_clear), same classifier
Confidence intervals: 95% percentile, 2000 group-bootstrap replicates,
resampling source_group_id with replacement inside each stratum.  Strata with
N < 15 or fewer than 8 source groups are marked exploratory and get no delta CI.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maritime_robust.common import ROOT as PROJECT, digest, read_jsonl, write_json  # noqa: E402

CIVILIAN = {14: 0, 15: 1, 17: 2, 18: 3, 19: 4, 20: 5, 21: 6, 22: 7}
CLASSES = {0: "container ship", 1: "car carrier", 2: "bulk carrier", 3: "oil tanker",
           4: "fishing boat", 5: "passenger ship", 6: "liquefied gas ship", 7: "barge"}
TYPES = ("cloud", "haze", "shadow", "glare", "blur")
SEVERITIES = ("mild", "medium", "severe")


def label_of(meta):
    if meta.get("label") is not None:
        return int(meta["label"])
    mapping = meta.get("mapping") or {}
    original = mapping.get("fgsc23_original_label")
    if original is None or int(original) not in CIVILIAN:
        raise ValueError(f"{meta['sample_id']}: cannot derive civilian label")
    return CIVILIAN[int(original)]


def strata_for(row):
    types = row["degradation_types"]
    severity = row["severity"]
    clear = not types and severity == "none"
    out = {}
    if clear:
        out["clear"] = True
    if types:
        out["any_degraded"] = True
    for t in TYPES:
        if t in types:
            out[f"type:{t}"] = True
    for s in SEVERITIES:
        if severity == s:
            out[f"severity:{s}"] = True
    if len(types) >= 2:
        out["multi_type"] = True
    if row.get("clutter"):
        out["clutter"] = True
    if clear and not row.get("clutter"):
        out["clear_no_clutter"] = True
    return out


def scores(y, p, classes):
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    if n == 0:
        return None
    correct = int((y == p).sum())
    recalls, f1s, present = {}, {}, []
    for c in classes:
        mask = y == c
        support = int(mask.sum())
        if not support:
            continue
        present.append(c)
        tp = int(((y == c) & (p == c)).sum())
        fp = int(((y != c) & (p == c)).sum())
        fn = support - tp
        recalls[str(c)] = tp / support
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1s[str(c)] = (2 * precision * tp / (precision + tp)) if tp else 0.0
    return {
        "n": n, "correct": correct, "accuracy": correct / n,
        "balanced_accuracy": float(np.mean(list(recalls.values()))) if recalls else None,
        "macro_f1": float(np.mean(list(f1s.values()))) if f1s else None,
        "per_class_recall": recalls, "support": {str(c): int((y == c).sum()) for c in present},
        "absent_classes": [c for c in classes if c not in present],
    }


def bootstrap(y, p, groups, classes, reps, rng):
    if reps <= 0 or len(y) < 2:
        return None
    by_group = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    keys = list(by_group)
    if len(keys) < 2:
        return None
    acc, bal = [], []
    for _ in range(reps):
        chosen = rng.choice(len(keys), size=len(keys), replace=True)
        idx = [i for k in chosen for i in by_group[keys[k]]]
        s = scores([y[i] for i in idx], [p[i] for i in idx], classes)
        acc.append(s["accuracy"])
        bal.append(s["balanced_accuracy"])
    return {"accuracy_ci": [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))],
            "balanced_accuracy_ci": [float(np.percentile(bal, 2.5)),
                                     float(np.percentile(bal, 97.5))]}


def boot_stat(y, p, groups, classes, reps, rng, key):
    by_group = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    keys = list(by_group)
    out = []
    for _ in range(reps):
        chosen = rng.choice(len(keys), size=len(keys), replace=True)
        idx = [i for k in chosen for i in by_group[keys[k]]]
        s = scores([y[i] for i in idx], [p[i] for i in idx], classes)
        out.append(s[key])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("fgsc23_val", "fgsrcs_dev"), required=True)
    parser.add_argument("--labels", default="")
    parser.add_argument("--predictions", default="")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", default="")
    parser.add_argument("--unconfirmed-ok", action="store_true",
                        help="smoke tests only; results are not reportable")
    args = parser.parse_args()

    root = PROJECT / f"reviews/real_degradation/{args.dataset}"
    labels_path = Path(args.labels) if args.labels else root / "labels.jsonl"
    pred_path = Path(args.predictions) if args.predictions else (
        PROJECT / f"reports/real_degradation_predictions_{args.dataset}.jsonl")
    labels = {r["sample_id"]: r for r in read_jsonl(labels_path)}
    predictions = read_jsonl(pred_path)
    if not labels:
        raise ValueError("no labels")
    if not args.unconfirmed_ok and not (root / "labels_summary.json").exists():
        raise ValueError("labels_summary.json missing: merge confirmed labels first")

    by_classifier = defaultdict(dict)
    for r in predictions:
        by_classifier[r["classifier"]][r["sample_id"]] = r
    missing = [(c, s) for c, rows in by_classifier.items() for s in labels if s not in rows]
    if missing:
        raise ValueError(f"predictions missing for {len(missing)} classifier/sample pairs")

    sample_ids = sorted(labels)
    y = np.array([label_of(labels[s]) for s in sample_ids])
    groups = [labels[s]["source_group_id"] for s in sample_ids]
    memberships = [strata_for(labels[s]) for s in sample_ids]
    classes = sorted(set(int(v) for v in y))
    rng = np.random.default_rng(17)
    reps = args.bootstrap if not args.unconfirmed_ok else min(args.bootstrap, 200)

    result = {
        "dataset": args.dataset, "n": len(sample_ids), "replicable": not args.unconfirmed_ok,
        "labels": str(labels_path.relative_to(PROJECT)) if labels_path.is_relative_to(PROJECT) else str(labels_path),
        "labels_sha256": digest(labels_path),
        "predictions": str(pred_path.relative_to(PROJECT)) if pred_path.is_relative_to(PROJECT) else str(pred_path),
        "predictions_sha256": digest(pred_path),
        "class_names": {str(c): CLASSES[c] for c in classes},
        "label_distribution": {
            "clear": sum(1 for m in memberships if "clear" in m),
            "degraded": sum(1 for m in memberships if "any_degraded" in m),
            "unlabeled_ambiguous": sum(1 for r in labels.values()
                                       if not r["degradation_types"] and r["severity"] != "none"),
            "types": {t: sum(1 for r in labels.values() if t in r["degradation_types"])
                      for t in TYPES},
            "severity": {s: sum(1 for r in labels.values() if r["severity"] == s)
                         for s in ("none",) + SEVERITIES + ("unable",)},
            "clutter": sum(1 for r in labels.values() if r.get("clutter")),
        },
        "classifiers": {},
        "backbones": {},
        "definitions": {
            "accuracy": "correct / N_stratum",
            "balanced_accuracy": "mean of per-class recall over labels present in stratum",
            "macro_f1": "unweighted mean F1 over labels present in stratum",
            "delta_vs_clear_pp": "100 * (metric_stratum - metric_clear), same classifier",
            "ci": "95% percentile group bootstrap over source_group_id, resampling groups within stratum",
            "exploratory": "N < 15 or < 8 source groups; no inferential claim",
            "strata": "observational, not randomized; clutter is a scene factor, not degradation",
        },
    }

    clear_idx = [i for i, m in enumerate(memberships) if "clear" in m]
    clear_y = y[clear_idx]
    clear_groups = [groups[i] for i in clear_idx]
    clear_scores = {c: scores(clear_y, np.array([by_classifier[c][sample_ids[i]]["prediction"]
                                                 for i in clear_idx]), classes)
                    for c in by_classifier}
    clear_ci = {c: bootstrap(clear_y, np.array([by_classifier[c][sample_ids[i]]["prediction"]
                                               for i in clear_idx]), clear_groups, classes, reps, rng)
                for c in by_classifier}

    for name in sorted(by_classifier):
        rows = by_classifier[name]
        p = np.array([rows[s]["prediction"] for s in sample_ids])
        entry = {"strata": {}}
        for stratum in sorted({k for m in memberships for k in m}):
            idx = [i for i, m in enumerate(memberships) if stratum in m]
            if not idx:
                continue
            sy, sp = y[idx], p[idx]
            sg = [groups[i] for i in idx]
            s = scores(sy, sp, classes)
            ci = bootstrap(sy, sp, sg, classes, reps, rng)
            n_groups = len(set(sg))
            entry["strata"][stratum] = {
                **s, "groups": n_groups,
                "exploratory": len(idx) < 15 or n_groups < 8,
                "accuracy_ci": ci["accuracy_ci"] if ci else None,
                "balanced_accuracy_ci": ci["balanced_accuracy_ci"] if ci else None,
                "delta_vs_clear_pp": None, "delta_ci_pp": None,
            }
            if stratum != "clear" and clear_scores[name] and len(clear_idx) >= 2:
                delta = 100 * (s["accuracy"] - clear_scores[name]["accuracy"])
                entry["strata"][stratum]["delta_vs_clear_pp"] = delta
                if (ci and clear_ci[name] and len(sg) >= 8 and
                        len(set(clear_groups)) >= 8):
                    a = boot_stat(sy, sp, sg, classes, reps, rng, "accuracy")
                    b = boot_stat(clear_y,
                                  np.array([by_classifier[name][sample_ids[i]]["prediction"]
                                            for i in clear_idx]),
                                  clear_groups, classes, reps, rng, "accuracy")
                    diff = 100 * (np.array(a) - np.array(b))
                    entry["strata"][stratum]["delta_ci_pp"] = [
                        float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
        result["classifiers"][name] = entry

    for name in result["classifiers"]:
        backbone, _, seed = name.rpartition("_s")
        agg = result["backbones"].setdefault(backbone, {"seeds": [], "strata": {}})
        agg["seeds"].append(int(seed))
    for backbone, agg in result["backbones"].items():
        names = [f"{backbone}_s{s}" for s in sorted(agg["seeds"])]
        strata = sorted({k for n in names for k in result["classifiers"][n]["strata"]})
        for stratum in strata:
            vals = [result["classifiers"][n]["strata"].get(stratum) for n in names]
            vals = [v for v in vals if v]
            if not vals:
                continue
            deltas = [v["delta_vs_clear_pp"] for v in vals if v["delta_vs_clear_pp"] is not None]
            agg["strata"][stratum] = {
                "n": vals[0]["n"], "groups": vals[0]["groups"],
                "accuracy_mean": float(np.mean([v["accuracy"] for v in vals])),
                "balanced_accuracy_mean": float(np.mean([v["balanced_accuracy"] for v in vals])),
                "macro_f1_mean": float(np.mean([v["macro_f1"] for v in vals])),
                "delta_mean": float(np.mean(deltas)) if deltas else None,
                "delta_min": float(np.min(deltas)) if deltas else None,
                "delta_max": float(np.max(deltas)) if deltas else None,
                "exploratory": any(v["exploratory"] for v in vals),
            }

    output = Path(args.output) if args.output else (
        PROJECT / f"reports/real_degradation_stratified_{args.dataset}.json")
    if not output.is_absolute():
        output = PROJECT / output
    write_json(output, result)
    print(json.dumps({"dataset": args.dataset, "n": result["n"],
                      "classifiers": len(result["classifiers"]),
                      "strata": sorted(result["backbones"][next(iter(result["backbones"]))]["strata"]),
                      "output": str(output.relative_to(PROJECT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
