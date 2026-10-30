#!/usr/bin/env python
"""Failure-mechanism statistics behind reports/G2_P0_FAILURE_MECHANISM.md.

Reads the per-sample probability vectors saved by
    scripts/baseline_val_conditions.py   -> reports/conditions_val/<model>/<condition>.jsonl
    scripts/efmnet_eval.py               -> reports/efmnet_val_conditions_samples.json
and writes reports/failure_analysis.json.

Contents
  groups      predefined single / combo / placement groups (no post-hoc selection)
  fdr         Benjamini-Hochberg over all family x condition tests
  margin      true-class confidence and top-1 minus runner-up margin per condition
  combos      extra margin loss of a combo beyond the additive effect of its parts
  allwrong    samples wrong in all four families, per condition
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from maritime_robust.common import ROOT as PROJECT, read_jsonl, write_json  # noqa: E402

FAMILY = {"convnext_tiny_combo_s17": "convnext_tiny", "convnext_tiny_combo_s29": "convnext_tiny",
          "convnext_tiny_combo_s43": "convnext_tiny", "swin_t_combo_s17": "swin_t",
          "swin_t_combo_s29": "swin_t", "swin_t_combo_s43": "swin_t",
          "resnet50_combo_s17": "resnet50", "efmnet": "efmnet_convnext_b"}
REPS = 2000
SEED = 0


def load_probabilities(rows):
    """family -> condition -> (n, 8) mean probability over seeds."""
    collected = {}
    for model_dir in sorted(p for p in (PROJECT / "reports/conditions_val").iterdir() if p.is_dir()):
        for path in sorted(model_dir.glob("*.jsonl")):
            lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
            by_id = {line["sample_id"]: line["probabilities"] for line in lines}
            collected.setdefault(FAMILY[model_dir.name], {}).setdefault(path.stem, []).append(
                np.array([by_id[r["sample_id"]] for r in rows]))
    efmnet = json.loads((PROJECT / "reports/efmnet_val_conditions_samples.json").read_text())
    for condition, block in efmnet.items():
        collected.setdefault("efmnet_convnext_b", {}).setdefault(condition, []).append(
            np.array(block["probs"]))
    return {family: {cond: np.mean(stack, axis=0) for cond, stack in conditions.items()}
            for family, conditions in collected.items()}


def margins(probs, labels):
    """true-class probability and top-1 minus runner-up margin (per sample)."""
    true_p = probs[np.arange(len(labels)), labels]
    ordered = np.sort(probs, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    predicted = probs.argmax(1)
    signed = np.where(predicted == labels, margin, -margin)
    return true_p, signed


def group_bootstrap(delta, groups, reps=REPS, seed=SEED):
    uniq = {g: np.where(groups == g)[0] for g in np.unique(groups)}
    keys = list(uniq)
    rng = np.random.default_rng(seed)
    stats = np.empty(reps)
    for i in range(reps):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([uniq[keys[j]] for j in pick])
        stats[i] = 100 * delta[idx].mean()
    return 100 * delta.mean(), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)), stats


def bh(pvalues, alpha=0.05):
    order = np.argsort(pvalues)
    m = len(pvalues)
    picked, threshold = [], alpha
    for rank, index in enumerate(order, start=1):
        if pvalues[index] <= alpha * rank / m:
            picked.append(index)
            threshold = alpha * rank / m
    rejected = [False] * m
    cutoff = max([alpha * (r + 1) / m for r, i in enumerate(order) if i in picked], default=0.0)
    for i, p in enumerate(pvalues):
        rejected[i] = p <= cutoff
    return rejected, cutoff


def group_of(condition):
    if condition == "clean":
        return None
    if condition.startswith("matched_") and condition.endswith(("_center", "_border")):
        return "placement"
    parts = condition.split("_")
    domain, kind = parts[0], parts[1]
    if "+" in kind:
        return "combo_both" if domain in ("matched", "shifted") else None
    return f"single_{domain}" if domain in ("matched", "shifted") else None


def main():
    rows = [r for r in read_jsonl(PROJECT / "data/manifests/fgsc23_research_civilian.jsonl")
            if r["split"] == "val"]
    labels = np.array([r["label"] for r in rows])
    groups = np.array([r["source_group_id"] for r in rows])
    families = load_probabilities(rows)
    conditions = [c for c in families["efmnet_convnext_b"] if c != "clean"]

    out = {"n": len(rows), "groups": int(len(np.unique(groups))), "reps": REPS,
           "family_conditions": {}, "condition_stats": {}, "fdr": {}, "groups_summary": {},
           "combos": {}, "allwrong": {}}

    # ---- per family / condition -------------------------------------------
    tests = []
    per_sample_margin = {}
    for family, table in families.items():
        clean_prob = table["clean"]
        _, clean_margin = margins(clean_prob, labels)
        clean_correct = (clean_prob.argmax(1) == labels).astype(float)
        clean_true_p, clean_margin_signed = margins(clean_prob, labels)
        out["family_conditions"][family] = {"clean": {
            "accuracy": float(clean_correct.mean()), "delta_pp": 0.0,
            "mean_true_class_prob": float(clean_true_p.mean()),
            "mean_margin": float(clean_margin_signed.mean()), "margin_delta": 0.0,
            "bh_significant": False}}
        per_sample_margin[family] = {}
        for condition in conditions:
            prob = table[condition]
            true_p, margin = margins(prob, labels)
            correct = (prob.argmax(1) == labels).astype(float)
            per_sample_margin[family][condition] = margin
            delta = correct - clean_correct
            mean, lo, hi, stats = group_bootstrap(delta, groups)
            p_one_sided = float((stats >= 0).mean())
            tests.append({"family": family, "condition": condition, "p": p_one_sided})
            out["family_conditions"][family][condition] = {
                "accuracy": float(correct.mean()), "delta_pp": mean, "ci_lo": lo, "ci_hi": hi,
                "p_one_sided": p_one_sided,
                "mean_true_class_prob": float(true_p.mean()),
                "mean_margin": float(margin.mean()),
                "margin_delta": float(margin.mean() - clean_margin.mean()),
            }

    # ---- BH-FDR over all family x condition tests --------------------------
    pvalues = np.array([t["p"] for t in tests])
    rejected, cutoff = bh(pvalues)
    for test, reject in zip(tests, rejected):
        test["bh_significant"] = bool(reject)
        out["family_conditions"][test["family"]][test["condition"]]["bh_significant"] = bool(reject)
    out["fdr"] = {"n_tests": len(tests), "alpha": 0.05, "cutoff": cutoff,
                  "n_significant": int(sum(rejected)),
                  "expected_false_positives_without_correction": 0.05 * len(tests),
                  "tests": tests}

    # ---- predefined groups -------------------------------------------------
    grouped = Counter()
    for condition in conditions:
        grouped[group_of(condition)] += 1
    for group, count in grouped.items():
        members = [c for c in conditions if group_of(c) == group]
        entry = {"n_conditions": count, "members": members, "families": {}}
        for family in families:
            stats = [out["family_conditions"][family][c] for c in members]
            entry["families"][family] = {
                "mean_delta_pp": float(np.mean([s["delta_pp"] for s in stats])),
                "mean_margin_delta": float(np.mean([s["margin_delta"] for s in stats])),
                "n_bh_significant": int(sum(1 for s in stats if s["bh_significant"])),
            }
        out["groups_summary"][group] = entry

    # ---- extra margin loss of combos --------------------------------------
    for condition in conditions:
        kind = condition.split("_")[1]
        if "+" not in kind:
            continue
        for family in families:
            table = out["family_conditions"][family]
            a, b = kind.split("+")
            domain = condition.split("_")[0]
            severity = "3" if condition.endswith("_3") else "5"
            sa, sb = f"{domain}_{a}_{severity}", f"{domain}_{b}_{severity}"
            if sa not in table or sb not in table:
                continue
            additive = (table["clean"]["mean_margin"] + table[sa]["margin_delta"]
                        + table[sb]["margin_delta"])
            actual = table[condition]["mean_margin"]
            out["combos"].setdefault(condition, {})[family] = {
                "margin_combo": actual, "margin_additive": additive,
                "extra_loss": actual - additive, "accuracy": table[condition]["accuracy"]}

    # ---- samples wrong in every family ------------------------------------
    family_correct = {}
    for family, table in families.items():
        family_correct[family] = {c: (table[c].argmax(1) == labels).astype(float)
                                  for c in ["clean"] + conditions}
    family_names = list(family_correct)
    for condition in conditions:
        wrong_all = np.all([family_correct[f][condition] < 0.5 for f in family_names], axis=0)
        out["allwrong"][condition] = {
            "n": int(wrong_all.sum()),
            "n_clean_correct_for_efmnet": int((wrong_all & (family_correct[
                "efmnet_convnext_b"]["clean"] > 0.5)).sum()),
            "examples": [{"sample_id": rows[i]["sample_id"], "path": rows[i]["path"],
                          "label": int(labels[i])} for i in np.where(wrong_all)[0][:10]],
        }

    # per-sample persistence across the nine conditions that passed the earlier gate
    passing = ["matched_glare_5", "matched_haze+glare_3", "matched_haze+glare_5",
               "matched_haze+cloud_5", "shifted_cloud_5", "shifted_haze+glare_5",
               "shifted_haze+cloud_3", "shifted_haze+cloud_5", "matched_glare_3_center"]
    fails = np.zeros(len(rows))
    for condition in passing:
        wrong_all = np.all([family_correct[f][condition] < 0.5 for f in family_names], axis=0)
        fails += wrong_all
    out["allwrong"]["persistence_in_passing_conditions"] = {
        "conditions": passing,
        "histogram": {int(k): int(v) for k, v in Counter(fails.astype(int)).items()},
        "samples_failing_3_or_more": [
            {"sample_id": rows[i]["sample_id"], "path": rows[i]["path"], "label": int(labels[i]),
             "n_conditions": int(fails[i])} for i in np.where(fails >= 3)[0]],
    }

    write_json(PROJECT / "reports/failure_analysis.json", out)
    print(json.dumps({"n_tests": len(tests), "bh_significant": int(sum(rejected)),
                      "cutoff": cutoff,
                      "groups": {g: v["n_conditions"] for g, v in out["groups_summary"].items()}},
                     indent=1))


if __name__ == "__main__":
    main()
