#!/usr/bin/env python
"""Cross-model degradation analysis on the validation split.

Inputs (all produced by scripts/efmnet_eval.py and scripts/baseline_val_conditions.py):
    reports/conditions_val/<model>/<condition>.jsonl     7 frozen classifiers, project pipeline
    reports/efmnet_val_conditions.json[_samples.json]    EFM-Net, project pipeline
    reports/efmnet_val_conditions_native.json[...]       EFM-Net, its own preprocessing

Outputs:
    reports/cross_model_degradation.json

Everything is paired per image: the same source image, corruption parameters and
seed are used for every model, so a difference is a within-image difference.

Groups for the bootstrap are `source_group_id` (duplicate components / scenes),
not individual images, because near-duplicate images are not independent.
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

FAMILY = {"convnext_tiny_combo_s17": "convnext_tiny",
          "convnext_tiny_combo_s29": "convnext_tiny",
          "convnext_tiny_combo_s43": "convnext_tiny",
          "swin_t_combo_s17": "swin_t",
          "swin_t_combo_s29": "swin_t",
          "swin_t_combo_s43": "swin_t",
          "resnet50_combo_s17": "resnet50",
          "efmnet": "efmnet_convnext_b"}
FOCUS = ("matched_haze_5", "shifted_haze_5", "matched_glare_5", "matched_cloud_5",
         "matched_haze+glare_5", "shifted_haze+glare_5", "matched_haze+cloud_5")
REPS = 2000


def load_frozen(rows):
    """model -> condition -> array of predictions in manifest order."""
    root = PROJECT / "reports/conditions_val"
    out = {}
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        per_condition = {}
        for path in sorted(model_dir.glob("*.jsonl")):
            lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            by_id = {line["sample_id"]: line["prediction"] for line in lines}
            missing = [r["sample_id"] for r in rows if r["sample_id"] not in by_id]
            if missing:
                raise SystemExit(f"{path} missing {len(missing)} samples")
            per_condition[path.stem] = by_id
        out[model_dir.name] = per_condition
    return out


def load_efmnet(rows, path):
    samples = json.loads((PROJECT / path.replace(".json", "_samples.json")).read_text())
    per_condition = {}
    for condition, block in samples.items():
        preds = block["pred"]
        if len(preds) != len(rows):
            raise SystemExit(f"{condition}: length mismatch")
        per_condition[condition] = {r["sample_id"]: p for r, p in zip(rows, preds)}
    return per_condition


def correctness(preds, rows):
    return np.array([int(preds[r["sample_id"]] == r["label"]) for r in rows], dtype=float)


def group_bootstrap(delta, groups, reps=REPS, seed=0):
    """Resample source groups; return (mean, lo, hi) in pp."""
    uniq = {g: np.where(groups == g)[0] for g in np.unique(groups)}
    keys = list(uniq)
    rng = np.random.default_rng(seed)
    stats = np.empty(reps)
    for i in range(reps):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([uniq[keys[j]] for j in pick])
        stats[i] = 100 * delta[idx].mean()
    return 100 * delta.mean(), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main():
    manifest = PROJECT / "data/manifests/fgsc23_research_civilian.jsonl"
    rows = [r for r in read_jsonl(manifest) if r["split"] == "val"]
    groups = np.array([r["source_group_id"] for r in rows])
    labels = np.array([r["label"] for r in rows])

    frozen = load_frozen(rows)
    efmnet = {"efmnet": load_efmnet(rows, "reports/efmnet_val_conditions.json")}
    efmnet_native = {"efmnet_native": load_efmnet(rows, "reports/efmnet_val_conditions_native.json")}

    models = {}
    for source in (frozen, efmnet):
        models.update(source)
    conditions = [c for c in models["efmnet"] if c != "clean"]

    analysis = {"n": len(rows), "groups": int(len(np.unique(groups))),
                "reps": REPS, "conditions": conditions,
                "models": {}, "families": {}, "gate": {}, "focus": {}}

    correctness_by_model = {}
    for name, per_condition in models.items():
        clean = correctness(per_condition["clean"], rows)
        correctness_by_model[name] = {"clean": clean}
        entry = {"clean_accuracy": float(clean.mean()), "conditions": {}}
        for condition in conditions:
            degraded = correctness(per_condition[condition], rows)
            correctness_by_model[name][condition] = degraded
            mean, lo, hi = group_bootstrap(degraded - clean, groups)
            entry["conditions"][condition] = {"accuracy": float(degraded.mean()),
                                              "delta_pp": mean, "ci_lo": lo, "ci_hi": hi}
        analysis["models"][name] = entry

    # family-level: average correctness over seeds inside a family
    family_members = {}
    for name in correctness_by_model:
        family_members.setdefault(FAMILY[name], []).append(name)
    for family, members in family_members.items():
        clean = np.mean([correctness_by_model[m]["clean"] for m in members], axis=0)
        entry = {"members": members, "clean_accuracy": float(clean.mean()), "conditions": {}}
        for condition in conditions:
            degraded = np.mean([correctness_by_model[m][condition] for m in members], axis=0)
            mean, lo, hi = group_bootstrap(degraded - clean, groups)
            entry["conditions"][condition] = {"accuracy": float(degraded.mean()),
                                              "delta_pp": mean, "ci_lo": lo, "ci_hi": hi}
        analysis["families"][family] = entry

    for condition in conditions:
        passed, near = [], []
        for family, entry in analysis["families"].items():
            stat = entry["conditions"][condition]
            if stat["delta_pp"] <= -5.0 and stat["ci_hi"] < 0:
                passed.append(family)
            elif stat["delta_pp"] <= -5.0:
                near.append(family)
        analysis["gate"][condition] = {"n_pass": len(passed), "pass": passed, "near_miss": near,
                                       "n_families": len(analysis["families"])}

    # focus conditions: transitions, per-class, confidence
    for condition in FOCUS:
        block = {}
        for name in ("efmnet",):
            clean_pred = np.array([models[name]["clean"][r["sample_id"]] for r in rows])
            deg_pred = np.array([models[name][condition][r["sample_id"]] for r in rows])
            fixed = (clean_pred != labels) & (deg_pred == labels)
            broken = (clean_pred == labels) & (deg_pred != labels)
            transitions = Counter((int(labels[i]), int(deg_pred[i])) for i in np.where(broken)[0])
            per_class = {int(c): float(np.mean(deg_pred[labels == c] == c))
                         for c in sorted(set(labels.tolist()))}
            per_class_clean = {int(c): float(np.mean(clean_pred[labels == c] == c))
                               for c in sorted(set(labels.tolist()))}
            block[name] = {
                "broken": int(broken.sum()), "fixed": int(fixed.sum()),
                "top_transitions": [[int(t), int(p), int(n)]
                                    for (t, p), n in transitions.most_common(8)],
                "per_class_recall": per_class,
                "per_class_recall_clean": per_class_clean,
            }
        # all-family failure: every family wrong on this image
        family_wrong = []
        for family, members in family_members.items():
            family_wrong.append(np.mean([correctness_by_model[m][condition] for m in members], axis=0) < 0.5)
        all_wrong = np.all(family_wrong, axis=0)
        block["all_families_wrong"] = int(all_wrong.sum())
        block["all_families_wrong_examples"] = [
            {"sample_id": rows[i]["sample_id"], "path": rows[i]["path"],
             "label": int(labels[i])}
            for i in np.where(all_wrong)[0][:10]]
        analysis["focus"][condition] = block

    # confidence shift (EFM-Net, project pipeline)
    samples = json.loads((PROJECT / "reports/efmnet_val_conditions_samples.json").read_text())
    confidence = {}
    for condition in ["clean"] + list(FOCUS):
        block = samples[condition]
        probs = np.array(block["probs"])
        y = np.array(block["label"])
        pred = np.array(block["pred"])
        confidence[condition] = {"mean_max_prob": float(probs.max(1).mean()),
                                 "mean_max_prob_correct": float(probs.max(1)[pred == y].mean()),
                                 "mean_max_prob_wrong": float(probs.max(1)[pred != y].mean())
                                 if (pred != y).any() else None}
    analysis["confidence_efmnet"] = confidence

    # preprocessing ablation for EFM-Net
    native = {}
    for condition in ["clean"] + conditions:
        block = efmnet_native["efmnet_native"][condition]
        native[condition] = float(correctness(block, rows).mean())
    project = {c: float(analysis["models"]["efmnet"]["conditions"][c]["accuracy"])
               for c in conditions}
    project_clean = analysis["models"]["efmnet"]["clean_accuracy"]
    ablation = {"clean": {"project": project_clean, "native": native["clean"]},
                "conditions": {}}
    for condition in conditions:
        delta_project = project[condition] - project_clean
        delta_native = native[condition] - native["clean"]
        ablation["conditions"][condition] = {
            "accuracy_project": project[condition], "accuracy_native": native[condition],
            "delta_project_pp": 100 * delta_project, "delta_native_pp": 100 * delta_native,
            "delta_difference_pp": 100 * (delta_native - delta_project)}
    analysis["preprocess_ablation"] = ablation

    write_json(PROJECT / "reports/cross_model_degradation.json", analysis)
    passing = {c: v["n_pass"] for c, v in analysis["gate"].items()}
    print(json.dumps({"families": list(analysis["families"]),
                      "max_n_pass": max(passing.values()),
                      "conditions_at_max": [c for c, n in passing.items() if n == max(passing.values())],
                      "clean": {k: round(v["clean_accuracy"], 4)
                                for k, v in analysis["models"].items()}},
                     indent=1))
    print(json.dumps({"preprocess_clean": ablation["clean"]}, indent=1))


if __name__ == "__main__":
    main()
