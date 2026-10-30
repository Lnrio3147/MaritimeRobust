#!/usr/bin/env python
"""Frozen-feature dual probe.

Design, thresholds and the decision ladder are frozen in
`reports/PROBE_PREREGISTRATION.md` BEFORE this script was first run.

  P1  cross-domain probe : linear head fitted on clean train features,
                            tested on degraded val features.
  P2  matched-domain probe: linear head fitted on features of the *same*
                            degradation applied to the 618 train images.
  controls: class-prior floor, label-shuffle null.
  reference: each model's own trained head.

Writes reports/frozen_probe.json and reports/probe_features/*.npz (cache).

    python scripts/frozen_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.models as tvm
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external/efmnet_repro"))

from maritime_robust.common import ROOT as PROJECT, read_jsonl, write_json  # noqa: E402
from maritime_robust.degradations import conditions, corrupt, corruption_seed  # noqa: E402
from maritime_robust.engine import load_run  # noqa: E402
from efmnet_eval import build_model as build_efmnet, efmnet_classes, project_to_efmnet  # noqa: E402

MEAN, STD, FILL = [.485, .456, .406], [.229, .224, .225], (124, 116, 104)
ROLE = "efmnet"
CONDITIONS = ["clean", "matched_glare_5", "matched_haze+glare_5",
              "matched_haze+cloud_5", "shifted_haze+glare_5"]
COMBO = CONDITIONS[2:]
C_GRID = [0.01, 0.03, 0.1, 0.3]
REPS, SEED = 2000, 0
T_HEADROOM, T_SHIFT = 3.0, 5.0   # pp, frozen in the pre-registration
FAMILIES = {
    "convnext_tiny": ("runs/fgsc23_research_civilian/convnext_tiny_combo_s17", "avgpool"),
    "swin_t": ("runs/fgsc23_research_civilian/swin_t_combo_s17", "avgpool"),
    "resnet50": ("runs/fgsc23_research_civilian/resnet50_combo_s17", "avgpool"),
    "efmnet_convnext_b": ("runs_external/fgsc23_civ8/fgsc23_civ8_10-29-17-34-01/checkpoint-best.pth",
                          "stages.3"),
}
CACHE = PROJECT / "reports/probe_features"


# --------------------------------------------------------------------------- images
def content(row):
    with Image.open(PROJECT / row["path"]) as handle:
        image = handle.convert("RGB")
    return ImageOps.contain(image, (224, 224), Image.Resampling.BICUBIC)


def pad_norm(image):
    padded = ImageOps.pad(image, (224, 224), Image.Resampling.BICUBIC, color=FILL)
    return TF.normalize(TF.to_tensor(padded), MEAN, STD)


def tensor_for(rows, cond):
    batch = []
    for row in rows:
        image = content(row)
        out, _ = corrupt(image, kind=cond["kind"], severity=cond["severity"],
                         seed=corruption_seed(row["sample_id"], ROLE, cond["kind"],
                                              cond["severity"], cond["domain"]),
                         domain=cond["domain"], placement=cond["placement"])
        batch.append(pad_norm(out))
    return torch.stack(batch)


# ------------------------------------------------------------------- feature capture
class Capture:
    def __init__(self, module):
        self.value = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.value = output

    def take(self):
        value, self.value = self.value, None
        return value


def pool(x):
    x = x.float()
    if x.ndim == 4:
        return x.mean((2, 3))
    if x.ndim == 3:
        return x.mean(2)
    return x


def get_module(model, dotted):
    for part in dotted.split("."):
        model = model[int(part)] if part.isdigit() else getattr(model, part)
    return model


def load_family(name, run, target):
    if name == "efmnet_convnext_b":
        rows = read_jsonl(PROJECT / "data/manifests/fgsc23_research_civilian.jsonl")
        names = efmnet_classes(rows)
        model = build_efmnet(names, PROJECT / run, "cuda")
        hook = model.stages[3]
        mapping = project_to_efmnet(rows, names)
        inverse = {v: k for k, v in mapping.items()}
    else:
        model, _, _ = load_run(PROJECT / run)
        hook = get_module(model, target)
        inverse = None
    return model.eval(), Capture(hook), inverse


def extract(models, tensor, batch=64):
    out = {name: {"features": [], "logits": []} for name in models}
    with torch.inference_mode():
        for start in range(0, len(tensor), batch):
            x = tensor[start:start + batch].cuda(non_blocking=True)
            for name, (model, capture, _) in models.items():
                with torch.autocast("cuda", dtype=torch.float16):
                    result = model(x)
                logits = result[0] if isinstance(result, (tuple, list)) else result
                out[name]["features"].append(pool(capture.take()).cpu().numpy())
                out[name]["logits"].append(logits.float().cpu().numpy())
    for name in out:
        out[name]["features"] = np.concatenate(out[name]["features"]).astype(np.float32)
        out[name]["logits"] = np.concatenate(out[name]["logits"]).astype(np.float32)
    return out


def unflatten(blob):
    nested = {}
    for key in blob:
        name, kind = key.split("__", 1)
        nested.setdefault(name, {})[kind] = blob[key]
    return nested


# ------------------------------------------------------------------------- statistics
def boot_ci(delta, groups, reps=REPS, seed=SEED):
    uniq = {g: np.where(groups == g)[0] for g in np.unique(groups)}
    keys = list(uniq)
    rng = np.random.default_rng(seed)
    stats = np.empty(reps)
    for i in range(reps):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([uniq[keys[j]] for j in pick])
        stats[i] = 100 * delta[idx].mean()
    return (100 * float(delta.mean()), float(np.percentile(stats, 2.5)),
            float(np.percentile(stats, 97.5)), stats)


def clustered_mean_ci(matrix, groups, reps=REPS, seed=SEED):
    uniq = {g: np.where(groups == g)[0] for g in np.unique(groups)}
    keys = list(uniq)
    rng = np.random.default_rng(seed)
    stats = np.empty(reps)
    for i in range(reps):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([uniq[keys[j]] for j in pick])
        stats[i] = 100 * matrix[:, idx].mean()
    return (100 * float(matrix.mean()), float(np.percentile(stats, 2.5)),
            float(np.percentile(stats, 97.5)))


def bh(pvalues, alpha=0.05):
    order = np.argsort(pvalues)
    m = len(pvalues)
    picked = [i for rank, i in enumerate(order, start=1) if pvalues[i] <= alpha * rank / m]
    cutoff = max([alpha * (r + 1) / m for r, i in enumerate(order) if i in picked], default=0.0)
    return [bool(p <= cutoff) for p in pvalues], cutoff


# ------------------------------------------------------------------------------ probe
def select_c(x, y):
    grid = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=3000, solver="lbfgs")), {"logisticregression__C": C_GRID},
        cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="accuracy", n_jobs=4)
    grid.fit(x, y)
    return float(grid.best_params_["logisticregression__C"])


def probe(x_fit, y_fit, x_test, c):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, solver="lbfgs", C=c))
    model.fit(x_fit, y_fit)
    return model.predict(x_test).astype(int)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(PROJECT / "data/manifests/fgsc23_research_civilian.jsonl")
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    y_train = np.array([r["label"] for r in train])
    y_val = np.array([r["label"] for r in val])
    groups = np.array([r["source_group_id"] for r in val])
    n_classes = int(max(y_train.max(), y_val.max())) + 1
    spec = {c["name"]: c for c in conditions() if c["name"] in CONDITIONS}
    print(json.dumps({"train": len(train), "val": len(val), "groups": len(np.unique(groups)),
                      "classes": n_classes}), flush=True)

    models = {name: load_family(name, run, target) for name, (run, target) in FAMILIES.items()}
    store = {}   # (split, condition) -> {family: {"features","logits"}}
    for split, split_rows in (("train", train), ("val", val)):
        for name in CONDITIONS:
            path = CACHE / f"{split}__{name.replace('+', '_plus_')}.npz"
            if path.exists():
                blob = np.load(path)
                store[(split, name)] = unflatten({k: blob[k] for k in blob.files})
                print(json.dumps({"cached": f"{split}/{name}"}), flush=True)
                continue
            tensor = tensor_for(split_rows, spec[name])
            nested = extract(models, tensor)
            store[(split, name)] = nested
            np.savez_compressed(path, **{f"{f}__{k}": v for f, b in nested.items()
                                         for k, v in b.items()})
            del tensor
            print(json.dumps({"extracted": f"{split}/{name}"}), flush=True)

    def features(split, cond, family):
        return store[(split, cond)][family]["features"]

    def logits(split, cond, family):
        return store[(split, cond)][family]["logits"]

    prior_label = int(np.bincount(y_train, minlength=n_classes).argmax())
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(y_train)

    out = {"preregistration": "reports/PROBE_PREREGISTRATION.md", "role": ROLE,
           "n_train": len(train), "n_val": len(val), "n_groups": int(len(np.unique(groups))),
           "conditions": CONDITIONS, "combo": COMBO, "thresholds": {
               "headroom_pp": T_HEADROOM, "shift_pp": T_SHIFT},
           "families": {}, "tests": []}

    for family, (_, _, inverse) in models.items():
        c = select_c(features("train", "clean", family), y_train)
        block = {"feature_dim": int(features("train", "clean", family).shape[1]),
                 "selected_C": c, "conditions": {}}
        for cond in CONDITIONS:
            x_tr = features("train", cond, family)
            x_te = features("val", cond, family)
            p1 = probe(features("train", "clean", family), y_train, x_te, c)
            p2 = p1 if cond == "clean" else probe(x_tr, y_train, x_te, c)
            sh = probe(features("train", "clean", family), shuffled, x_te, c)
            raw = logits("val", cond, family).argmax(1)
            model_pred = np.array([inverse[int(p)] for p in raw]) if inverse else raw
            entry = {
                "model_acc": float((model_pred == y_val).mean()),
                "p1_acc": float((p1 == y_val).mean()),
                "p2_acc": float((p2 == y_val).mean()),
                "prior_acc": float((prior_label == y_val).mean()),
                "shuffle_acc": float((sh == y_val).mean()),
                "model_pred": model_pred.tolist(), "p1_pred": p1.tolist(), "p2_pred": p2.tolist(),
            }
            for tag, pred in (("p1", p1), ("p2", p2)):
                mean, lo, hi, stats = boot_ci((pred == y_val).astype(float)
                                              - (model_pred == y_val).astype(float), groups)
                entry[f"{tag}_delta_pp"] = mean
                entry[f"{tag}_ci"] = [lo, hi]
                # one-sided p for H1 "probe is better than the deployed model":
                # P(bootstrap delta <= 0). P(delta >= 0) would invert the test.
                entry[f"{tag}_p"] = float((stats <= 0).mean())
            block["conditions"][cond] = entry
            if cond != "clean":
                out["tests"].append({"family": family, "condition": cond,
                                     "p1_p": entry["p1_p"], "p2_p": entry["p2_p"]})
        out["families"][family] = block
        print(json.dumps({"family": family, "C": c, "dims": block["feature_dim"]}), flush=True)

    # BH over the 12 (family x combo-condition) tests pre-registered in §6,
    # plus the 16-test version that additionally contains matched_glare_5 as a
    # sensitivity analysis (not part of the frozen family).
    combo_tests = [t for t in out["tests"] if t["condition"] in COMBO]
    for tag in ("p1", "p2"):
        rejected, cutoff = bh([t[f"{tag}_p"] for t in combo_tests])
        for test, reject in zip(combo_tests, rejected):
            test[f"{tag}_bh"] = bool(reject)
            out["families"][test["family"]]["conditions"][test["condition"]][f"{tag}_bh"] = bool(reject)
        for test in out["tests"]:
            test.setdefault(f"{tag}_bh", False)
        wide_rejected, wide_cutoff = bh([t[f"{tag}_p"] for t in out["tests"]])
        out.setdefault("fdr", {})[tag] = {
            "n_tests": len(combo_tests), "cutoff": cutoff, "n_significant": int(sum(rejected)),
            "sensitivity_with_glare": {"n_tests": len(out["tests"]), "cutoff": wide_cutoff,
                                       "n_significant": int(sum(wide_rejected))}}

    # ---- aggregate over the 12 combo cells --------------------------------
    cells = [(family, cond) for family in FAMILIES for cond in COMBO]

    def correct(key):
        return np.stack([(np.array(out["families"][f]["conditions"][c][key]) == y_val)
                         for f, c in cells]).astype(float)

    d_h = correct("p2_pred") - correct("model_pred")
    d_s = correct("p2_pred") - correct("p1_pred")
    d_t = correct("p1_pred") - correct("model_pred")
    summary = {"cells": [f"{f}/{c}" for f, c in cells]}
    for tag, matrix in (("H", d_h), ("S", d_s), ("T", d_t)):
        mean, lo, hi = clustered_mean_ci(matrix, groups)
        summary[tag] = {"mean_pp": mean, "ci": [lo, hi]}
    summary["n_bh_p2"] = int(sum(t.get("p2_bh", False) for t in combo_tests))
    summary["n_bh_p1"] = int(sum(t.get("p1_bh", False) for t in combo_tests))
    summary["shuffle_vs_prior_max_gap_pp"] = 100 * max(
        abs(out["families"][f]["conditions"][c]["shuffle_acc"]
            - out["families"][f]["conditions"][c]["prior_acc"]) for f, c in cells)
    out["summary"] = summary
    out["decision"] = decide(summary)
    write_json(PROJECT / "reports/frozen_probe.json", out)
    print(json.dumps({"summary": summary, "decision": out["decision"]}, indent=1), flush=True)


def decide(s):
    """Mechanical ladder from PROBE_PREREGISTRATION.md §7."""
    H, S, T = s["H"]["mean_pp"], s["S"]["mean_pp"], s["T"]["mean_pp"]
    headroom_ok = H >= T_HEADROOM and s["H"]["ci"][0] > 0 and s["n_bh_p2"] >= 7
    shift_ok = S >= T_SHIFT and s["S"]["ci"][0] > 0
    if H < T_HEADROOM:
        branch, hypothesis = "R1", "表征层损失：线性净空不足，终止新模块开发"
    elif not headroom_ok:
        branch, hypothesis = "R0", "不确定：净空的分组 CI 或 BH 计数未过护栏，终止新模块开发"
    elif not shift_ok:
        branch, hypothesis = "R3", "决策层假设：干净拟合线性边界几乎追平域匹配边界，模型自身分类头是短板"
    else:
        branch, hypothesis = "R2", "特征分布偏移假设：需域匹配边界才能取出信息，轻量特征对齐"
    return {"branch": branch, "hypothesis": hypothesis, "H_pp": H, "S_pp": S, "T_pp": T,
            "headroom_guard": bool(headroom_ok), "shift_guard": bool(shift_ok),
            "terminate": branch in ("R0", "R1")}


if __name__ == "__main__":
    main()
