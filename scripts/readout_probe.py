#!/usr/bin/env python
"""D3 probe: does the severity-5 jump survive a different decision rule?

The D2 probe showed representation drift is superadditive at *every* severity
while accuracy excess appears only at s=5. If the jump is produced by the
learned classifier's boundary, a simpler readout on the same frozen features
should show a different curve; if it survives any readout, the jump is in the
representation's geometry.

Two readouts, both trained on clean training features only (they never see a
degradation):
    logreg    multinomial logistic regression on L2-normalised descriptors
    prototype nearest class centroid, cosine

    python scripts/readout_probe.py --run runs/fgsc23_civ8_fold0/convnext_tiny_none_e40_s17

Output: reports/readout_probe_s{seed}.json (curves plus the model's own curve
copied from reports/feature_drift_s{seed}.json for comparison).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import explore_eval as ee  # noqa: E402
import feature_drift_probe as fdp  # noqa: E402
import m1_eval as me  # noqa: E402
from maritime_robust.common import read_jsonl, write_json  # noqa: E402
from maritime_robust.engine import metrics  # noqa: E402

PAIRS = fdp.PAIRS


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def evaluate(pred_fn, feats, labels, conds, n_classes):
    return {c["name"]: metrics(labels.tolist(), pred_fn(feats[c["name"]]).tolist(),
                                n_classes)["accuracy"] for c in conds}


def curve(acc):
    def e(pair, s):
        a, b = PAIRS[pair]
        c0 = acc["clean"]
        return 100 * ((c0 - acc[f"matched_{pair}_{s}"])
                      - (c0 - acc[f"matched_{a}_{s}"]) - (c0 - acc[f"matched_{b}_{s}"]))
    return {pair: {str(s): e(pair, s) for s in (2, 3, 4, 5)} for pair in PAIRS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = ROOT / args.run
    model, settings = ee.load_any(run)
    seed = settings["seed"]
    n = settings["num_classes"]
    manifest = Path(settings["manifest"])
    rows = read_jsonl(manifest)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    y_train = np.array([r["label"] for r in train])
    y_val = np.array([r["label"] for r in val])

    conds = ee.all_conditions() + fdp.EXTRA
    clean_cond = next(c for c in conds if c["kind"] == "clean")

    train_feat = fdp.features(model, train, [ee.content(r) for r in train], [clean_cond])["clean"]
    val_feat = fdp.features(model, val, [ee.content(r) for r in val], conds)

    X, Xv = l2(train_feat), {k: l2(v) for k, v in val_feat.items()}

    logreg = LogisticRegression(C=1.0, max_iter=5000, n_jobs=-1).fit(X, y_train)
    centroids = np.stack([X[y_train == c].mean(0) for c in range(n)])
    centroids = l2(centroids)

    readouts = {
        "logreg": lambda F: logreg.predict(F),
        "prototype": lambda F: (F @ centroids.T).argmax(1),
    }

    out = {"run": args.run, "seed": seed, "n_train": len(train), "n_val": len(val),
           "clean_train_accuracy": {"logreg": float((logreg.predict(X) == y_train).mean()),
                                    "prototype": float(((X @ centroids.T).argmax(1) == y_train).mean())},
           "readouts": {}}

    for name, fn in readouts.items():
        acc = evaluate(fn, Xv, y_val, conds, n)
        out["readouts"][name] = {"clean_accuracy": acc["clean"], "curve": curve(acc),
                                 "combo_accuracy": acc["matched_haze+glare_5"],
                                 "conditions": {k: round(v, 4) for k, v in acc.items()}}

    drift_path = ROOT / f"reports/feature_drift_s{seed}.json"
    if drift_path.exists():
        model_curve = json.loads(drift_path.read_text())["curve"]
        out["model_curve"] = {p: {s: v["accuracy_pp"] for s, v in pts.items()}
                              for p, pts in model_curve.items()}

    write_json(ROOT / (args.out or f"reports/readout_probe_s{seed}.json"), out)

    print(f"seed {seed}  clean acc logreg {100*out['readouts']['logreg']['clean_accuracy']:.2f} "
          f"prototype {100*out['readouts']['prototype']['clean_accuracy']:.2f} "
          f"(model {100*json.loads(drift_path.read_text())['clean_accuracy']:.2f})")
    for pair in PAIRS:
        for s in ("2", "3", "4", "5"):
            row = [f"s{s}"]
            for name in readouts:
                row.append(f"{name} {out['readouts'][name]['curve'][pair][s]:+6.1f}")
            if "model_curve" in out:
                row.append(f"model {out['model_curve'][pair][s]:+6.1f}")
            print(f"  {pair:11s} " + "  ".join(row))


if __name__ == "__main__":
    main()
