"""Capacity without fitting capacity: a frozen-feature linear probe over the grid.

The fine-tuned capacity study is confounded — 412 training images cannot feed an
89M-parameter backbone, and indeed clean accuracy is flat from Tiny to Base. A
linear probe isolates the other half of the question: with only a linear head,
does a richer frozen representation produce a smaller compound excess?

Same decomposition and same JSON schema as ``overlap_main_grid.py``, so
``scale_aggregate.py`` reads both without changes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

import explore_eval as ee  # noqa: E402
import m1_eval as me  # noqa: E402
import m2_eval as m2  # noqa: E402
from maritime_robust.common import ROOT, read_jsonl, write_json  # noqa: E402
from maritime_robust.engine import ShipDataset, loader  # noqa: E402
from maritime_robust.methods import build as build_method  # noqa: E402

PAIRS = (("haze", "glare"), ("haze", "cloud"))
SEVERITIES = (2, 3, 4, 5)


def features(model, tensor, batch=64):
    out = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(tensor), batch):
            out.append(model.features_of(tensor[i:i + batch].cuda()).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--model", default="convnext_small")
    ap.add_argument("--views", type=int, default=4,
                    help="augmented views per training image fed to the probe")
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = PROJECT / (args.out_dir or f"reports/probe_{args.model}")

    manifest = PROJECT / f"data/manifests/folds/fgsc23_civ8_fold{args.fold}.jsonl"
    rows = read_jsonl(manifest)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]

    torch.hub.set_dir(str(ROOT / "artifacts/weights"))
    model = build_method(args.model, 8, "none").cuda().eval()

    # --- probe training data: frozen features of the frozen augmentation recipe
    ds = ShipDataset(train_rows, train=True, augmentation="combined", seed=args.seed)
    X, y = [], []
    for view in range(args.views):
        ds.epoch = view
        for batch in loader(ds, 32, 4, shuffle=False):
            X.append(features(model, batch[0]))
            y.append(batch[1].numpy())
    X = np.concatenate(X)
    y = np.concatenate(y)
    print(f"probe train {X.shape}", flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=args.c, max_iter=3000, multi_class="multinomial")
    clf.fit(scaler.transform(X), y)

    def predict(tensor):
        return clf.predict(scaler.transform(features(model, tensor)))

    labels = np.array([r["label"] for r in val_rows])
    contents = [ee.content(r) for r in val_rows]
    conds = {c["name"]: c for c in m2.conditions_47()}

    wanted = ["clean"]
    for family in ("haze", "glare", "cloud"):
        wanted += [f"matched_{family}_{s}" for s in range(1, 6)]
    for a, b in PAIRS:
        wanted += [f"matched_{a}+{b}_{s}" for s in SEVERITIES]

    errors, acc = {}, {}
    for name in wanted:
        tensor = me.batch_of(val_rows, contents, conds[name], with_mask=False)
        pred = predict(tensor)
        errors[name] = pred != labels
        acc[name] = float(np.mean(pred == labels))
        print(f"{name:26s} acc {100 * acc[name]:6.2f}", flush=True)

    clean = acc["clean"]
    records = []
    for a, b in PAIRS:
        for s in SEVERITIES:
            ka, kb, kab = (f"matched_{a}_{s}", f"matched_{b}_{s}", f"matched_{a}+{b}_{s}")
            union = 100 * float(np.mean(errors[ka] | errors[kb]))
            overlap = 100 * float(np.mean(errors[ka] & errors[kb]))
            records.append({
                "pair": f"{a}+{b}", "severity": s,
                "drop_a_pp": 100 * (clean - acc[ka]), "drop_b_pp": 100 * (clean - acc[kb]),
                "drop_pair_pp": 100 * (clean - acc[kab]),
                "error_pair_pp": 100 * (1 - acc[kab]), "error_union_pp": union,
                "overlap_pp": overlap,
                "E_pp": 100 * (clean - acc[kab]) - 100 * (clean - acc[ka]) - 100 * (clean - acc[kb]),
                "delta_pp": 100 * (1 - acc[kab]) - union,
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"fold{args.fold}_s{args.seed}.json", {
        "run": f"linear_probe/{args.model}", "fold": args.fold, "seed": args.seed,
        "model": args.model, "views": args.views, "C": args.c,
        "clean_accuracy": clean, "accuracy": acc, "pairs": records})
    print("\npair        sev  drop_a drop_b drop_ab  delta")
    for r in records:
        print(f"{r['pair']:10s} {r['severity']:3d} {r['drop_a_pp']:6.1f} {r['drop_b_pp']:6.1f} "
              f"{r['drop_pair_pp']:7.1f} {r['delta_pp']:+7.1f}")


if __name__ == "__main__":
    main()
