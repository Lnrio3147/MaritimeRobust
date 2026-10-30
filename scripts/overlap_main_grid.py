#!/usr/bin/env python
"""Error-overlap decomposition for the protocol's own haze/glare/cloud grid.

The composition study (``composition_interaction.py``) showed that a naive
``E = drop(a+b) - drop(a) - drop(b)`` mixes true interaction with the overlap
of the two single-corruption error sets. This script applies the same
decomposition to the paper's registered 47-condition grid, where the compounds
are haze+glare and haze+cloud at matched severities:

    E_pp     = drop(a+b) - drop(a) - drop(b)                (accuracy space)
    delta_pp = e(a+b) - |E_a u E_b|                         (error space)

``delta`` is the superadditivity that survives once the two corruptions are
allowed to break the same images independently. No training; frozen baseline.

    python scripts/overlap_main_grid.py --fold 0 --seed 17
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT, ROOT / "scripts"):
    sys.path.insert(0, str(path))

import explore_eval as ee  # noqa: E402
import m1_eval as me  # noqa: E402
import m2_eval as m2  # noqa: E402
from maritime_robust.common import ROOT as PROJECT, read_jsonl, write_json  # noqa: E402

PAIRS = (("haze", "glare"), ("haze", "cloud"))
SEVERITIES = (2, 3, 4, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--model", default="convnext_tiny")
    ap.add_argument("--out-dir", default="reports/overlap")
    args = ap.parse_args()

    run = PROJECT / (f"runs/fgsc23_civ8_fold{args.fold}/"
                     f"{args.model}_none_e40_s{args.seed}")
    model, settings = ee.load_any(run)
    rows = [r for r in read_jsonl(Path(settings["manifest"])) if r["split"] == "val"]
    labels = np.array([r["label"] for r in rows])
    contents = [ee.content(r) for r in rows]

    conds = {c["name"]: c for c in m2.conditions_47()}
    # Only the conditions this decomposition needs, still scored through the
    # registered builder so the images are identical to every other evaluation.
    wanted = ["clean"]
    for family in ("haze", "glare", "cloud"):
        wanted += [f"matched_{family}_{s}" for s in range(1, 6)]
    for a, b in PAIRS:
        wanted += [f"matched_{a}+{b}_{s}" for s in SEVERITIES]
    missing = [w for w in wanted if w not in conds]
    if missing:
        print("missing conditions:", missing)
        return

    errors, acc = {}, {}
    for name in wanted:
        tensor = me.batch_of(rows, contents, conds[name], with_mask=False)
        pred = ee.predict(model, tensor) if hasattr(ee, "predict") else None
        if pred is None:
            import torch
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                logits = model(tensor.cuda()).float()
            pred = logits.argmax(1).cpu().numpy()
        errors[name] = pred != labels
        acc[name] = float(np.mean(pred == labels))
        print(f"{name:26s} acc {100*acc[name]:6.2f}", flush=True)

    clean = acc["clean"]
    records = []
    for a, b in PAIRS:
        for s in SEVERITIES:
            ka, kb, kab = (f"matched_{a}_{s}", f"matched_{b}_{s}",
                           f"matched_{a}+{b}_{s}")
            drop_a = 100 * (clean - acc[ka])
            drop_b = 100 * (clean - acc[kb])
            drop_ab = 100 * (clean - acc[kab])
            e_ab = 100 * (1 - acc[kab])
            overlap = 100 * float(np.mean(errors[ka] & errors[kb]))
            union = 100 * float(np.mean(errors[ka] | errors[kb]))
            records.append({
                "pair": f"{a}+{b}", "severity": s,
                "drop_a_pp": drop_a, "drop_b_pp": drop_b, "drop_pair_pp": drop_ab,
                "error_pair_pp": e_ab, "error_union_pp": union,
                "overlap_pp": overlap,
                "E_pp": drop_ab - drop_a - drop_b,
                "delta_pp": e_ab - union})

    print()
    print(f"{'pair':12s} {'sev':>3s} {'drop_a':>7s} {'drop_b':>7s} {'drop_ab':>8s} "
          f"{'E':>7s} {'union':>7s} {'delta':>7s}")
    for r in records:
        print(f"{r['pair']:12s} {r['severity']:3d} {r['drop_a_pp']:7.1f} "
              f"{r['drop_b_pp']:7.1f} {r['drop_pair_pp']:8.1f} {r['E_pp']:+7.1f} "
              f"{r['error_union_pp']:7.1f} {r['delta_pp']:+7.1f}")

    write_json(PROJECT / args.out_dir / f"fold{args.fold}_s{args.seed}.json",
               {"run": str(run.relative_to(PROJECT)), "fold": args.fold,
                "seed": args.seed, "clean_accuracy": clean,
                "accuracy": acc, "pairs": records})


if __name__ == "__main__":
    main()
