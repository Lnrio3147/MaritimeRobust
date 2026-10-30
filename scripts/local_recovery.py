#!/usr/bin/env python
"""Local counterfactual recovery (route B), zero training.

For each test image, the degraded image and the clean image share identical
geometry. We replace one cell of a 3x3 grid (over the contained content, before
letterbox padding) in the degraded image with the *true clean content* of the
same cell, and re-predict. This bounds what any restoration method could
recover if it perfectly restored that region, and localises where recoverable
evidence sits.

Definitions (per model and condition):
  recoverable set  samples where degraded prediction is wrong AND clean is right.
  rescue@cell      #{recoverable and cell-replaced prediction correct} / |recoverable|
  error induction  #{degraded right AND clean right AND cell-replaced wrong} /
                   #{degraded right AND clean right}
  margin           top-1 probability minus top-2 probability.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_common import (PROJECT, condition, content, degrade, load_models, margin,  # noqa: E402
                          pad_norm, probs_for, test_rows)
from maritime_robust.common import Budget, write_json, write_jsonl  # noqa: E402

CONDITIONS = ["matched_haze_5", "matched_glare_5", "matched_haze+glare_5",
              "matched_haze+cloud_5"]
GRID = 3


def cell_box(height, width, row, col):
    ye = [round(height * k / GRID) for k in range(GRID + 1)]
    xe = [round(width * k / GRID) for k in range(GRID + 1)]
    return ye[row], ye[row + 1], xe[col], xe[col + 1]


def compose(degraded, clean, box):
    y0, y1, x0, x1 = box
    d = np.asarray(degraded).copy()
    c = np.asarray(clean)
    d[y0:y1, x0:x1] = c[y0:y1, x0:x1]
    return Image.fromarray(d)


def main():
    rows = test_rows()
    labels = [r["label"] for r in rows]
    n = max(labels) + 1
    models = load_models()
    clean_images = [content(r) for r in rows]
    result = {"n": len(rows), "grid": GRID, "conditions": CONDITIONS, "models": {}}
    records = []
    with Budget("evaluation"):
        for name, (model, _) in models.items():
            entry = {"conditions": {}}
            clean_probs = probs_for(model, [pad_norm(im) for im in clean_images])
            clean_pred = clean_probs.argmax(1).tolist()
            clean_margin = margin(clean_probs)
            for i, pred, mg in zip(range(len(rows)), clean_pred, clean_margin):
                records.append({"model": name, "condition": "clean", "variant": "clean",
                                "sample_id": rows[i]["sample_id"], "label": labels[i],
                                "prediction": pred, "margin": round(mg, 6)})
            for cond_name in CONDITIONS:
                cond = condition(cond_name)
                degraded_images = [degrade(im, r, cond)[0]
                                   for im, r in zip(clean_images, rows)]
                deg_probs = probs_for(model, [pad_norm(im) for im in degraded_images])
                deg_pred = deg_probs.argmax(1).tolist()
                deg_margin = margin(deg_probs)
                recoverable = [i for i in range(len(rows))
                               if deg_pred[i] != labels[i] and clean_pred[i] == labels[i]]
                stable = [i for i in range(len(rows))
                          if deg_pred[i] == labels[i] and clean_pred[i] == labels[i]]
                degraded_wrong = [i for i in range(len(rows)) if deg_pred[i] != labels[i]]
                for i, pred, mg in zip(range(len(rows)), deg_pred, deg_margin):
                    records.append({"model": name, "condition": cond_name, "variant": "degraded",
                                    "sample_id": rows[i]["sample_id"], "label": labels[i],
                                    "prediction": pred, "margin": round(mg, 6)})
                cell_entries = []
                for row in range(GRID):
                    for col in range(GRID):
                        tensors = []
                        for d, c in zip(degraded_images, clean_images):
                            h, w = c.size[1], c.size[0]
                            tensors.append(pad_norm(compose(d, c, cell_box(h, w, row, col))))
                        probs = probs_for(model, tensors)
                        preds = probs.argmax(1).tolist()
                        mg = margin(probs)
                        rescues = [i for i in recoverable if preds[i] == labels[i]]
                        hurts = [i for i in stable if preds[i] != labels[i]]
                        changed = [i for i in range(len(rows)) if preds[i] != deg_pred[i]]
                        cell_entries.append({
                            "row": row, "col": col,
                            "accuracy": float(np.mean([p == l for p, l in zip(preds, labels)])),
                            "rescue_count": len(rescues),
                            "rescue_rate_recoverable": len(rescues) / max(1, len(recoverable)),
                            "rescue_share_degraded_wrong": len(rescues) /
                                                            max(1, len(degraded_wrong)),
                            "error_induction_count": len(hurts),
                            "error_induction_rate_stable": len(hurts) / max(1, len(stable)),
                            "changed": len(changed),
                            "margin_delta_recoverable": float(np.mean(
                                [mg[i] - deg_margin[i] for i in recoverable]))
                                if recoverable else 0.0,
                        })
                        for i, pred, m in zip(range(len(rows)), preds, mg):
                            records.append({"model": name, "condition": cond_name,
                                            "variant": f"cell_{row}_{col}",
                                            "sample_id": rows[i]["sample_id"],
                                            "label": labels[i], "prediction": pred,
                                            "margin": round(m, 6)})
                entry["conditions"][cond_name] = {
                    "clean_accuracy": float(np.mean([p == l for p, l in zip(clean_pred, labels)])),
                    "degraded_accuracy": float(np.mean([p == l for p, l in zip(deg_pred, labels)])),
                    "recoverable": len(recoverable),
                    "degraded_wrong": len(degraded_wrong),
                    "stable": len(stable),
                    "cells": cell_entries,
                }
                grid = [[round(cell_entries[r * GRID + c]["rescue_rate_recoverable"], 3)
                         for c in range(GRID)] for r in range(GRID)]
                print(json.dumps({"model": name, "condition": cond_name,
                                  "clean": entry["conditions"][cond_name]["clean_accuracy"],
                                  "degraded": entry["conditions"][cond_name]["degraded_accuracy"],
                                  "recoverable": len(recoverable), "rescue_grid": grid}),
                      flush=True)
            result["models"][name] = entry
    write_json(PROJECT / "reports/local_recovery.json", result)
    write_jsonl(PROJECT / "reports/local_recovery_records.jsonl", records)


if __name__ == "__main__":
    main()
