#!/usr/bin/env python
"""Zero-training restoration probe (route A, classical methods only).

For three seed-17 combined-augmentation baselines and six evaluation
conditions, compare:

  identity      no restoration
  clahe         tile-wise contrast equalisation on luminance (8x8 tiles)
  satfill       saturated-pixel detection + nearest-valid-neighbour fill
  fusion_clahe  mean softmax of identity and clahe
  fusion_satfill

All restoration is applied to the contained image before letterbox padding,
so the geometry seen by the classifier is unchanged. The clean condition must
reproduce the saved clean predictions exactly (pipeline-parity check).

Primary criterion is classification accuracy / macro-F1 on the same samples;
PSNR/SSIM are deliberately not computed here because the score is task gain.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_common import (PROJECT, check_clean_parity, condition, content, degrade,  # noqa: E402
                          load_models, margin, pad_norm, probs_for, saved_clean_predictions,
                          score_of, test_rows)
from maritime_robust.common import Budget, write_json, write_jsonl  # noqa: E402

CONDITIONS = [("clean", "clean"),
              ("matched_haze_5", "haze 5"),
              ("matched_glare_5", "glare 5"),
              ("matched_haze+glare_5", "haze+glare 5"),
              ("matched_haze+cloud_5", "haze+cloud 5"),
              ("shifted_haze+glare_5", "shifted haze+glare 5")]
METHODS = ["identity", "clahe", "satfill", "fusion_clahe", "fusion_satfill"]


def clahe(image, tiles=8, clip=2.0):
    """Contrast-limited adaptive histogram equalisation on the Y channel."""
    ycbcr = np.array(image.convert("YCbCr"), dtype=np.uint8)
    y = ycbcr[..., 0].astype(np.float32)
    height, width = y.shape
    tile_h, tile_w = math.ceil(height / tiles), math.ceil(width / tiles)
    padded = np.pad(y, ((0, tiles * tile_h - height), (0, tiles * tile_w - width)),
                    mode="reflect")
    maps = np.zeros((tiles, tiles, 256), dtype=np.float32)
    limit = clip * (tile_h * tile_w) / 256.0
    for i in range(tiles):
        for j in range(tiles):
            tile = padded[i * tile_h:(i + 1) * tile_h, j * tile_w:(j + 1) * tile_w]
            hist = np.bincount(tile.astype(np.uint8).ravel(), minlength=256).astype(np.float32)
            excess = np.maximum(hist - limit, 0).sum()
            hist = np.minimum(hist, limit) + excess / 256.0
            cdf = hist.cumsum()
            cdf /= cdf[-1]
            maps[i, j] = cdf * 255.0
    yy = (np.arange(height) + 0.5) / tile_h - 0.5
    xx = (np.arange(width) + 0.5) / tile_w - 0.5
    y0 = np.clip(np.floor(yy).astype(int), 0, tiles - 1)
    x0 = np.clip(np.floor(xx).astype(int), 0, tiles - 1)
    y1 = np.clip(y0 + 1, 0, tiles - 1)
    x1 = np.clip(x0 + 1, 0, tiles - 1)
    wy = np.clip(yy - y0, 0, 1)[:, None]
    wx = np.clip(xx - x0, 0, 1)[None, :]
    idx = y.astype(np.uint8)
    m00 = maps[y0[:, None], x0[None, :], idx]
    m01 = maps[y0[:, None], x1[None, :], idx]
    m10 = maps[y1[:, None], x0[None, :], idx]
    m11 = maps[y1[:, None], x1[None, :], idx]
    out = (m00 * (1 - wy) * (1 - wx) + m01 * (1 - wy) * wx +
           m10 * wy * (1 - wx) + m11 * wy * wx)
    ycbcr[..., 0] = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(ycbcr, "YCbCr").convert("RGB")


def saturated_fraction(image, threshold=250):
    a = np.asarray(image)
    return float((a.max(2) >= threshold).mean())


def satfill(image, threshold=250, smooth_iterations=4):
    """Fill clipped highlight pixels from the nearest unsaturated neighbour."""
    a = np.asarray(image, dtype=np.uint8)
    mask = a.max(2) >= threshold
    if not mask.any():
        return image, 0.0
    indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
    filled = a[tuple(indices)]
    out = a.copy()
    out[mask] = filled[mask]
    if smooth_iterations:
        f = out.astype(np.float32)
        for _ in range(smooth_iterations):
            blurred = np.stack([uniform_filter(f[..., c], 3, mode="nearest")
                                for c in range(3)], -1)
            f[mask] = blurred[mask]
        out = np.clip(f, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB"), float(mask.mean())


def restore(image, method):
    if method == "identity":
        return image, 0.0
    if method == "clahe":
        return clahe(image), saturated_fraction(image)
    if method == "satfill":
        return satfill(image)
    raise KeyError(method)


def main():
    rows = test_rows()
    labels = [r["label"] for r in rows]
    n = max(labels) + 1
    models = load_models()
    result = {"n": len(rows), "conditions": [c for c, _ in CONDITIONS],
              "methods": METHODS, "models": {}}
    records = []
    with Budget("evaluation"):
        for name, (model, _) in models.items():
            entry = {"conditions": {}}
            for cond_name, display in CONDITIONS:
                cond = condition(cond_name)
                base_images = []
                for row in rows:
                    base_images.append(degrade(content(row), row, cond)[0])
                sat = float(np.mean([saturated_fraction(im) for im in base_images]))
                per_method = {}
                for method in ("identity", "clahe", "satfill"):
                    tensors = [pad_norm(restore(im, method)[0]) for im in base_images]
                    probs = probs_for(model, tensors)
                    score, preds = score_of(probs, labels, n)
                    if cond_name == "clean" and method == "identity":
                        check_clean_parity(name, preds)
                    per_method[method] = {"score": score, "preds": preds, "probs": probs}
                    for row, label, pred, p in zip(rows, labels, preds, probs):
                        records.append({"model": name, "condition": cond_name, "method": method,
                                        "sample_id": row["sample_id"], "label": label,
                                        "prediction": pred, "probs": [round(float(v), 6) for v in p]})
                for method in ("clahe", "satfill"):
                    fusion = (per_method["identity"]["probs"] + per_method[method]["probs"]) / 2
                    score, preds = score_of(fusion, labels, n)
                    per_method[f"fusion_{method}"] = {"score": score, "preds": preds, "probs": fusion}
                    for row, label, pred, p in zip(rows, labels, preds, fusion):
                        records.append({"model": name, "condition": cond_name,
                                        "method": f"fusion_{method}",
                                        "sample_id": row["sample_id"], "label": label,
                                        "prediction": pred,
                                        "probs": [round(float(v), 6) for v in p]})
                ref = per_method["identity"]["preds"]
                cond_entry = {"display": display, "saturation_fraction": sat, "methods": {}}
                for method in METHODS:
                    probs = per_method[method]["probs"]
                    m = per_method[method]
                    cond_entry["methods"][method] = {
                        "accuracy": m["score"]["accuracy"],
                        "macro_f1": m["score"]["macro_f1"],
                        "per_class_recall": m["score"]["per_class_recall"],
                        "delta_accuracy_pp": 100 * (m["score"]["accuracy"] -
                                                    per_method["identity"]["score"]["accuracy"]),
                        "changed": int(sum(a != b for a, b in zip(m["preds"], ref))),
                        "rescued": int(sum(a != b and a == l for a, b, l in
                                           zip(m["preds"], ref, labels))),
                        "hurt": int(sum(a != b and b == l for a, b, l in
                                        zip(m["preds"], ref, labels))),
                        "mean_margin": float(np.mean(margin(probs))),
                    }
                entry["conditions"][cond_name] = cond_entry
                print(json.dumps({"model": name, "condition": cond_name,
                                  "identity": per_method["identity"]["score"]["accuracy"],
                                  "clahe": per_method["clahe"]["score"]["accuracy"],
                                  "satfill": per_method["satfill"]["score"]["accuracy"],
                                  "fusion_clahe": per_method["fusion_clahe"]["score"]["accuracy"],
                                  "sat": round(sat, 4)}), flush=True)
            result["models"][name] = entry
    write_json(PROJECT / "reports/restoration_probe.json", result)
    write_jsonl(PROJECT / "reports/restoration_probe_records.jsonl", records)


if __name__ == "__main__":
    main()
