"""Is the severity-5 superadditivity produced by 8-bit highlight clipping?

Everything is composed exactly as degradations.corrupt() does, in sRGB space,
with one change: after composition the highlight is either

  clip     hard clip at 1.0 (this reproduces corrupt() bit for bit - control)
  shoulder a soft roll-off above 0.7 instead of clipping

Single conditions are re-rendered with the same renderer, so the marginal drops
used for the additive prediction are measured in the same world as the compound.
If the excess-over-additive disappears under `shoulder`, clipping is the
mechanism; if it persists, clipping is not the explanation.

    python scripts/hdr_render_probe.py [--fold 0] [--seed 17]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import explore_eval as ee  # noqa: E402
from maritime_robust.common import read_jsonl  # noqa: E402
from maritime_robust.degradations import (  # noqa: E402
    RECIPE, smooth_field, corruption_seed)
from maritime_robust.engine import metrics  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

RES = (224, 224)
KNEE = 0.7


def compose(image: Image.Image, kind: str, severity: int, seed: int,
            mode: str) -> tuple[np.ndarray, float]:
    """Same operations as corrupt(), with a selectable highlight transfer."""
    rng = np.random.default_rng(seed)
    a = np.array(image.convert("RGB"), dtype=np.float32) / 255
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = (xx + 0.5) / w
    yy = (yy + 0.5) / h
    i = severity - 1
    for part in kind.split("+"):
        field = smooth_field(rng, (w, h), 6)
        if part == "haze":
            t0 = RECIPE["haze_transmission"][i]
            t = np.clip(t0 + .18 * (field - .5), .05, 1)
            airlight = rng.uniform(.85, .98, 3).astype(np.float32)
            a = a * t[..., None] + airlight * (1 - t[..., None])
        else:
            cx, cy = rng.uniform(.1, .9, 2)
            if part == "cloud":
                spatial = field + .6 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / .06)
                threshold = np.quantile(spatial, 1 - RECIPE["cloud_area"][i])
                support = spatial >= threshold
                soft = gaussian_filter(support.astype(np.float32), sigma=max(1, min(h, w) * .015))
                alpha = soft * RECIPE["cloud_alpha"][i] * (.65 + .35 * field)
                a = a * (1 - alpha[..., None]) + .98 * alpha[..., None]
            else:
                sigma = RECIPE["glare_sigma"][i]
                bloom = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
                gain = RECIPE["glare_strength"][i] * bloom
                a = a + gain[..., None] * np.array([1, .98, .90], dtype=np.float32)
    if mode == "shoulder":
        over = a > KNEE
        a = np.where(over, KNEE + (1 - KNEE) * np.tanh((a - KNEE) / (1 - KNEE)), a)
        a = np.clip(a, 0, 1)
    else:
        a = np.clip(a, 0, 1)
    sat = float((a.max(2) >= 0.99).mean())
    return (a * 255 + .5).astype(np.uint8), sat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="reports/hdr_render_probe.json")
    args = ap.parse_args()

    run = ROOT / f"runs/fgsc23_civ8_fold{args.fold}/convnext_tiny_none_e40_s{args.seed}"
    manifest = ROOT / f"data/manifests/folds/fgsc23_civ8_fold{args.fold}.jsonl"
    model, settings = ee.load_any(run)
    rows = [r for r in read_jsonl(manifest) if r["split"] == "val"]
    labels = [r["label"] for r in rows]
    images = [ee.content(r) for r in rows]

    def score(imgs):
        tensor = torch.stack([ee.pad_norm(x if isinstance(x, Image.Image) else Image.fromarray(x))
                              for x in imgs])
        preds = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for start in range(0, len(tensor), 64):
                preds.append(model(tensor[start:start + 64].cuda()).float().argmax(1).cpu())
        return torch.cat(preds).numpy()

    clean_acc = metrics(labels, score(images).tolist(), settings["num_classes"])["accuracy"]

    result = {"clean_accuracy": clean_acc, "modes": {}}
    for mode in ("clip", "shoulder"):
        drops, sats = {}, {}
        conds = ([f"{k}_{s}" for k in ("haze", "cloud", "glare") for s in range(1, 6)]
                 + [f"{k}_{s}" for k in ("haze+glare", "haze+cloud") for s in (2, 3, 4, 5)])
        for cond in conds:
            kind, sev = cond.rsplit("_", 1)
            sev = int(sev)
            imgs, ss = [], []
            for r, im in zip(rows, images):
                arr, sat = compose(im, kind, sev,
                                   corruption_seed(r["sample_id"], "efmnet", kind, sev, "matched"),
                                   mode)
                imgs.append(arr)
                ss.append(sat)
            acc = metrics(labels, score(imgs).tolist(), settings["num_classes"])["accuracy"]
            drops[cond] = 100 * (clean_acc - acc)
            sats[cond] = float(np.mean(ss))

        out = {}
        for cond in (c for c in conds if "+" in c):
            kind, sev = cond.rsplit("_", 1)
            a, b = kind.split("+")
            marginal = drops[f"{a}_{sev}"] + drops[f"{b}_{sev}"]
            out[cond] = {"drop_pp": drops[cond], "marginal_pp": marginal,
                         "excess_pp": drops[cond] - marginal, "sat_fraction": sats[cond]}
        result["modes"][mode] = {"drops": drops, "compound": out}
        for cond, v in out.items():
            print(f"{mode:9s} {cond:20s} drop {v['drop_pp']:6.1f} excess {v['excess_pp']:+6.1f} "
                  f"sat {v['sat_fraction']:.3f}", flush=True)

    (ROOT / args.out).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
