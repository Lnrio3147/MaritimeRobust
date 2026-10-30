"""Is the severity-5 cliff a sensor-saturation regime?

For every condition, measure how much of the corrupted image is driven to
sensor saturation (any channel >= 0.99) and how much is fully blown out
(all channels >= 0.95). Then line these up against the measured
superadditivity E(s) from reports/severity_curve_results.json.

No training, no model: this is a property of the corruption pipeline.

    python scripts/saturation_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from maritime_robust.common import read_jsonl  # noqa: E402
from maritime_robust.degradations import corrupt, corruption_seed  # noqa: E402

ROLE = "efmnet"
RES = (224, 224)
CONDITIONS = ([("haze", s) for s in range(1, 6)]
              + [("cloud", s) for s in range(1, 6)]
              + [("glare", s) for s in range(1, 6)]
              + [(k, s) for k in ("haze+glare", "haze+cloud") for s in (2, 3, 4, 5)])


def main() -> None:
    rows = [r for r in read_jsonl(ROOT / "data/manifests/folds/fgsc23_civ8_fold0.jsonl")
            if r["split"] == "val"]
    images = []
    for r in rows:
        with Image.open(ROOT / r["path"]) as fh:
            images.append(ImageOps.contain(fh.convert("RGB"), RES, Image.Resampling.BICUBIC))

    out = {}
    for kind, sev in CONDITIONS:
        sat, blown, mean_l, contrast = [], [], [], []
        for row, img in zip(rows, images):
            degraded, _ = corrupt(
                img, kind=kind, severity=sev,
                seed=corruption_seed(row["sample_id"], ROLE, kind, sev, "matched"),
                domain="matched", placement="random")
            a = np.asarray(degraded.resize(RES, Image.Resampling.BICUBIC), dtype=np.float32) / 255
            sat.append(float((a.max(2) >= 0.99).mean()))
            blown.append(float((a.min(2) >= 0.95).mean()))
            lum = (a @ np.array([.299, .587, .114], dtype=np.float32))
            mean_l.append(float(lum.mean()))
            contrast.append(float(lum.std()))
        out[f"matched_{kind}_{sev}"] = {
            "sat_fraction": float(np.mean(sat)), "blown_fraction": float(np.mean(blown)),
            "mean_luminance": float(np.mean(mean_l)), "contrast": float(np.mean(contrast)),
        }

    (ROOT / "reports/saturation_probe.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out.items():
        print(f"{k:26s} sat {v['sat_fraction']:.4f}  blown {v['blown_fraction']:.4f}  "
              f"lum {v['mean_luminance']:.3f}  contrast {v['contrast']:.3f}")


if __name__ == "__main__":
    main()
