"""Does the estimated target field match the degradation it was estimated from?

Read-only diagnostic.  Nothing here trains, tunes or alters the estimator; it
re-reads artefacts that already exist:

* ``runs_main/maritime_main_v2/fields/bank.npz`` -- the fields the NCF arm was
  trained on, in the order they were estimated;
* ``.../fields/unlabeled_target_images.jsonl`` -- provenance for each field
  (source image, and the token that encodes family and severity);
* ``.../fields/target_images/`` -- the corrupted target image each field was
  estimated from.

Two quantities per (family, severity):

* *actual*: PSNR between the clean target image and the corrupted image the
  estimator saw.  This is the damage the field is supposed to explain.
* *reconstructed*: PSNR between the same clean image passed through the stored
  field (blur, then gain/addition) and that corrupted image.

A field that captures the degradation raises the second number well above the
degradation's own strength, and the gap between the two shrinks as severity
rises.  A field that does not capture it leaves them similar, and the estimated
parameters will not move with severity.

The script also measures how strong the *evaluation* compounds are relative to
the singles the bank was built from, because a bank of singles is applied to a
compound test set.

    python scripts/maritime_field_match.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maritime_robust.cifar_field_estimator import gaussian_kernels  # noqa: E402
from maritime_robust.main_protocol import (apply_components, clean_image,  # noqa: E402
                                           partition_rows)
from maritime_robust.common import read_jsonl  # noqa: E402

RUN = ROOT / "runs_main/maritime_main_v2"
FIELDS = RUN / "fields"
CONFIG = ROOT / "configs/maritime_main_v2.json"
REPORT = ROOT / "reports"


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def reconstruct(clean: torch.Tensor, k: torch.Tensor, n: torch.Tensor,
                blur: torch.Tensor) -> torch.Tensor:
    """One field, one application: blur, then gain and addition (clipped)."""
    h, w = clean.shape[-2:]
    kk = F.interpolate(k[None], size=(h, w), mode="bilinear", align_corners=False)
    nn = F.interpolate(n[None], size=(h, w), mode="bilinear", align_corners=False)
    kernel = gaussian_kernels(blur[None])
    padded = F.pad(clean[None], (5, 5, 5, 5), mode="replicate")
    out = F.conv2d(padded, kernel.repeat_interleave(clean.shape[0], 0),
                   groups=clean.shape[0]).clamp(0, 1)
    return (kk * out + nn).clamp(0, 1)


def main() -> None:
    config = json.loads(CONFIG.read_text())
    families = config["weather_families"]
    severities = config["severities"]
    size = config["image_size"]

    bank = np.load(FIELDS / "bank.npz")
    records = read_jsonl(FIELDS / "unlabeled_target_images.jsonl")
    rows = {r["sample_id"]: r for r in read_jsonl(ROOT / config["manifest"])}

    # The bank order is the estimation order: source, family, severity.
    expected = 0
    for source in {r["source_id"] for r in records}:
        for family in families:
            for severity in severities:
                expected += 1
    assert expected == len(bank["K"]) == len(records), (expected, len(bank["K"]), len(records))

    per_condition = {}
    for index, record in enumerate(records):
        source = rows[record["source_id"]]
        clean_path = ROOT / source["path"]
        clean = TF.to_tensor(clean_image(clean_path, size))
        corrupted_path = ROOT / record["path"]
        corrupted = TF.to_tensor(Image.open(corrupted_path).convert("RGB"))
        found = None
        for family in families:
            for severity in severities:
                token = hashlib.sha256(f"{record['source_id']}:{family}:{severity}".encode()).hexdigest()
                if token == record["image_id"]:
                    found = (family, severity)
        assert found, record["image_id"]
        family, severity = found
        field = reconstruct(clean, torch.tensor(bank["K"][index]), torch.tensor(bank["N"][index]),
                            torch.tensor(bank["blur"][index]))
        per_condition.setdefault((family, severity), []).append({
            "actual": psnr(clean.numpy(), corrupted.numpy()),
            "reconstructed": psnr(clean.numpy(), field.numpy()),
            "k_mean": float(bank["K"][index].mean()),
            "n_absmean": float(np.abs(bank["N"][index]).mean()),
            "noise": float(bank["sigma"][index]),
            "sigma_x": float(bank["blur"][index][0]),
            "sigma_y": float(bank["blur"][index][1]),
            "theta": float(bank["blur"][index][2]),
        })

    table = {}
    for (family, severity), items in per_condition.items():
        table[f"{family}_{severity}"] = {
            key: float(np.mean([item[key] for item in items]))
            for key in ("actual", "reconstructed", "k_mean", "n_absmean", "noise",
                        "sigma_x", "sigma_y", "theta")
        }

    # Strength of the evaluation conditions, rendered exactly as at test time.
    roles = partition_rows(read_jsonl(ROOT / config["manifest"]), config)
    evaluation = roles["evaluation"][:40]
    strength = {}
    for severity in severities:
        for pair in [(a, b) for i, a in enumerate(families) for b in families[i + 1:]]:
            values = []
            for row in evaluation:
                clean = clean_image(ROOT / row["path"], size)
                compound = apply_components(clean.copy(), list(pair), severity,
                                            row["sample_id"], "evaluation", config)
                values.append(psnr(np.asarray(clean, np.float64) / 255.0,
                                   np.asarray(compound, np.float64) / 255.0))
            strength[f"{'+'.join(pair)}_{severity}"] = float(np.mean(values))
        for family in families:
            values = []
            for row in evaluation:
                clean = clean_image(ROOT / row["path"], size)
                single = apply_components(clean.copy(), [family], severity,
                                          row["sample_id"], "evaluation", config)
                values.append(psnr(np.asarray(clean, np.float64) / 255.0,
                                   np.asarray(single, np.float64) / 255.0))
            strength[f"{family}_{severity}"] = float(np.mean(values))

    payload = {"fields_per_condition": {f"{k[0]}_{k[1]}": len(v) for k, v in per_condition.items()},
               "target_conditions": table, "evaluation_strength_psnr": strength,
               "evaluation_images_sampled": len(evaluation)}
    (REPORT / "maritime_field_match.json").write_text(json.dumps(payload, indent=1))
    write_markdown(payload, families, severities, REPORT / "MARITIME_FIELD_MATCH.md")
    for family in families:
        print(family, [round(table[f"{family}_{s}"]["actual"], 1) for s in severities],
              "recon", [round(table[f"{family}_{s}"]["reconstructed"], 1) for s in severities])


def write_markdown(payload, families, severities, path):
    table = payload["target_conditions"]
    strength = payload["evaluation_strength_psnr"]
    lines = ["# 目标域估计场与实际退化的匹配程度", "",
             "只读诊断：不训练、不调参、不修改估计器。每个单元格是该 (族, 严重度) 上 "
             f"{next(iter(payload['fields_per_condition'].values()))} 个目标图像的平均值。", "",
             "`actual`：干净图与该场估计时所看的退化图之间的 PSNR（要解释的损伤强度，越低越强）。"
             "`recon`：干净图经过存下来的场之后，与那张退化图之间的 PSNR（越高说明场解释得越好）。"
             "`gap`：两者之差。", "",
             "| 族 | 严重度 | actual | recon | gap | K 均值 | \\|N\\| 均值 | noise σ | σx | σy |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for family in families:
        for severity in severities:
            row = table[f"{family}_{severity}"]
            lines.append(
                f"| {family} | {severity} | {row['actual']:.2f} | {row['reconstructed']:.2f} "
                f"| {row['reconstructed'] - row['actual']:+.2f} | {row['k_mean']:.3f} "
                f"| {row['n_absmean']:.3f} | {row['noise']:.4f} "
                f"| {row['sigma_x']:.2f} | {row['sigma_y']:.2f} |")
    lines += ["", "## 估计场随严重度是否变化", "",
              "若场抓住了退化，K、\\|N\\|、noise σ 应随严重度单调变化。", "",
              "| 族 | 指标 | " + " | ".join(f"sev {s}" for s in severities) + " |",
              "|---|---|" + "---:|" * len(severities)]
    for family in families:
        for key in ("k_mean", "n_absmean", "noise", "sigma_x"):
            values = [table[f"{family}_{s}"][key] for s in severities]
            lines.append(f"| {family} | {key} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += ["", "## 评测条件本身的强度（PSNR，越低越强）", "",
              "| 条件 | " + " | ".join(f"sev {s}" for s in severities) + " |",
              "|---|---:|" + "---:|" * len(severities)]
    keys = sorted({k.rsplit("_", 1)[0] for k in strength})
    for key in keys:
        lines.append(f"| {key} | " + " | ".join(f"{strength[f'{key}_{s}']:.2f}" for s in severities) + " |")
    lines += ["", f"评测强度在 {payload['evaluation_images_sampled']} 张开发集图像上按正式渲染器测得；"
              "场上限为 8×8 网格，且场库只含单项退化。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
