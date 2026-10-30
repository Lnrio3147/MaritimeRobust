#!/usr/bin/env python
"""Occlusion intervention on FGSC-23 civilian test images.

FGSC-23 ships neither boxes nor masks (verified: no `target_mask` in the
manifest, no annotation files in the raw release). A part-level mask study is
therefore not possible on the main dataset; FGSRCS has oriented boxes but a
different taxonomy and domain. This experiment instead destroys image regions
with a fixed gray patch, before letterbox padding, and compares:

  clean                 no patch
  center_third          square of 1/3 image area at the image centre
  corner_third          same-size square at the calmest corner (area control)
  center_half           square of 1/2 image area at the centre (dose)

All three baseline models are evaluated on the same 236 test images. The clean
condition must reproduce the saved clean predictions exactly; that is the
pipeline-parity check.

The intervention tests position specificity and error structure, not ship
parts. A positive result (centre much worse than corner, confusion pairs
matching severe degradation) supports "localized central evidence loss"; it
cannot localize bow/stern/superstructure.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maritime_robust.common import ROOT as PROJECT, Budget, read_jsonl, write_json, write_jsonl  # noqa: E402
from maritime_robust.engine import load_run, metrics  # noqa: E402

MODELS = {
    "resnet50": "runs/fgsc23_research_civilian/resnet50_combo_s17",
    "convnext_tiny": "runs/fgsc23_research_civilian/convnext_tiny_combo_s17",
    "swin_t": "runs/fgsc23_research_civilian/swin_t_combo_s17",
}
DISPLAY = {"resnet50": "ResNet-50", "convnext_tiny": "ConvNeXt-Tiny", "swin_t": "Swin-Tiny"}
CONDITIONS = [("clean", "clean", 0.0), ("corner_third", "corner", 1 / 3),
              ("center_third", "center", 1 / 3), ("center_half", "center", 1 / 2)]
FILL = (124, 116, 104)


class OccludedTest(Dataset):
    def __init__(self, rows, mode, fraction):
        self.rows, self.mode, self.fraction = rows, mode, fraction

    def __len__(self):
        return len(self.rows)

    def image(self, index):
        with Image.open(PROJECT / self.rows[index]["path"]) as f:
            image = f.convert("RGB")
        image = ImageOps.contain(image, (224, 224), Image.Resampling.BICUBIC)
        if self.mode != "clean":
            image = occlude(image, self.mode, self.fraction)
        return ImageOps.pad(image, (224, 224), Image.Resampling.BICUBIC, color=FILL)

    def __getitem__(self, index):
        tensor = TF.normalize(TF.to_tensor(self.image(index)),
                              [.485, .456, .406], [.229, .224, .225])
        return tensor, self.rows[index]["label"], index


def occlude(image, mode, fraction):
    array = np.asarray(image).copy()
    height, width = array.shape[:2]
    side = min(int(round((fraction * height * width) ** 0.5)), height, width)
    if mode == "center":
        y0, x0 = (height - side) // 2, (width - side) // 2
    else:
        gray = array.mean(2)
        dx = np.pad(np.abs(np.diff(gray, axis=0)), ((0, 1), (0, 0)))
        dy = np.pad(np.abs(np.diff(gray, axis=1)), ((0, 0), (0, 1)))
        energy = dx + dy
        corners = [(0, 0), (0, width - side), (height - side, 0), (height - side, width - side)]
        score = [energy[y:y + side, x:x + side].mean() for y, x in corners]
        y0, x0 = corners[int(np.argmin(score))]
    array[y0:y0 + side, x0:x0 + side] = FILL
    return Image.fromarray(array)


@torch.inference_mode()
def predict(model, dataset, num_classes, batch_size=32, workers=4):
    model.eval()
    records = []
    for x, y, indices in DataLoader(dataset, batch_size, num_workers=workers, shuffle=False):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x.cuda(non_blocking=True))
        probs = logits.float().softmax(1).cpu().numpy()
        for index, label, p in zip(indices.tolist(), y.tolist(), probs):
            row = dataset.rows[index]
            records.append({"sample_id": row["sample_id"], "label": label,
                            "prediction": int(p.argmax()), "max_prob": float(p.max())})
    score = metrics([r["label"] for r in records], [r["prediction"] for r in records], num_classes)
    return score, records


def confusion_pairs(records, labels, top=5):
    counts = {}
    for r in records:
        if r["prediction"] != r["label"]:
            pair = (labels[r["label"]], labels[r["prediction"]])
            counts[pair] = counts.get(pair, 0) + 1
    return [[a, b, n] for (a, b), n in sorted(counts.items(), key=lambda kv: -kv[1])[:top]]


def main():
    rows = [r for r in read_jsonl(PROJECT / "data/manifests/fgsc23_research_civilian.jsonl")
            if r["split"] == "test"]
    labels = {r["label"]: r["label_name"] for r in rows}
    n = max(labels) + 1
    result = {"n": len(rows), "conditions": {}, "models": {}}
    with Budget("evaluation") as budget:
        for name, rel in MODELS.items():
            model, settings, _ = load_run(PROJECT / rel)
            saved_clean = read_jsonl(PROJECT / rel / "evaluation/clean.jsonl")
            saved_degraded = read_jsonl(PROJECT / rel / "evaluation/matched_haze+glare_5.jsonl")
            per_condition = {}
            for cond_name, mode, fraction in CONDITIONS:
                score, records = predict(model, OccludedTest(rows, mode, fraction), n)
                if cond_name == "clean":
                    same = [r["prediction"] for r in records] == [r["prediction"] for r in saved_clean]
                    score["parity_with_saved_clean"] = same
                    if not same:
                        raise AssertionError(f"{name}: clean predictions changed")
                per_condition[cond_name] = {"score": score, "records": records}
                budget.check()
            clean_records = per_condition["clean"]["records"]
            clean_accuracy = per_condition["clean"]["score"]["accuracy"]
            clean_wrong = {r["sample_id"] for r in clean_records if r["prediction"] != r["label"]}
            degraded_wrong = {r["sample_id"] for r in saved_degraded if r["prediction"] != r["label"]}
            entry = {"conditions": {}, "confusions": {}}
            for cond_name, _, _ in CONDITIONS:
                recs = per_condition[cond_name]["records"]
                wrong = {r["sample_id"] for r in recs if r["prediction"] != r["label"]}
                entry["conditions"][cond_name] = {
                    "score": per_condition[cond_name]["score"],
                    "accuracy": per_condition[cond_name]["score"]["accuracy"],
                    "macro_f1": per_condition[cond_name]["score"]["macro_f1"],
                    "delta_accuracy_pp": 100 * (per_condition[cond_name]["score"]["accuracy"] -
                                                clean_accuracy),
                    "wrong": len(wrong)}
                if cond_name != "clean":
                    entry["conditions"][cond_name]["flips_vs_clean"] = len(wrong - clean_wrong)
                    entry["conditions"][cond_name]["recovered_vs_clean"] = len(clean_wrong - wrong)
                entry["confusions"][cond_name] = confusion_pairs(recs, labels)
            center_wrong = {r["sample_id"] for r in per_condition["center_third"]["records"]
                            if r["prediction"] != r["label"]}
            entry["overlap"] = {
                "center_vs_haze_glare_5_jaccard": len(center_wrong & degraded_wrong) /
                                                  max(1, len(center_wrong | degraded_wrong)),
                "center_errors_also_degraded_wrong": len(center_wrong & degraded_wrong) /
                                                     max(1, len(center_wrong)),
                "degraded_errors_also_center_wrong": len(center_wrong & degraded_wrong) /
                                                     max(1, len(degraded_wrong)),
            }
            entry["degraded_reference"] = {
                "accuracy": json.loads((PROJECT / rel / "evaluation/matched_haze+glare_5.json")
                                       .read_text())["accuracy"],
                "confusions": confusion_pairs(saved_degraded, labels)}
            result["models"][name] = entry
            print(json.dumps({"model": name, "clean": per_condition["clean"]["score"]["accuracy"],
                              "corner": per_condition["corner_third"]["score"]["accuracy"],
                              "center": per_condition["center_third"]["score"]["accuracy"],
                              "center_half": per_condition["center_half"]["score"]["accuracy"]}), flush=True)
    write_json(PROJECT / "reports/part_occlusion.json", result)


if __name__ == "__main__":
    main()
