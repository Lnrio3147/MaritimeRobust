"""Shared pipeline for zero-training probes (restoration / local recovery).

Replicates the frozen evaluation pipeline exactly: contain -> corrupt (same
source-level seed) -> pad -> ImageNet normalize. No training-time geometry
augmentation is applied.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maritime_robust.common import ROOT as PROJECT, read_jsonl  # noqa: E402
from maritime_robust.degradations import conditions as all_conditions  # noqa: E402
from maritime_robust.degradations import corrupt, corruption_seed  # noqa: E402
from maritime_robust.engine import metrics  # noqa: E402

MEAN = [.485, .456, .406]
STD = [.229, .224, .225]
FILL = (124, 116, 104)
MODELS = {
    "resnet50": "runs/fgsc23_research_civilian/resnet50_combo_s17",
    "convnext_tiny": "runs/fgsc23_research_civilian/convnext_tiny_combo_s17",
    "swin_t": "runs/fgsc23_research_civilian/swin_t_combo_s17",
}
DISPLAY = {"resnet50": "ResNet-50", "convnext_tiny": "ConvNeXt-Tiny", "swin_t": "Swin-Tiny"}


def test_rows():
    return [r for r in read_jsonl(PROJECT / "data/manifests/fgsc23_research_civilian.jsonl")
            if r["split"] == "test"]


def condition(name):
    for c in all_conditions():
        if c["name"] == name:
            return dict(c)
    raise KeyError(name)


def content(row):
    with Image.open(PROJECT / row["path"]) as f:
        image = f.convert("RGB")
    return ImageOps.contain(image, (224, 224), Image.Resampling.BICUBIC)


def degrade(image, row, cond):
    seed = corruption_seed(row["sample_id"], "test", cond["kind"], cond["severity"],
                           cond["domain"], 0)
    return corrupt(image, seed=seed, kind=cond["kind"], severity=cond["severity"],
                   domain=cond["domain"], placement=cond["placement"])


def pad_norm(image):
    padded = ImageOps.pad(image, (224, 224), Image.Resampling.BICUBIC, color=FILL)
    return TF.normalize(TF.to_tensor(padded), MEAN, STD)


def to_tensor(image):
    return pad_norm(image)


@torch.inference_mode()
def probs_for(model, tensors, batch_size=64):
    model.eval()
    out = []
    for i in range(0, len(tensors), batch_size):
        x = torch.stack(tensors[i:i + batch_size]).cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        out.append(logits.float().softmax(1).cpu())
    return torch.cat(out)


def score_of(probs, labels, n):
    preds = probs.argmax(1).tolist()
    return metrics(labels, preds, n), preds


def margin(probs):
    top2 = probs.topk(2, dim=1).values
    return (top2[:, 0] - top2[:, 1]).tolist()


def load_models(names=None):
    from maritime_robust.engine import load_run
    loaded = {}
    for name in names or MODELS:
        model, settings, _ = load_run(PROJECT / MODELS[name])
        loaded[name] = (model, settings)
    return loaded


def saved_clean_predictions(name):
    rows = read_jsonl(PROJECT / MODELS[name] / "evaluation/clean.jsonl")
    return [r["prediction"] for r in rows]


def check_clean_parity(name, preds):
    expected = saved_clean_predictions(name)
    if preds != expected:
        bad = sum(a != b for a, b in zip(preds, expected))
        raise AssertionError(f"{name}: clean predictions differ in {bad}/{len(preds)} samples")
