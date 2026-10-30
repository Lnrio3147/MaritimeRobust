"""Training views for the second screening batch.

``AugDataset`` reproduces ``engine.ShipDataset.image`` draw-for-draw (same
geometry seeds, same combined-recipe draws, same corruption seeds) so that a
"none" arm trained through it is numerically comparable to the S1/S2 baseline.
It adds two things the frozen class cannot express:

* ``kind`` reported per sample, for degradation-group DRO (T2);
* a curriculum over severity and single/combined mixture (T3).
"""
from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from maritime_robust.common import ROOT, stable_seed
from maritime_robust.degradations import corrupt, corruption_seed
from maritime_robust.engine import combined_recipe
from maritime_robust.methods import clip_mask

MEAN, STD, FILL = [.485, .456, .406], [.229, .224, .225], (124, 116, 104)
KINDS = ("clean", "haze", "cloud", "glare", "haze+glare", "haze+cloud")
KIND_INDEX = {k: i for i, k in enumerate(KINDS)}


class MaskedClean(Dataset):
    """Validation view for M1: same geometry as ``ShipDataset(val_rows)`` (contain
    then pad, no augmentation, no corruption) with the mask channel appended."""

    META = json.dumps({"kind": "clean", "severity": 0, "domain": "matched",
                       "placement": "random"}, separators=(",", ":"))

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(ROOT / row["path"]) as handle:
            image = handle.convert("RGB")
        image = ImageOps.contain(image, (224, 224), Image.Resampling.BICUBIC)
        raw = TF.to_tensor(ImageOps.pad(image, (224, 224), Image.Resampling.BICUBIC, color=FILL))
        tensor = torch.cat([TF.normalize(raw, MEAN, STD), clip_mask(raw)], 0)
        return tensor, row["label"], index, self.META


class AugDataset(Dataset):
    def __init__(self, rows, seed=17, curriculum=False, total_epochs=40, with_mask=False,
                 recipe_path=None):
        self.rows = rows
        self.seed = seed
        self.epoch = 0
        self.curriculum = curriculum
        self.with_mask = with_mask
        self.total_epochs = total_epochs
        self.recipe = combined_recipe(recipe_path)
        self.clean_probability = self.recipe["clean_probability"]

    def __len__(self):
        return len(self.rows)

    def schedule(self):
        """T3: single-only, low severity -> full recipe, high severity."""
        if not self.curriculum:
            return None
        frac = self.epoch / max(1, self.total_epochs)
        if frac < 1 / 3:
            return {"single": 1.0, "single_sev": [1, 2], "combined_sev": [1, 2]}
        if frac < 2 / 3:
            return {"single": 0.5, "single_sev": [1, 2, 3], "combined_sev": [1, 2]}
        return None

    def draw(self, rng):
        stage = self.schedule()
        # identical draw order to ShipDataset: one clean gate, then mixture
        if rng.random() < self.clean_probability:
            return {"kind": "clean", "severity": 0}
        if stage is None:
            if rng.random() < self.recipe["degraded_mixture"]["single"]:
                return {"kind": str(rng.choice(self.recipe["single_kinds"])),
                        "severity": int(rng.choice(self.recipe["single_severities"]))}
            return {"kind": str(rng.choice(self.recipe["combined_kinds"])),
                    "severity": int(rng.choice(self.recipe["combined_severities"]))}
        if rng.random() < stage["single"]:
            return {"kind": str(rng.choice(self.recipe["single_kinds"])),
                    "severity": int(rng.choice(stage["single_sev"]))}
        return {"kind": str(rng.choice(self.recipe["combined_kinds"])),
                "severity": int(rng.choice(stage["combined_sev"]))}

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(ROOT / row["path"]) as handle:
            image = handle.convert("RGB")
        image = ImageOps.contain(image, (224, 224), Image.Resampling.BICUBIC)
        rng = np.random.default_rng(stable_seed(self.seed, self.epoch, row["sample_id"], "geometry"))
        if rng.random() < .5:
            image = ImageOps.mirror(image)
        if rng.random() < .5:
            image = ImageOps.flip(image)
        image = image.rotate(90 * int(rng.integers(4)), expand=True)
        rng = np.random.default_rng(stable_seed(self.seed, self.epoch, row["sample_id"], "augmentation"))
        condition = self.draw(rng)
        seed = corruption_seed(row["sample_id"], f"train_epoch_{self.epoch}", condition["kind"],
                               condition["severity"], "matched", self.seed)
        image, _ = corrupt(image, seed=seed, kind=condition["kind"],
                           severity=condition["severity"], domain="matched", placement="random")
        image = ImageOps.pad(image, (224, 224), method=Image.Resampling.BICUBIC, color=FILL)
        raw = TF.to_tensor(image)
        tensor = TF.normalize(raw, MEAN, STD)
        if self.with_mask:
            tensor = torch.cat([tensor, clip_mask(raw)], 0)
        return tensor, row["label"], KIND_INDEX[condition["kind"]]


class DriftDataset(AugDataset):
    """``AugDataset`` plus the uncorrupted view of the *same* geometry.

    Both views share the geometry draw, so the second one is the reference for a
    per-image drift signal ``d = 1 − cos(f(clean), f(aug))`` measured with a
    frozen teacher. Draw-for-draw identical to ``AugDataset`` otherwise, so an
    α = 0 run through this class must reproduce the S1 baseline.
    """

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(ROOT / row["path"]) as handle:
            image = handle.convert("RGB")
        image = ImageOps.contain(image, (224, 224), Image.Resampling.BICUBIC)
        rng = np.random.default_rng(stable_seed(self.seed, self.epoch, row["sample_id"], "geometry"))
        if rng.random() < .5:
            image = ImageOps.mirror(image)
        if rng.random() < .5:
            image = ImageOps.flip(image)
        image = image.rotate(90 * int(rng.integers(4)), expand=True)
        clean = TF.normalize(TF.to_tensor(
            ImageOps.pad(image, (224, 224), Image.Resampling.BICUBIC, color=FILL)), MEAN, STD)
        rng = np.random.default_rng(stable_seed(self.seed, self.epoch, row["sample_id"], "augmentation"))
        condition = self.draw(rng)
        seed = corruption_seed(row["sample_id"], f"train_epoch_{self.epoch}", condition["kind"],
                               condition["severity"], "matched", self.seed)
        image, _ = corrupt(image, seed=seed, kind=condition["kind"],
                           severity=condition["severity"], domain="matched", placement="random")
        image = ImageOps.pad(image, (224, 224), method=Image.Resampling.BICUBIC, color=FILL)
        tensor = TF.normalize(TF.to_tensor(image), MEAN, STD)
        return tensor, clean, row["label"], KIND_INDEX[condition["kind"]]
