"""Find, per family, the level whose single-family damage equals a target.

The coverage arms are all matched at a 20 pp single-family drop. That matches the
*components*, but a compound image carries two of them, so a pair arm applies
roughly twice the damage per image that the singles control does. Any comparison
between paired and unpaired augmentation therefore confounds pairing with damage
per sample. This calibrates a second ladder at twice the damage so the confound
can be removed: an unpaired arm at that level carries the same damage per image
as a pair at the 20 pp level.

Levels are found by sweeping the continuous level parameter on the validation
split of one fold with a trained checkpoint, and picking the level whose accuracy
drop is closest to the target. Where a family cannot reach the target even at the
top of the grid, the maximum reachable level is recorded and the family is
flagged -- a matched-damage arm is then not exactly matched, and the report says
which families are short.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from maritime_robust.common import ROOT, Budget, read_jsonl, seed_all, write_json  # noqa: E402
from maritime_robust.corruptions_extra import FAMILIES, apply  # noqa: E402
from maritime_robust.engine import ShipDataset, loader, predict  # noqa: E402
from maritime_robust.methods import build as build_method  # noqa: E402


class LevelDataset(ShipDataset):
    """Validation images with one family applied at a fixed level."""

    def __init__(self, rows, family, level, seed=0):
        super().__init__(rows)
        self.family, self.level, self.base_seed = family, level, seed

    def image(self, index):
        image, meta = super().image(index)
        row = self.rows[index]
        corrupted = apply(image, self.family, self.level,
                          seed=(self.base_seed * 1_000_003 + int(row["sample_id"])) % (2**31))
        return corrupted, {"kind": self.family, "level": self.level}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--model", default="convnext_tiny")
    ap.add_argument("--run-dir", default=None,
                    help="checkpoint to calibrate against; defaults to the baseline cell")
    ap.add_argument("--target", type=float, default=40.0)
    ap.add_argument("--grid", default="0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95")
    ap.add_argument("--out", default="reports/lattice_levels.json")
    args = ap.parse_args()

    run_dir = ROOT / (args.run_dir or
                      f"runs/fgsc23_civ8_fold{args.fold}/{args.model}_none_e40_s{args.seed}")
    checkpoint = torch.load(run_dir / "best.pt", map_location="cuda", weights_only=False)
    manifest = PROJECT / f"data/manifests/folds/fgsc23_civ8_fold{args.fold}.jsonl"
    rows = read_jsonl(manifest)
    val_rows = [r for r in rows if r["split"] == "val"]
    n = max(r["label"] for r in rows) + 1

    config = json.loads((ROOT / "configs/protocol.json").read_text())
    grid = [float(v) for v in args.grid.split(",")]
    levels, reachable, clean = {}, {}, None

    with Budget("evaluation") as budget:
        seed_all(args.seed)
        torch.set_num_threads(4)
        model = build_method(args.model, n, "none").cuda()
        model.load_state_dict(checkpoint["model"])
        model.eval()
        clean, _ = predict(model, ShipDataset(val_rows), n, config["batch_size"],
                           config["workers"], budget)
        clean_acc = clean["accuracy"]
        print(json.dumps({"clean_accuracy": clean_acc}), flush=True)

        for family in FAMILIES:
            curve = []
            for level in grid:
                result, _ = predict(model, LevelDataset(val_rows, family, level),
                                    n, config["batch_size"], config["workers"], budget)
                curve.append((level, (clean_acc - result["accuracy"]) * 100))
                print(json.dumps({"family": family, "level": level,
                                  "drop_pp": curve[-1][1]}), flush=True)
            best = min(curve, key=lambda kv: abs(kv[1] - args.target))
            levels[family] = best[0]
            reachable[family] = {"level": best[0], "drop_pp": best[1],
                                 "max_drop_pp": max(drop for _, drop in curve),
                                 "matched": abs(best[1] - args.target) <= 5.0}

    write_json(PROJECT / args.out,
               {"target_pp": args.target, "fold": args.fold, "seed": args.seed,
                "checkpoint": str(run_dir.relative_to(ROOT)), "clean_accuracy": clean_acc,
                "levels": levels, "detail": reachable})
    short = [f for f, d in reachable.items() if not d["matched"]]
    print(json.dumps({"levels": levels, "short": short}))


if __name__ == "__main__":
    main()
