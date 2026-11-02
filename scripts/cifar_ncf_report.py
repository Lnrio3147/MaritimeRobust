"""Adjudicate the NCF arms (docs/MODULE_NCF.md §7).

Two questions, both decided by accuracy on the same 601-condition grid:

  F2  does composing two sampled fields beat one sampled field?
      If not, the module's core claim -- joint coverage in field space -- is
      unsupported and only the weaker "deployment-aware augmentation" survives.

  F1  on conditions containing a family the field bank never contained, does
      NCF beat AugMix? That is the direct retest of the paper's P10 failure.

Writes reports/NCF_REPORT.md.

    python scripts/cifar_ncf_report.py --tier a
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from maritime_robust.external import cifar_c  # noqa: E402

HELD_OUT_FAMILIES = ("fog", "frost", "snow")


def load(arm: str) -> dict | None:
    path = PROJECT / "reports/cifar_grid" / arm / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["accuracy"]


def split(acc: dict) -> dict:
    """Cell means for singles, all pairs, and pairs with/without held-out families."""
    out = {"clean": [acc["clean"]], "singles": [], "pairs": [],
           "pairs_held": [], "pairs_other": []}
    for key, value in acc.items():
        if key == "clean":
            continue
        parts = key.rsplit("_", 1)
        fam, sev = parts[0], parts[1]
        if "+" in fam:
            out["pairs"].append(value)
            (out["pairs_held"] if any(f in HELD_OUT_FAMILIES for f in fam.split("+"))
             else out["pairs_other"]).append(value)
        else:
            out["singles"].append(value)
    return {k: (sum(v) / len(v) * 100 if v else float("nan")) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=None, help="comma-separated arm tags")
    ap.add_argument("--out", default="reports/NCF_REPORT.md")
    args = ap.parse_args()

    arms = (args.arms.split(",") if args.arms else
            ["clean_s17", "augmix_s17", "singles_s17", "lattice_k15_s17",
             "ncf_s_a_s17", "ncf_c_a_s17",
             "ncf_c_a_r2_s17", "ncf_c_b_r2_s17", "ncf_c_ab_r2_s17",
             "ncf_c_bb_r2_s17", "ncf_s_ab_r2_s17", "ncf_c_ab_r2_s29"])
    prefix = "cifar10_resnet18_cifar_"
    rows = {a: split(m) for a in arms if (m := load(prefix + a))}

    lines = ["# NCF arm adjudication", "",
             "Accuracy (%) on the same 601-condition grid as Table 1. "
             "`paired-held` is the mean over pair conditions containing fog, frost or snow "
             "-- the families excluded from the tier-B field bank and never named to the "
             "estimator in any bank.", "",
             "| arm | clean | singles | pairs | pairs w/ held-out families | pairs w/o |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm, r in rows.items():
        lines.append(f"| {arm} | {r['clean']:.2f} | {r['singles']:.2f} | {r['pairs']:.2f} | "
                     f"{r['pairs_held']:.2f} | {r['pairs_other']:.2f} |")
    (PROJECT / args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
