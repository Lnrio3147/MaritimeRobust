"""Adjudicate the otheraug arm against its pre-registered predictions.

The arm trains on photometric/geometric augmentations whose forms the test
families share, but on none of the families themselves. O1-O3 were frozen in
``reports/PREREG_CIFAR.md`` before the arm ran; this script only reads the
stored flags and reports which ones held.

    python scripts/cifar_otheraug_report.py --arm otheraug_s17 --seed 17

Writes reports/OTHERAUG.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PREFIX = "cifar10_resnet18_cifar_"


def load_accuracy(arm: str) -> dict:
    path = PROJECT / "reports/cifar_grid" / (PREFIX + arm) / "summary.json"
    if not path.exists():
        raise SystemExit(f"no grid for {arm}: run cifar_eval.py first")
    return json.loads(path.read_text())["accuracy"]


def means(acc: dict) -> dict:
    pairs = [v for k, v in acc.items() if "+" in k]
    singles = [v for k, v in acc.items() if "+" not in k and k != "clean"]
    return {"clean": acc["clean"] * 100,
            "singles": sum(singles) / len(singles) * 100,
            "pairs": sum(pairs) / len(pairs) * 100,
            "n_pairs": len(pairs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="otheraug_s17")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="reports/OTHERAUG.md")
    args = ap.parse_args()

    arm = means(load_accuracy(args.arm))
    k15 = means(load_accuracy(f"lattice_k15_s{args.seed}"))
    augmix = means(load_accuracy(f"augmix_s{args.seed}"))
    clean = means(load_accuracy(f"clean_s{args.seed}"))
    heldout = json.loads((PROJECT / "reports/cifar_heldout.json").read_text())

    delta = heldout[args.arm]["heldout_delta_s3"]
    delta_clean = heldout[f"clean_s{args.seed}"]["heldout_delta_s3"]
    delta_k15 = heldout[f"lattice_k15_s{args.seed}"]["heldout_delta_s3"]

    o1_threshold = k15["pairs"] - 5.0
    o1 = arm["pairs"] <= o1_threshold
    o1_strong = k15["pairs"] - arm["pairs"] < 5.0   # the row that would hurt the claim
    o2 = delta > 1.0
    o3 = arm["clean"] >= 95.0

    def mark(ok):
        return "holds" if ok else "fails"

    lines = [
        "# otheraug: pre-registered predictions O1-O3", "",
        f"Arm `{args.arm}`: augmentation families disjoint from the fifteen test "
        "corruption families, same 200-epoch recipe, same 601-condition grid.", "",
        "| arm | clean | singles | pairs |", "|---|---:|---:|---:|",
    ]
    for name, m in (("clean", clean), ("AugMix", augmix), ("lattice k=15", k15),
                    (args.arm, arm)):
        lines.append(f"| {name} | {m['clean']:.2f} | {m['singles']:.2f} | {m['pairs']:.2f} |")
    lines += ["", "| id | registered | measured | verdict |", "|---|---|---|---|",
              f"| O1 | pairs <= {o1_threshold:.2f} (k=15 minus 5 pp) | "
              f"{arm['pairs']:.2f} (gap {k15['pairs'] - arm['pairs']:+.2f} pp) | {mark(o1)} |",
              f"| O2 | held-out excess at s3 > +1 pp | {delta:+.2f} pp | {mark(o2)} |",
              f"| O3 | clean >= 95.0 | {arm['clean']:.2f} | {mark(o3)} |", "",
              "Context: held-out excess at s3 is "
              f"{delta_clean:+.2f} pp for clean, {delta_k15:+.2f} pp for k=15, "
              f"{heldout[f'augmix_s{args.seed}']['heldout_delta_s3']:+.2f} pp for AugMix.", ""]

    if o1 and o2 and o3:
        verdict = ("O1 and O2 both hold: compound accuracy tracks whether the "
                   "evaluated families were trained on, not augmentation strength. "
                   "Section 4.7 gains this arm as a control; the claim is unchanged.")
    elif not o1 and o3:
        verdict = ("O1 fails -- a family-disjoint arm reaches within 5 pp of k=15. "
                   "The family-coverage explanation must be withdrawn: the coverage "
                   "result reduces to augmentation strength, and Section 4.7 and the "
                   "abstract have to be rewritten.")
        if o1_strong:
            verdict += " The arm is at or above k=15, so the coverage arms have no margin left."
    elif not o2 and o1:
        verdict = ("O2 fails while O1 holds: the excess can be removed by any "
                   "auxiliary task, so `delta` no longer isolates family coverage "
                   "and the claim is downgraded accordingly.")
    else:
        verdict = ("O3 fails: the arm itself is unstable, so no inference is drawn "
                   "from it. Re-check training before using these numbers.")
    lines += [f"**Verdict.** {verdict}", ""]

    (PROJECT / args.out).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
