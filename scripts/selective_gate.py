#!/usr/bin/env python
"""Apply the pre-registered selective-prediction gate (G1-G3).

Reads ``reports/selective/fold{f}_s{seed}.json`` for f in 0..2 and seed in
{17, 29}; compares ``drift_nearest`` against the entropy baseline and writes
``reports/selective/GATE.json``. Gate definitions: reports/PREREG_SELECTIVE_PROBE.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from maritime_robust.common import write_json  # noqa: E402

UNITS = [(f, s) for f in (0, 1, 2) for s in (17, 29)]
GAIN_PP = 0.05
MIN_POSITIVE = 5


def diff(unit, group):
    cells = unit["summary"][group]["auc"]
    return cells["drift_nearest"] - cells["ent"]


def main():
    units = {(f, s): json.loads((ROOT / f"reports/selective/fold{f}_s{s}.json").read_text())
             for f, s in UNITS}
    out = {"prereg": "reports/PREREG_SELECTIVE_PROBE.md",
           "criterion": {"gain": GAIN_PP, "min_positive_units": f"{MIN_POSITIVE}/{len(UNITS)}"},
           "units": {}}
    for group, key in (("combo_s5", "G1"), ("combo_low", "G2"),
                       ("single", "G3a"), ("clean", "G3b")):
        deltas = {f"f{f}_s{s}": diff(units[(f, s)], group) for f, s in UNITS}
        values = list(deltas.values())
        entry = {"deltas": deltas, "mean": sum(values) / len(values),
                 "positive_units": sum(v > 0 for v in values), "n_units": len(values)}
        if key.startswith("G1") or key.startswith("G2"):
            entry["pass"] = entry["mean"] >= GAIN_PP and entry["positive_units"] >= MIN_POSITIVE
        else:
            entry["pass"] = entry["mean"] >= -GAIN_PP
        out["units"][key] = {"group": group, **entry}

    out["admitted"] = all(v["pass"] for v in out["units"].values())
    out["circularity_control"] = {
        f"f{f}_s{s}": {g: units[(f, s)]["summary"][g]["auc"]["drift_pred"]
                       - units[(f, s)]["summary"][g]["auc"]["ent"]
                       for g in ("combo_s5", "combo_low")}
        for f, s in UNITS}
    for name, entry in out["units"].items():
        print(f"{name} {entry['group']:10s} mean {entry['mean']:+.3f} "
              f"positive {entry['positive_units']}/{entry['n_units']} pass={entry['pass']}")
    print("admitted:", out["admitted"])
    write_json(ROOT / "reports/selective/GATE.json", out)


if __name__ == "__main__":
    main()
