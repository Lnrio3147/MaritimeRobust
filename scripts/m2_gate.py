#!/usr/bin/env python
"""Apply the pre-registered M2 admission gate to the per-seed evaluations.

Reads ``reports/m2/{d1,d1inv}_s{17,29}.json`` and writes ``reports/m2/GATE.json``
with the G1-G4 verdicts of reports/PREREG_DRIFT_WEIGHTING.md, for both the
primary arm (alpha=+1) and the pre-registered directional control (alpha=-1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from maritime_robust.common import write_json  # noqa: E402

SEEDS = (17, 29)
G1_MIN_PP = 1.5
G4_MAX_CLEAN_DROP_PP = 1.0


def arm(tag):
    runs = {}
    for seed in SEEDS:
        path = ROOT / f"reports/m2/{tag}_s{seed}.json"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        runs[seed] = json.loads(path.read_text())
    low = {s: runs[s]["summary"]["combo_low"] for s in SEEDS}
    clean = {s: runs[s]["summary"]["clean"] for s in SEEDS}
    delta = {s: low[s]["delta_pp"] for s in SEEDS}
    ci = {s: low[s]["cluster_bootstrap"]["ci95_pp"] for s in SEEDS}
    clean_delta = {s: clean[s]["delta_pp"] for s in SEEDS}
    gate = {
        "G1_magnitude": {"threshold_pp": G1_MIN_PP, "value_pp": sum(delta.values()) / len(SEEDS),
                         "pass": sum(delta.values()) / len(SEEDS) >= G1_MIN_PP},
        "G2_consistency": {"per_seed_pp": delta, "pass": all(v > 0 for v in delta.values())},
        "G3_significance": {"ci95_pp": ci,
                            "pass": all(lo > 0 for lo, _ in ci.values())},
        "G4_clean": {"per_seed_pp": clean_delta, "threshold_pp": -G4_MAX_CLEAN_DROP_PP,
                     "pass": all(v >= -G4_MAX_CLEAN_DROP_PP for v in clean_delta.values())},
    }
    gate["admitted"] = all(gate[k]["pass"] for k in
                           ("G1_magnitude", "G2_consistency", "G3_significance", "G4_clean"))
    return gate


def main():
    out = {"prereg": "reports/PREREG_DRIFT_WEIGHTING.md",
           "arms": {tag: arm(tag) for tag in ("d1", "d1inv")}}
    for tag, gate in out["arms"].items():
        print(f"{tag:6s} admitted={gate['admitted']}")
        for key in ("G1_magnitude", "G2_consistency", "G3_significance", "G4_clean"):
            print(f"  {key:18s} pass={gate[key]['pass']}  "
                  + json.dumps({k: v for k, v in gate[key].items() if k != 'pass'}))
    write_json(ROOT / "reports/m2/GATE.json", out)


if __name__ == "__main__":
    main()
