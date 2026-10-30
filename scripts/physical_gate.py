#!/usr/bin/env python
"""Apply the pre-registered renderer-validity gate (P1-P4).

Reads ``reports/physical/fold0_s{17,29}.json`` and writes
``reports/physical/GATE.json``. Definitions: reports/PREREG_PHYSICAL_RENDERER.md.

P1  E(5) >= 5 pp             the cliff survives the re-rendering
P2  |E(s)| <= 3 for s=2,3,4   additivity at low severity survives
P3  severity matching        |measured - target| <= 0.05 saturation, 0.03 transmission
P4  depth ablation           |E_depth - E_uniform| >= 3 pp (secondary)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from maritime_robust.common import write_json  # noqa: E402

SEEDS = (17, 29)
P1_MIN_PP = 5.0
P2_MAX_PP = 3.0
SAT_TOLERANCE = 0.05
TRANSMISSION_TOLERANCE = 0.03
P4_MIN_PP = 3.0


def evaluate(payload):
    excess = {int(k): v for k, v in payload["excess"].items()}
    matching = {}
    for severity, cell in excess.items():
        haze = payload["cells"][f"physical_haze_{severity}"]
        matching[severity] = {
            "saturation_delta": abs(cell["measured_saturation"] - cell["target_saturation"]),
            "transmission_delta": abs(haze["measured_transmission"]
                                      - payload["targets"][str(severity)]["haze_transmission"]),
        }
    ablation = {int(k): v for k, v in payload["depth_ablation"].items()}
    return {
        "P1_cliff_survives": {"E5_pp": excess[5]["E_pp"], "threshold_pp": P1_MIN_PP,
                              "pass": excess[5]["E_pp"] >= P1_MIN_PP},
        "P2_low_severity_additive": {
            "E_pp": {s: excess[s]["E_pp"] for s in (2, 3, 4)}, "threshold_pp": P2_MAX_PP,
            "pass": all(abs(excess[s]["E_pp"]) <= P2_MAX_PP for s in (2, 3, 4))},
        "P3_severity_matching": {
            "per_severity": matching,
            "pass": all(m["saturation_delta"] <= SAT_TOLERANCE
                        and m["transmission_delta"] <= TRANSMISSION_TOLERANCE
                        for m in matching.values())},
        "P4_depth_ablation": {
            "per_severity": ablation, "threshold_pp": P4_MIN_PP,
            "pass": any(abs(v["delta_pp"]) >= P4_MIN_PP for v in ablation.values())},
    }


def main():
    out = {"prereg": "reports/PREREG_PHYSICAL_RENDERER.md", "per_seed": {}}
    for seed in SEEDS:
        payload = json.loads((ROOT / f"reports/physical/fold0_s{seed}.json").read_text())
        out["per_seed"][str(seed)] = evaluate(payload)

    # Stage 2 (pre-registration section 4): retrain under the physical renderer.
    # Only P1 and P2 decide anything here; P3 is a property of the renderer, not
    # of the model, and was already settled in Stage 1.
    stage2 = {}
    for seed in SEEDS:
        path = ROOT / f"reports/physical/convnext_tiny_p1_e40_s{seed}.json"
        stage2[str(seed)] = evaluate(json.loads(path.read_text())) if path.exists() else None
    if all(v is not None for v in stage2.values()):
        out["stage2_per_seed"] = stage2
        out["stage2_superadditive"] = all(stage2[str(s)]["P1_cliff_survives"]["pass"]
                                          for s in SEEDS)
        out["stage2_decision"] = (
            "superadditivity at s=5 survives independent rendering and matched training"
            if out["stage2_superadditive"] else
            "superadditivity does not survive matched training: the phenomenon is "
            "renderer-dependent and the claims must be narrowed")

    gates = ("P1_cliff_survives", "P2_low_severity_additive", "P3_severity_matching")
    out["stage1_admitted"] = all(out["per_seed"][str(s)][g]["pass"]
                                 for s in SEEDS for g in gates)
    decisions = {
        False: "Stage 2 (retrain with the physical renderer) is mandatory before any "
               "conclusion about renderer dependence",
        True: "the superadditive structure survives an independent rendering",
    }
    out["decision"] = decisions[out["stage1_admitted"]]

    for seed in SEEDS:
        print(f"seed {seed}")
        for name, entry in out["per_seed"][str(seed)].items():
            detail = {k: v for k, v in entry.items() if k != "pass"}
            print(f"  {name:26s} pass={entry['pass']}  {json.dumps(detail)}")
    print("stage1_admitted:", out["stage1_admitted"])
    print("decision:", out["decision"])
    if "stage2_per_seed" in out:
        for seed in SEEDS:
            entry = out["stage2_per_seed"][str(seed)]
            print(f"stage2 seed {seed}: E5={entry['P1_cliff_survives']['E5_pp']:+.2f} "
                  f"E2..4={json.dumps(entry['P2_low_severity_additive']['E_pp'])}")
        print("stage2 superadditive:", out["stage2_superadditive"])
        print("stage2 decision:", out["stage2_decision"])
    write_json(ROOT / "reports/physical/GATE.json", out)


if __name__ == "__main__":
    main()
