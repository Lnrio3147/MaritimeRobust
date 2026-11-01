"""Held-out-pair summary for the CIFAR-10-C coverage arms (pre-registration B1-B3).

Reads the per-condition rows that ``cifar_aggregate.py`` writes to
``reports/cifar_grid.json`` and reports, per arm, the mean ``delta`` on the five
pairs that no training pool contains, plus the contrasts that B1 and B2 freeze:

    B1  delta on held-out pairs falls monotonically with pool width k
    B2  the paired arm beats the singles arm by less than -1 pp
    B3  no arm loses more than 1.5 pp of clean test accuracy

No GPU, no model: this runs off the stored per-image flags.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT / "src"))
from maritime_robust.external.cifar_c import HELD_OUT  # noqa: E402


def canon(pair):
    return frozenset(pair)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="reports/cifar_grid.json")
    args = ap.parse_args()
    payload = json.loads((PROJECT / args.json).read_text())
    held = {canon(p) for p in HELD_OUT}

    lines = ["# CIFAR-10-C coverage arms, held-out pairs", ""]
    summary = {}
    for arm, entry in payload.items():
        rows = entry.get("rows") or []
        if not rows:
            continue
        ho = [r for r in rows if canon((r["a"], r["b"])) in held]
        pooled = [r for r in rows if r["severity"] == 3]
        ho3 = [r for r in ho if r["severity"] == 3]
        summary[arm] = {
            "heldout_delta_all": float(np.mean([r["delta_pp"] for r in ho])),
            "heldout_delta_s3": float(np.mean([r["delta_pp"] for r in ho3])),
            "heldout_super_s3": float(np.mean([r["super_pp"] for r in ho3])),
            "pooled_delta_all": float(np.mean([r["delta_pp"] for r in pooled])),
            "conditions": len(rows),
        }
        lines += [f"## {arm}", "",
                  f"- held-out pairs, all severities: delta {summary[arm]['heldout_delta_all']:+.2f} pp",
                  f"- held-out pairs, s3: delta {summary[arm]['heldout_delta_s3']:+.2f} pp "
                  f"(super {summary[arm]['heldout_super_s3']:+.2f})",
                  f"- all pairs, s3: delta {summary[arm]['pooled_delta_all']:+.2f} pp"]
    lines += ["", "## Frozen contrasts", ""]

    def seed_of(name):
        m = re.search(r"_s(\d+)$", name)
        return m.group(1) if m else ""

    clean = {seed_of(a): a for a in summary if a.startswith("clean")}
    for width in ("lattice_k15", "lattice_k40"):
        for seed, arm in sorted((seed_of(a), a) for a in summary if a.startswith(width)):
            if seed in clean:
                d = summary[arm]["heldout_delta_s3"] - summary[clean[seed]]["heldout_delta_s3"]
                lines.append(f"- {arm} minus {clean[seed]} (held-out, s3): {d:+.2f} pp")

    # B2: the paired-minus-singles contrast, computed per seed where both arms exist.
    paired = {seed_of(a): a for a in summary if a.startswith("lattice_k15")}
    single = {seed_of(a): a for a in summary if a.startswith("singles")}
    b2 = {}
    for seed in sorted(set(paired) & set(single)):
        d = summary[paired[seed]]["heldout_delta_s3"] - summary[single[seed]]["heldout_delta_s3"]
        b2[seed] = d
        lines.append(f"- B2 paired-minus-singles, seed {seed} (held-out, s3): {d:+.2f} pp "
                     f"(threshold: < -1 pp)")
    if len(b2) > 1:
        values = np.array(list(b2.values()))
        lines.append(f"- B2 across {len(b2)} seeds: mean {values.mean():+.2f} pp, "
                     f"range {values.min():+.2f} … {values.max():+.2f} pp")
    summary["_b2_per_seed"] = b2
    (PROJECT / "reports/CIFAR_HELDOUT.md").write_text("\n".join(lines) + "\n")
    (PROJECT / "reports/cifar_heldout.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
