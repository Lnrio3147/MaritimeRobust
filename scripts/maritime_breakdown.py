"""Per-family, per-combination and per-severity breakdown of the maritime arms.

Reads only saved ``predictions.npz`` files, so it can be rerun whenever a seed
finishes.  Every cell is the mean over the seeds given on the command line; the
seed range is printed next to it so two seeds are never mistaken for a sample.

    python scripts/maritime_breakdown.py --seeds 17 29 43 59

Writes ``reports/maritime_main_v2_breakdown.json`` and ``MARITIME_MAIN_V2_BREAKDOWN.md``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports"
ARMS = ["standard", "augmix", "strong_combo", "pair_coverage", "ncf"]
VERSIONS = ["maritime_main_v2", "maritime_main_v2_b"]


def seed_dir(seed):
    for version in VERSIONS:
        if (REPORT / version / f"standard_s{seed}" / "predictions.npz").exists():
            return REPORT / version
    raise FileNotFoundError(seed)


def load(seed):
    directory = seed_dir(seed)
    return {arm: np.load(directory / f"{arm}_s{seed}" / "predictions.npz",
                         allow_pickle=False) for arm in ARMS}


def keys_by_structure(files):
    singles, pairs = {}, {}
    for key in files:
        if key in ("labels", "sample_ids", "clean"):
            continue
        name, severity = key.rsplit("_", 1)
        severity = int(severity)
        if "+" in name:
            pairs.setdefault(tuple(name.split("+")), {})[severity] = key
        else:
            singles.setdefault(name, {})[severity] = key
    return singles, pairs


def cell(values):
    return {"mean": float(np.mean(values)), "min": float(np.min(values)),
            "max": float(np.max(values))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43, 59])
    ap.add_argument("--config", type=Path, default=ROOT / "configs/maritime_main_v2.json")
    args = ap.parse_args()

    config = json.loads(args.config.read_text())
    families = config["weather_families"]
    severities = config["severities"]
    held = config["held_family"]
    from itertools import combinations
    weather_pairs = list(combinations(families, 2))

    data = {seed: load(seed) for seed in args.seeds}
    ref = data[args.seeds[0]]["standard"]
    singles, pairs = keys_by_structure(ref.files)

    out = {"seeds": args.seeds, "arms": {}}
    for arm in ARMS:
        acc = {}
        for seed in args.seeds:
            d = data[seed][arm]
            y = d["labels"]
            acc[seed] = {
                "clean": float((d["clean"] == y).mean()),
                "single_all": float(np.mean([(d[k] == y).mean() for f in singles for k in singles[f].values()])),
                "weather_pair": float(np.mean([(d[pairs[p][s]] == y).mean()
                                               for p in weather_pairs for s in severities])),
                "severity": {s: float(np.mean([(d[pairs[p][s]] == y).mean()
                                               for p in weather_pairs])) for s in severities},
                "family_single": {f: float(np.mean([(d[singles[f][s]] == y).mean()
                                                    for s in severities])) for f in singles},
                "pair": {f"{'+'.join(p)}": float(np.mean([(d[pairs[p][s]] == y).mean()
                                                          for s in severities]))
                         for p in sorted(pairs)},
                "family_by_severity": {
                    **{f: {s: float((d[singles[f][s]] == y).mean()) for s in severities}
                       for f in singles},
                    **{f"{'+'.join(p)}": {s: float((d[pairs[p][s]] == y).mean()) for s in severities}
                       for p in sorted(pairs)},
                },
            }
        out["arms"][arm] = acc

    # Aggregate across seeds.
    agg = {"arms": {}}
    for arm in ARMS:
        acc = out["arms"][arm]
        rows = {}
        for key in ("clean", "single_all", "weather_pair"):
            rows[key] = cell([acc[s][key] for s in args.seeds])
        rows["severity"] = {str(sv): cell([acc[s]["severity"][sv] for s in args.seeds])
                            for sv in severities}
        rows["family_single"] = {f: cell([acc[s]["family_single"][f] for s in args.seeds])
                                 for f in families + [held]}
        rows["pair"] = {p: cell([acc[s]["pair"][p] for s in args.seeds]) for p in acc[args.seeds[0]]["pair"]}
        agg["arms"][arm] = rows
    agg["contrasts"] = {}
    for other in ("standard", "augmix", "strong_combo", "pair_coverage"):
        rows = {"severity": {}, "pair": {}}
        for sv in severities:
            vals = [out["arms"]["ncf"][s]["severity"][sv] - out["arms"][other][s]["severity"][sv]
                    for s in args.seeds]
            rows["severity"][str(sv)] = cell(vals)
        for p in out["arms"]["ncf"][args.seeds[0]]["pair"]:
            vals = [out["arms"]["ncf"][s]["pair"][p] - out["arms"][other][s]["pair"][p]
                    for s in args.seeds]
            rows["pair"][p] = cell(vals)
        vals = [out["arms"]["ncf"][s]["weather_pair"] - out["arms"][other][s]["weather_pair"]
                for s in args.seeds]
        rows["weather_pair"] = cell(vals)
        agg["contrasts"][f"ncf_minus_{other}"] = rows

    agg["per_seed"] = out["arms"]
    agg["seeds"] = args.seeds
    (REPORT / "maritime_main_v2_breakdown.json").write_text(json.dumps(agg, indent=1))
    write_markdown(agg, out, REPORT / "MARITIME_MAIN_V2_BREAKDOWN.md", severities, families, held)
    print(json.dumps({a: agg["arms"][a]["weather_pair"] for a in ARMS}, indent=1))


def write_markdown(agg, raw, path, severities, families, held):
    def f(x):
        return f"{100 * x:.2f}"

    lines = ["# 海事五臂：按退化族、组合与严重度分解", "",
             f"seeds：{', '.join(str(s) for s in raw['seeds'])}。单元格为 seed 均值（百分点），"
             "括号内为 seed 间范围，不是置信区间。", "",
             "## 总表", "",
             "| 臂 | Clean | 单项（20 条件） | 天气复合（15 条件） |",
             "|---|---:|---:|---:|"]
    for arm in ARMS:
        r = agg["arms"][arm]
        lines.append(
            f"| {arm} "
            f"| {f(r['clean']['mean'])} ({f(r['clean']['min'])}–{f(r['clean']['max'])}) "
            f"| {f(r['single_all']['mean'])} ({f(r['single_all']['min'])}–{f(r['single_all']['max'])}) "
            f"| {f(r['weather_pair']['mean'])} ({f(r['weather_pair']['min'])}–{f(r['weather_pair']['max'])}) |")
    lines += ["", "## 按严重度（15 个天气配对上的均值）", "",
              "| 臂 | " + " | ".join(f"sev {s}" for s in severities) + " |",
              "|---|" + "---:|" * len(severities)]
    for arm in ARMS:
        r = agg["arms"][arm]["severity"]
        lines.append(f"| {arm} | " + " | ".join(f(r[str(s)]["mean"]) for s in severities) + " |")
    lines += ["", "## 按单项退化族（5 个严重度均值）", "",
              "| 臂 | " + " | ".join(families + [held]) + " |",
              "|---|" + "---:|" * (len(families) + 1)]
    for arm in ARMS:
        r = agg["arms"][arm]["family_single"]
        lines.append(f"| {arm} | " + " | ".join(f(r[k]["mean"]) for k in families + [held]) + " |")
    lines += ["", "## 按组合（5 个严重度均值）", "",
              "| 臂 | " + " | ".join(agg["arms"][ARMS[0]]["pair"].keys()) + " |",
              "|---|" + "---:|" * len(agg["arms"][ARMS[0]]["pair"])]
    for arm in ARMS:
        r = agg["arms"][arm]["pair"]
        lines.append(f"| {arm} | " + " | ".join(f(r[k]["mean"]) for k in r) + " |")
    lines += ["", "## NCF 相对各臂的差值（正数表示 NCF 更高）", "",
              "| 对比 | 天气复合 | " + " | ".join(f"sev {s}" for s in severities) + " |",
              "|---|---:|" + "---:|" * len(severities)]
    for key, r in agg["contrasts"].items():
        lines.append(f"| {key.replace('_minus_', ' − ')} | {f(r['weather_pair']['mean'])} "
                     f"({f(r['weather_pair']['min'])}–{f(r['weather_pair']['max'])}) | "
                     + " | ".join(f(r["severity"][str(s)]["mean"]) for s in severities) + " |")
    lines += ["", "## 逐组合差值（NCF − 对照，5 个严重度均值）", "",
              "| 对比 | " + " | ".join(agg["arms"][ARMS[0]]["pair"].keys()) + " |",
              "|---|" + "---:|" * len(agg["arms"][ARMS[0]]["pair"])]
    for key, r in agg["contrasts"].items():
        lines.append(f"| {key.replace('_minus_', ' − ')} | "
                     + " | ".join(f(r["pair"][p]["mean"]) for p in r["pair"]) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
