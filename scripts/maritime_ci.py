"""Cluster-bootstrap confidence intervals for the maritime five-arm table.

Resampling unit is ``source_group_id``: images that are duplicates of one
another stay together in every replicate, so a duplicated frame cannot count
as two independent observations.  All five arms are scored on the same
bootstrap replicates, which makes every difference a paired difference.

    python scripts/maritime_ci.py --seeds 17 29 --replicates 10000

Writes ``reports/maritime_main_v2_ci.json`` and ``reports/MARITIME_MAIN_V2_CI.md``.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/maritime_main_v2.json"
REPORT = ROOT / "reports"
VERSION = "maritime_main_v2"

ARMS = ["standard", "augmix", "strong_combo", "pair_coverage", "ncf"]
# The four contrasts the protocol asks for, in the order they are reported.
CONTRASTS = [
    ("ncf", "standard"),
    ("ncf", "augmix"),
    ("ncf", "strong_combo"),
    ("ncf", "pair_coverage"),
]
MAIN = ["standard", "augmix", "strong_combo", "pair_coverage", "ncf"]
GROUPS = {"clean": "clean", "single": "single", "weather_pair": "weather_pair",
          "held_pair": "held_pair"}


def conditions(config):
    fams = config["weather_families"]
    sev = config["severities"]
    weather_pair = [f"{'+'.join(p)}_{s}" for p in combinations(fams, 2) for s in sev]
    held_pair = [f"{'+'.join([fams[0], config['held_family']])}_{s}" for s in sev]
    singles = [f"{f}_{s}" for f in [*fams, config["held_family"]] for s in sev]
    return {
        "clean": ["clean"],
        "single": singles,
        "weather_pair": weather_pair,
        "held_pair": held_pair,
    }


def seed_dir(seed, versions):
    """Locate the report directory holding one seed's predictions."""
    for version in versions:
        candidate = REPORT / version / f"{ARMS[0]}_s{seed}" / "predictions.npz"
        if candidate.exists():
            return REPORT / version
    raise FileNotFoundError(f"no predictions for seed {seed} under {versions}")


def load(seed, needed, versions=("maritime_main_v2",)):
    out = {}
    directory = seed_dir(seed, versions)
    for arm in ARMS:
        out[arm] = np.load(directory / f"{arm}_s{seed}" / "predictions.npz",
                           allow_pickle=False)
    # Identity checks: same samples, same labels, same condition set, same shapes.
    ref = out[ARMS[0]]
    ids = ref["sample_ids"].astype(str)
    labels = ref["labels"]
    for arm in ARMS:
        d = out[arm]
        assert np.array_equal(d["sample_ids"].astype(str), ids), f"sample ids differ: {arm}"
        assert np.array_equal(d["labels"], labels), f"labels differ: {arm}"
        assert set(d.files) == set(ref.files), f"condition set differs: {arm}"
        for key in ref.files:
            if key in ("labels", "sample_ids"):
                continue
            assert d[key].shape == ref[key].shape, f"shape differs: {arm} {key}"
    missing = [k for k in needed if k not in ref.files]
    assert not missing, f"conditions absent from predictions: {missing[:5]}"
    return out, ids, labels


def group_index(ids, manifest):
    rows = {r["sample_id"]: r for r in
            (json.loads(line) for line in open(manifest, encoding="utf-8"))}
    groups = [rows[str(i)]["source_group_id"] or f"solo_{i}" for i in ids]
    uniq = sorted(set(groups))
    idx = {g: np.flatnonzero(np.array(groups) == g) for g in uniq}
    return uniq, idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[17, 29])
    ap.add_argument("--replicates", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=20261102)
    ap.add_argument("--out-dir", type=Path, default=REPORT)
    ap.add_argument("--versions", nargs="+", default=[VERSION],
                    help="report directories under reports/ searched for each seed")
    args = ap.parse_args()

    config = json.loads(CONFIG.read_text())
    conds = conditions(config)
    needed = [k for group in conds.values() for k in group]

    per_seed, checks = {}, {}
    for seed in args.seeds:
        data, ids, labels = load(seed, needed, args.versions)
        uniq, idx = group_index(ids, ROOT / config["manifest"])
        sizes = np.array([len(idx[g]) for g in uniq])
        checks[str(seed)] = {
            "n_samples": int(len(ids)),
            "n_source_groups": int(len(uniq)),
            "group_size_histogram": {str(k): int(v) for k, v in
                                     zip(*np.unique(sizes, return_counts=True))},
            "n_conditions": int(len([f for f in data[ARMS[0]].files
                                     if f not in ("labels", "sample_ids")])),
            "identical_across_arms": "sample_ids, labels, condition keys and array shapes",
            "degradation_renderer": "deterministic in (sample_id, condition); "
                                    "no arm or seed enters the renderer",
        }
        rng = np.random.default_rng(args.bootstrap_seed + seed)
        draws = rng.integers(0, len(uniq), size=(args.replicates, len(uniq)))  # (R, G)
        denom = sizes[draws].sum(axis=1).astype(np.float64)

        def accuracy(arm, group):
            cols = []
            for key in conds[group]:
                ok = (data[arm][key] == labels).astype(np.float64)
                cluster_sum = np.array([ok[idx[g]].sum() for g in uniq])
                cols.append(cluster_sum[draws].sum(axis=1) / denom)
            return np.mean(cols, axis=0)

        rows = {"arms": {}, "contrasts": {}, "point": {}}
        point = {"arms": {}, "contrasts": {}}
        for arm in MAIN:
            rows["arms"][arm], point["arms"][arm] = {}, {}
            for group in GROUPS:
                vec = accuracy(arm, group)
                rows["arms"][arm][group] = {
                    "mean": float(vec.mean()),
                    "lo": float(np.percentile(vec, 2.5)),
                    "hi": float(np.percentile(vec, 97.5)),
                }
                point["arms"][arm][group] = float(np.mean(
                    [(data[arm][k] == labels).mean() for k in conds[group]]))
        for a, b in CONTRASTS:
            key = f"{a}_minus_{b}"
            rows["contrasts"][key], point["contrasts"][key] = {}, {}
            for group in ("single", "weather_pair", "held_pair"):
                diff = accuracy(a, group) - accuracy(b, group)
                rows["contrasts"][key][group] = {
                    "delta": float(diff.mean()),
                    "lo": float(np.percentile(diff, 2.5)),
                    "hi": float(np.percentile(diff, 97.5)),
                    "p_positive": float((diff > 0).mean()),
                }
                point["contrasts"][key][group] = float(
                    np.mean([(data[a][k] == labels).mean() for k in conds[group]])
                    - np.mean([(data[b][k] == labels).mean() for k in conds[group]]))
        rows["point"] = point
        per_seed[str(seed)] = rows

    across = {}
    for arm in MAIN:
        across[arm] = {group: _range([per_seed[str(s)]["point"]["arms"][arm][group]
                                      for s in args.seeds]) for group in GROUPS}
    for a, b in CONTRASTS:
        key = f"{a}_minus_{b}"
        across[key] = {group: _range([per_seed[str(s)]["point"]["contrasts"][key][group]
                                      for s in args.seeds])
                       for group in ("single", "weather_pair", "held_pair")}

    payload = {"replicates": args.replicates, "bootstrap_seed": args.bootstrap_seed,
               "cluster": "source_group_id (duplicate frames kept together)",
               "ci": "percentile 2.5-97.5 over cluster replicates",
               "versions": args.versions, "seeds": args.seeds, "checks": checks, "per_seed": per_seed,
               "across_seeds": across}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{VERSION}_ci.json").write_text(json.dumps(payload, indent=1))
    write_markdown(payload, args.out_dir / "MARITIME_MAIN_V2_CI.md")
    for seed in args.seeds:
        r = per_seed[str(seed)]["contrasts"]
        print(f"seed {seed}: " + ", ".join(
            f"{k} weather_pair {100 * v['weather_pair']['delta']:+.2f} pp "
            f"[{100 * v['weather_pair']['lo']:+.2f},{100 * v['weather_pair']['hi']:+.2f}]"
            for k, v in r.items()))


def _range(vals):
    return {"mean": float(np.mean(vals)), "seed_min": float(min(vals)),
            "seed_max": float(max(vals))}


def write_markdown(payload, path):
    lines = ["# 海事五臂：按 source_group_id 聚类的 bootstrap 置信区间", "",
             f"重采样单位为 `source_group_id`：同一原图的重复帧整体重采样，"
             f"因此重复图像不会当独立样本。重复 {payload['replicates']} 次，"
             f"百分位区间 2.5–97.5。seeds：{', '.join(str(s) for s in payload['seeds'])}。", "",
             "表中数值单位为百分点；评测图像、标签、条件集合与数组形状跨臂逐项校验一致；退化渲染器只依赖 "
             "`(sample_id, condition)`，不依赖臂或 seed。", ""]
    for seed in payload["seeds"]:
        s = payload["per_seed"][str(seed)]
        c = payload["checks"][str(seed)]
        lines += [f"## seed {seed}", "",
                  f"{c['n_samples']} 张评测图，{c['n_source_groups']} 个 source group，"
                  f"{c['n_conditions']} 个条件；组大小分布 {c['group_size_histogram']}。", "",
                  "| 臂 | Clean | 单项 | 天气复合 | 池外族对 haze+defocus |",
                  "|---|---:|---:|---:|---:|"]
        for arm in MAIN:
            r, pt = s["arms"][arm], s["point"]["arms"][arm]
            lines.append(
                f"| {arm} | {_cell(r['clean'], pt['clean'])} "
                f"| {_cell(r['single'], pt['single'])} "
                f"| {_cell(r['weather_pair'], pt['weather_pair'])} "
                f"| {_cell(r['held_pair'], pt['held_pair'])} |")
        lines += ["", "配对差值（同一重采样内相减，正数表示前者更好）：", "",
                  "| 对比 | 单项 | 天气复合 | 池外族对 |", "|---|---:|---:|---:|"]
        for key, row in s["contrasts"].items():
            lines.append(f"| {key.replace('_minus_', ' − ')} | {_cell(row['single'])} "
                         f"| {_cell(row['weather_pair'])} | {_cell(row['held_pair'])} |")
        lines.append("")
    lines += ["## 跨 seed", "",
              f"{len(payload['seeds'])} 个 seed：下面是 seed 间范围，不是置信区间；"
              "不得把 seed 当独立样本扩大样本量。", "",
              "| 对比 | 天气复合均值 | seed 间范围 |", "|---|---:|---|"]
    for key, row in payload["across_seeds"].items():
        if "_minus_" not in key:
            continue
        r = row["weather_pair"]
        lines.append(f"| {key.replace('_minus_', ' − ')} | {100 * r['mean']:+.2f} "
                     f"| {100 * r['seed_min']:+.2f} … {100 * r['seed_max']:+.2f} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _cell(r, point=None):
    signed = r.get("delta") is not None
    f = "{:+.2f}".format if signed else "{:.2f}".format
    scale = 100.0
    centre = r.get("delta") if signed else (point if point is not None else r.get("mean"))
    body = f"{f(scale * centre)} [{f(scale * r['lo'])}, {f(scale * r['hi'])}]"
    if "p_positive" in r:
        body += f" ({r['p_positive'] * 100:.0f}% 正)"
    return body


if __name__ == "__main__":
    main()
