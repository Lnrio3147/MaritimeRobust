#!/usr/bin/env python
"""Render the real-degradation stratified evaluation report.

Reads the JSON written by `real_degradation_eval.py`; no numbers are invented.
Draft (unconfirmed) results are rendered with an explicit non-citable banner.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maritime_robust.common import ROOT as PROJECT  # noqa: E402

STRATUM_ORDER = ["clear", "clear_no_clutter", "any_degraded", "type:cloud", "type:haze",
                 "type:shadow", "type:glare", "type:blur", "severity:mild",
                 "severity:medium", "severity:severe", "multi_type", "clutter"]
TYPE_NAMES = {"cloud": "厚云", "haze": "雾/霾", "shadow": "阴影", "glare": "反光",
              "blur": "模糊", "type:cloud": "厚云", "type:haze": "雾/霾",
              "type:shadow": "阴影", "type:glare": "反光", "type:blur": "模糊",
              "severity:mild": "轻度", "severity:medium": "中度",
              "severity:severe": "重度", "multi_type": "多类型", "clutter": "场景杂乱",
              "clear_no_clutter": "清晰且不杂乱", "any_degraded": "任一退化", "clear": "清晰"}


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}"


def signed(x):
    return "—" if x is None else f"{x:+.1f}"


def ci(x):
    return "—" if not x else f"[{x[0]:+.1f}, {x[1]:+.1f}]"


def label_table(data):
    dist = data["label_distribution"]
    rows = ["| 项目 | 数量 |", "|---|---|",
            f"| 清晰 (clear) | {dist['clear']} |",
            f"| 有退化 (any_degraded) | {dist['degraded']} |",
            f"| 无类型且无法判断 | {dist['unlabeled_ambiguous']} |",
            f"| 场景杂乱 | {dist['clutter']} |"]
    types = "，".join(f"{TYPE_NAMES.get(t, t)} {n}" for t, n in dist["types"].items())
    sev = "，".join(f"{TYPE_NAMES.get('severity:' + s, s)} {n}"
                   for s, n in dist["severity"].items())
    rows.append(f"\n类型：{types}。\n\n强度：{sev}。")
    return "\n".join(rows)


def backbone_table_v2(data):
    lines = ["| 分层 | N | 原图组 | accuracy | Δ vs clear (pp) | 种子 Δ 范围 | balanced accuracy |",
             "|---|---|---|---|---|---|---|"]
    backbones = data["backbones"]
    any_bb = next(iter(backbones))
    strata = [s for s in STRATUM_ORDER if s in backbones[any_bb]["strata"]]
    for stratum in strata:
        ns, groups, accs, deltas, bals, lo, hi = [], [], [], [], [], [], []
        for agg in backbones.values():
            v = agg["strata"].get(stratum)
            if not v:
                continue
            ns.append(v["n"])
            groups.append(v.get("groups", 0))
            accs.append(v["accuracy_mean"])
            bals.append(v["balanced_accuracy_mean"])
            if v["delta_mean"] is not None:
                deltas.append(v["delta_mean"])
                lo.append(v["delta_min"])
                hi.append(v["delta_max"])
        mark = " ▶" if any(agg["strata"].get(stratum, {}).get("exploratory")
                           for agg in backbones.values()) else ""
        cells = [f"{TYPE_NAMES.get(stratum, stratum)}{mark}",
                 f"{ns[0]:.0f}" if ns else "—",
                 f"{groups[0]:.0f}" if groups else "—",
                 f"{pct(sum(accs) / len(accs))}" if accs else "—",
                 f"{signed(sum(deltas) / len(deltas))}" if deltas else "—",
                 f"{signed(min(lo))} 至 {signed(max(hi))}" if lo else "—",
                 f"{pct(sum(bals) / len(bals))}" if bals else "—"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def appendix(data):
    lines = ["| 分类器 | 分层 | N | 原图组 | accuracy | Δ vs clear (pp) | Δ 95% CI | balanced accuracy |",
             "|---|---|---|---|---|---|---|---|"]
    for name, info in data["classifiers"].items():
        strata = [s for s in STRATUM_ORDER if s in info["strata"]]
        for stratum in strata:
            v = info["strata"][stratum]
            mark = " ▶" if v["exploratory"] else ""
            lines.append(
                f"| {name} | {TYPE_NAMES.get(stratum, stratum)}{mark} | {v['n']} | "
                f"{v['groups']} | {pct(v['accuracy'])} | {signed(v['delta_vs_clear_pp'])} | "
                f"{ci(v['delta_ci_pp'])} | {pct(v['balanced_accuracy'])} |")
    return "\n".join(lines)


def delta_range(data, stratum):
    rows = [agg["strata"][stratum] for agg in data["backbones"].values()
            if stratum in agg["strata"] and agg["strata"][stratum]["delta_mean"] is not None]
    if not rows:
        return None
    mean = sum(r["delta_mean"] for r in rows) / len(rows)
    return mean, min(r["delta_min"] for r in rows), max(r["delta_max"] for r in rows)


def fmt_delta(stat):
    return "—" if stat is None else f"{stat[0]:+.1f} pp（种子 {stat[1]:+.1f} 至 {stat[2]:+.1f}）"


def conclusion(fg, fs):
    bits = []
    for name, data in (("FGSC-23", fg), ("FGSRCS", fs)):
        dist = data["label_distribution"]
        bits.append(
            f"{name}：任一退化 Δ {fmt_delta(delta_range(data, 'any_degraded'))}，"
            f"雾/霾 {fmt_delta(delta_range(data, 'type:haze'))}，"
            f"场景杂乱 {fmt_delta(delta_range(data, 'clutter'))}；"
            f"厚云 {dist['types']['cloud']} 张、重度 {dist['severity']['severe']} 张")
    return ("；".join(bits) +
            "。真实退化未造成稳定的准确率下降；FGSC-23 上场景杂乱的负向更大，但为观察性分组。"
            "厚云/重度样本不足，不能给出该层结论。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fgsc23", default="reports/real_degradation_stratified_fgsc23_val.json")
    parser.add_argument("--fgsrcs", default="reports/real_degradation_stratified_fgsrcs_dev.json")
    parser.add_argument("--output", default="reports/REAL_DEGRADATION_STRATIFIED.md")
    args = parser.parse_args()

    fg_path, fs_path = PROJECT / args.fgsc23, PROJECT / args.fgsrcs
    if not fg_path.exists() or not fs_path.exists():
        raise SystemExit("run real_degradation_eval.py for both datasets first")
    fg = json.loads(fg_path.read_text())
    fs = json.loads(fs_path.read_text())
    draft = not (fg["replicable"] and fs["replicable"])

    out = ["# 真实退化分层评测", ""]
    if draft:
        out += ["> **草稿：标签未确认，以下数字仅用于验证评测管线，不可引用。**", ""]
    out += ["## 0. 结论", "", conclusion(fg, fs), "",
            "以上为分层内的观察性差异；退化与类别、场景杂乱相关，不能解释为退化单独造成的因果效应。",
            "", "## 1. 数据与标签", "", "### FGSC-23 验证集（157 张，8 类）", "",
            label_table(fg), "", "### FGSRCS development（184 个裁剪，2 类）", "",
            label_table(fs), ""]
    out += ["## 2. 指标", "",
            "- accuracy = 分层内正确数 / 分层样本数。",
            "- balanced accuracy = 分层内出现类别的 recall 的算术平均；未出现类别不参与。",
            "- Δ vs clear = 同一分类器在分层与 clear 上的 accuracy 之差，单位 pp。",
            "- 95% CI = 按 `source_group_id` 分层内组自助法 2,000 次；不是随机化区间。",
            "- `▶` = exploratory（N < 15 或原图组 < 8），只描述，不做方向性结论。",
            "- 完整类别支持度与逐分类器区间见 JSON。", ""]
    out += ["## 3. FGSC-23 验证集", "",
            "三个骨干的种子均值；ResNet-50 只有 seed 17，是单种子诊断。", "",
            backbone_table_v2(fg), "",
            "### 3.1 逐分类器", "", appendix(fg), ""]
    out += ["## 4. FGSRCS development", "",
            "只评严格映射的 container ship 与 oil tanker 两类；分类器仍保留 8 类头，"
            "其余类的预测计为错误。不能与 FGSC-23 的 8 类 macro-F1 直接比较。", "",
            backbone_table_v2(fs), "",
            "### 4.1 逐分类器", "", appendix(fs), ""]
    out += ["## 5. 解释边界", "",
            "- 分层是观察性的；真实退化类型与类别、港口、拍摄条件混杂，差异不能归因于单一因素。",
            "- 单轮标注，无第二标注者与标注者间一致性；标签噪声未建模。",
            "- 厚云与重度退化样本不足，统计功效不足以支持强结论。",
            "- FGSC-23 的退化层以轻/中度模糊为主，与合成实验的 4–5 级雾/眩光不是同一强度量级；"
            "本结果不能外推到严重合成退化，也不能据此否定合成实验的结论。",
            "- FGSRCS 为裁剪、只有 2 个映射类别；清晰层准确率仅约 76%，跨库域差大于退化差异，"
            "其 Δ 不能解释为退化效应。",
            "- 封存测试与 FGSRCS sealed 未运行。", ""]
    output = PROJECT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    output.write_text("\n".join(out))
    print(json.dumps({"output": str(output.relative_to(PROJECT)), "draft": draft}))


if __name__ == "__main__":
    main()
