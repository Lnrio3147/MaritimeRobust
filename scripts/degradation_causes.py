#!/usr/bin/env python
"""Decompose the degradation accuracy drop into candidate mechanisms.

Reads only saved per-sample predictions from the combined-augmentation
baselines (the comparison anchor for any new module). Five views:

  1. flips: clean-correct samples that turn wrong under degradation, versus
     samples that were already wrong on clean input;
  2. cross-model consensus: if all three architectures fail the same sample,
     the information is likely gone; if one still succeeds, there is headroom;
  3. correlation with physical degradation parameters (transmission, cloud
     area, saturated fraction), pooled and within severity;
  4. confidence of errors: wrong predictions that are still >= 0.9 confident;
  5. class-level losses and where those samples go.

Writes reports/degradation_causes.json and reports/DEGRADATION_CAUSES.md.
"""
import json

import numpy as np

from maritime_robust.common import ROOT, read_jsonl, write_json

SCOPE = "civilian"
RUNS = {
    "resnet50": f"runs/fgsc23_research_{SCOPE}/resnet50_combo_s17",
    "convnext_tiny": f"runs/fgsc23_research_{SCOPE}/convnext_tiny_combo_s17",
    "swin_t": f"runs/fgsc23_research_{SCOPE}/swin_t_combo_s17",
}
DISPLAY = {"resnet50": "ResNet-50", "convnext_tiny": "ConvNeXt-Tiny",
           "swin_t": "Swin-Tiny"}
KEY = ["clean", "matched_haze_5", "matched_cloud_5", "matched_glare_5",
       "matched_haze+glare_5", "matched_haze+cloud_5"]
PARAMS = {
    "haze": ["transmission_mean", "center_proxy_affected_fraction"],
    "cloud": ["cloud_area_fraction", "affected_fraction",
              "center_proxy_affected_fraction"],
    "glare": ["saturated_fraction", "affected_fraction",
              "center_proxy_affected_fraction"],
    "haze+glare": ["transmission_mean", "saturated_fraction",
                   "center_proxy_affected_fraction"],
    "haze+cloud": ["transmission_mean", "cloud_area_fraction",
                   "center_proxy_affected_fraction"],
}
KINDS = list(PARAMS)
HIGH_CONF = 0.9


def load(run):
    run = ROOT / run
    summary = json.loads((run / "evaluation/summary.json").read_text())
    preds = {c: read_jsonl(run / "evaluation" / f"{c}.jsonl") for c in summary}
    config = json.loads((run / "config.json").read_text())
    return {"summary": summary, "preds": preds,
            "classes": {int(k): v for k, v in config["classes"].items()}}


def accuracy(rows):
    return float(np.mean([r["prediction"] == r["label"] for r in rows]))


def flips(preds, cond):
    clean = {r["sample_id"]: r for r in preds["clean"]}
    clean_ok = [r for r in preds[cond]
                if clean[r["sample_id"]]["prediction"] == clean[r["sample_id"]]["label"]]
    if not clean_ok:
        return {"clean_correct": 0, "flipped": 0, "flip_rate": None}
    flipped = sum(r["prediction"] != r["label"] for r in clean_ok)
    return {"clean_correct": len(clean_ok), "flipped": flipped,
            "flip_rate": flipped / len(clean_ok)}


def max_prob(row):
    p = row.get("probabilities")
    return float(max(p)) if p else None


def confidence(preds, cond):
    rows = preds[cond]
    good = [max_prob(r) for r in rows if r["prediction"] == r["label"]]
    bad = [max_prob(r) for r in rows if r["prediction"] != r["label"]]
    bad = [b for b in bad if b is not None]
    return {"correct_mean_maxprob": float(np.mean(good)) if good else None,
            "error_mean_maxprob": float(np.mean(bad)) if bad else None,
            "errors": len(bad),
            "overconfident_errors": sum(b >= HIGH_CONF for b in bad),
            "overconfident_share": (sum(b >= HIGH_CONF for b in bad) / len(bad))
                                   if bad else None}


def consensus(preds_by_model, cond):
    maps = {}
    for m, p in preds_by_model.items():
        maps[m] = {r["sample_id"]: r for r in p[cond]}
    ids = sorted(set.intersection(*[set(v) for v in maps.values()]))
    counts, correct = [], {m: [] for m in maps}
    for i in ids:
        row = [maps[m][i]["prediction"] == maps[m][i]["label"] for m in maps]
        counts.append(sum(row))
        for m, ok in zip(maps, row):
            correct[m].append(ok)
    counts = np.array(counts)
    per_model = {}
    for m in maps:
        wrong = np.array([not ok for ok in correct[m]])
        per_model[m] = {
            "errors": int(wrong.sum()),
            "errors_universal": int(np.sum(wrong & (counts == 0))),
            "share_errors_universal": (float(np.sum(wrong & (counts == 0)) / wrong.sum())
                                       if wrong.sum() else None),
        }
    return {"n": len(ids),
            "share_all_wrong": float(np.mean(counts == 0)),
            "share_some_correct": float(np.mean(counts > 0)),
            "share_all_correct": float(np.mean(counts == len(maps))),
            "wrong_by_count": {int(k): int(np.sum(counts == k))
                               for k in range(len(maps) + 1)},
            "per_model": per_model}


def ranks(x):
    """Average ranks so tied values (including constant arrays) get equal rank."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    out = np.empty(len(x), float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        out[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(x, y):
    rx, ry = ranks(np.asarray(x, float)), ranks(np.asarray(y, float))
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def parameter_analysis(data, kind):
    """Pool matched severities; also compute within-severity rank correlation."""
    out = {"params": {}, "severity": {}}
    for model, entry in data.items():
        preds = entry["preds"]
        conds = [c for c in preds if c.startswith("matched_") and c[8:].rsplit("_", 1)[0] == kind]
        for param in PARAMS[kind]:
            xs, ys, within = [], [], []
            for c in sorted(conds):
                rows = preds[c]
                xs += [r["degradation"].get(param) for r in rows]
                ys += [int(r["prediction"] == r["label"]) for r in rows]
                vals = [(r["degradation"].get(param), int(r["prediction"] == r["label"]))
                        for r in rows]
                vals = [(v, y) for v, y in vals if v is not None]
                if len(vals) > 10:
                    within.append(spearman([v for v, _ in vals], [y for _, y in vals]))
            vals = [(x, y) for x, y in zip(xs, ys) if x is not None]
            if len(vals) < 10:
                continue
            pooled = spearman([v for v, _ in vals], [y for _, y in vals])
            within = [w for w in within if w is not None]
            # severity-5 tertiles
            severe = [r for r in preds.get(f"matched_{kind}_5", [])
                      if r["degradation"].get(param) is not None]
            tert = None
            spread = [r["degradation"][param] for r in severe]
            if len(severe) >= 30 and max(spread) > min(spread):
                vals5 = sorted(severe, key=lambda r: r["degradation"][param])
                third = len(vals5) // 3
                low, high = vals5[:third], vals5[-third:]
                tert = {"low_acc": accuracy(low), "high_acc": accuracy(high),
                        "low_range": [vals5[0]["degradation"][param],
                                      low[-1]["degradation"][param]],
                        "high_range": [high[0]["degradation"][param],
                                       vals5[-1]["degradation"][param]]}
            out["params"].setdefault(param, {})[model] = {
                "pooled_rho": pooled,
                "within_severity_rho": float(np.mean(within)) if within else None,
                "severity5_tertiles": tert,
            }
        # per-severity accuracy for context
        for c in sorted(conds):
            out["severity"].setdefault(c, {})[model] = accuracy(preds[c])
    return out


def class_table(data, model="swin_t"):
    entry = data[model]
    preds = entry["preds"]
    names = entry["classes"]
    labels = sorted(names)
    rows = []
    clean = {r["sample_id"]: r for r in preds["clean"]}
    for lab in labels:
        idx = [r["sample_id"] for r in preds["clean"] if r["label"] == lab]
        if not idx:
            continue
        row = {"class": names[lab], "n": len(idx),
               "clean": np.mean([clean[i]["prediction"] == lab for i in idx])}
        for c in ("matched_glare_5", "matched_haze+glare_5", "matched_haze+cloud_5"):
            rows_c = {r["sample_id"]: r for r in preds[c]}
            row[c] = np.mean([rows_c[i]["prediction"] == lab for i in idx])
            row[c + "_dest"] = {}
            for i in idx:
                if rows_c[i]["prediction"] != lab:
                    d = names[rows_c[i]["prediction"]]
                    row[c + "_dest"][d] = row[c + "_dest"].get(d, 0) + 1
        rows.append(row)
    return rows


def pct(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}"


def main():
    data = {m: load(r) for m, r in RUNS.items()}
    out = {"scope": SCOPE, "models": list(data), "key": {}, "consensus": {},
           "parameters": {}, "classes": None}

    for m, entry in data.items():
        out["key"][m] = {}
        for c in KEY:
            if c not in entry["preds"]:
                continue
            out["key"][m][c] = {
                "accuracy": accuracy(entry["preds"][c]),
                "flips": flips(entry["preds"], c),
                "confidence": confidence(entry["preds"], c),
            }

    for c in KEY:
        if all(c in e["preds"] for e in data.values()):
            out["consensus"][c] = consensus({m: e["preds"] for m, e in data.items()},
                                            c)

    for kind in KINDS:
        out["parameters"][kind] = parameter_analysis(data, kind)

    out["classes"] = class_table(data)

    write_json(ROOT / "reports/degradation_causes.json", out)
    (ROOT / "reports/DEGRADATION_CAUSES.md").write_text(render(out, data))
    print("wrote reports/DEGRADATION_CAUSES.md")


def render(out, data):
    lines = ["# 退化性能下降的机制诊断", ""]
    lines += [
        "数据只来自组合增强基线（民船 8 类、seed 17、256 条件中的关键项），"
        "所有指标由保存的逐样本预测重算。目标是回答“下降来自哪里、哪部分还可恢复”，"
        "而不是再列一遍准确率。", "",
        "## 1. 翻转：干净答对、退化答错", "",
        "| 条件 | 模型 | 准确率 | 干净答对样本 | 翻转数 | 翻转率 | 错误均值置信度 | 高置信错误占比 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in KEY:
        for m in RUNS:
            e = out["key"].get(m, {}).get(c)
            if not e:
                continue
            f, cf = e["flips"], e["confidence"]
            lines.append(
                f"| {c} | {DISPLAY[m]} | {pct(e['accuracy'])} | {f['clean_correct']} | "
                f"{f['flipped']} | {pct(f['flip_rate'])} | {pct(cf['error_mean_maxprob'])} | "
                f"{pct(cf['overconfident_share'])} |")
    lines += ["", "解读：翻转率的分母是该模型在干净集答对的样本，衡量“已学会的样本在退化下丢了多少”；"
              "错误均值置信度越高，说明模型不是不知道，而是自信地错。", ""]

    lines += ["## 2. 跨模型共识：信息还在不在", "",
              "三个架构（ResNet-50 / ConvNeXt-Tiny / Swin-Tiny，同种子、同组合增强）"
              "对同一测试样本的判定。前两列以全部样本为分母；后三列以该模型自身的错误为分母，"
              "衡量它的错误里有多少是“三个模型都不会”。", "",
              "| 条件 | 三模型全错样本占比 | 三模型全对占比 | Swin 错误中全错 | ConvNeXt 错误中全错 | ResNet 错误中全错 | 样本数分布 (0/1/2/3 模型对) |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for c, e in out["consensus"].items():
        dist = "/".join(str(e["wrong_by_count"][k]) for k in range(4))
        pm = e["per_model"]
        lines.append(
            f"| {c} | {pct(e['share_all_wrong'])}% | {pct(e['share_all_correct'])}% | "
            f"{pct(pm['swin_t']['share_errors_universal'])}% | "
            f"{pct(pm['convnext_tiny']['share_errors_universal'])}% | "
            f"{pct(pm['resnet50']['share_errors_universal'])}% | {dist} |")
    lines += ["", "全错的样本更可能是信息已不可辨认；至少一个模型答对的样本，说明判别信息仍在，"
              "下降来自模型利用信息的方式，这是新模块的可争取空间。", ""]

    lines += ["## 3. 物理参数与错误的关系", "",
              "透射率越低表示雾越厚；云面积、受影响面积、饱和面积越大表示遮挡/过曝越重。"
              "ρ 为参数与逐样本正确性（1=对）的 Spearman 相关：对透射率，正 ρ 表示越清晰越容易对；"
              "对面积/比例类参数，负 ρ 表示越大越容易错。", ""]
    for kind in KINDS:
        p = out["parameters"][kind]
        lines += [f"### {kind}", "",
                  "| 参数 | 模型 | 合并 ρ | 严重度内平均 ρ | 五级低三分位准确率 | 五级高三分位准确率 |",
                  "|---|---|---:|---:|---:|---:|"]
        for param, by_model in p["params"].items():
            for m, e in by_model.items():
                t = e["severity5_tertiles"] or {}
                within = ("—" if e["within_severity_rho"] is None
                          else f"{e['within_severity_rho']:+.2f}")
                pooled = ("—" if e["pooled_rho"] is None
                          else f"{e['pooled_rho']:+.2f}")
                lines.append(
                    f"| {param} | {DISPLAY[m]} | {pooled} | {within} | "
                    f"{pct(t.get('low_acc'))} | {pct(t.get('high_acc'))} |")
        lines.append("")

    lines += ["## 4. 类别级损失（Swin-Tiny，组合增强）", "",
              "| 类别 | n | 干净 | 眩光5 | 雾+眩光5 | 雾+云5 | 雾+眩光5 主要去向 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for r in out["classes"]:
        dest = sorted(r["matched_haze+glare_5_dest"].items(), key=lambda x: -x[1])[:2]
        dest = "、".join(f"{k}×{v}" for k, v in dest) or "—"
        lines.append(f"| {r['class']} | {r['n']} | {pct(r['clean'])} | {pct(r['matched_glare_5'])} | "
                     f"{pct(r['matched_haze+glare_5'])} | {pct(r['matched_haze+cloud_5'])} | {dest} |")
    lines.append("")

    lines += ["## 5. 机制判断", ""]
    lines += mechanism_lines(out)
    lines += ["", "## 6. 对新模块的含义", ""]
    lines += module_lines(out)
    lines += ["", "## 复现", "", "```bash",
              ".venv/bin/python scripts/degradation_causes.py", "```", ""]
    return "\n".join(lines)


def mechanism_lines(out):
    # computed statements only, with explicit denominators
    lines = []
    c = out["consensus"].get("matched_haze+glare_5")
    if c:
        sw = c["per_model"]["swin_t"]
        lines.append(
            f"- 五级雾+眩光下 {pct(c['share_all_wrong'])}% 的样本三个模型全错；"
            f"Swin-Tiny 的 {sw['errors']} 个错误中 {sw['errors_universal']} 个"
            f"（{pct(sw['share_errors_universal'])}%）属于这种全错样本，"
            f"其余 {pct(1 - sw['share_errors_universal'])}% 至少有一个模型能答对，是可恢复空间。")
    c = out["consensus"].get("matched_glare_5")
    if c:
        lines.append(
            f"- 五级眩光下三模型全错只占全部样本的 {pct(c['share_all_wrong'])}%，"
            "单退化下判别信息基本还在，下降更多来自模型没有利用好，而不是信息消失。"
            f"（Swin 错误中全错占比 {pct(c['per_model']['swin_t']['share_errors_universal'])}%）")
    # confidence
    e = out["key"].get("swin_t", {}).get("matched_haze+glare_5")
    if e:
        lines.append(
            f"- Swin-Tiny 在五级雾+眩光下错误样本的平均最大置信度 {pct(e['confidence']['error_mean_maxprob'])}%，"
            f"其中 {pct(e['confidence']['overconfident_share'])}% 的错误置信度 ≥0.9："
            "错误不是低置信犹豫，而是高置信误判，置信度本身不能直接当拒识依据。")
    # pooled vs within-severity correlations
    for kind in KINDS:
        p = out["parameters"][kind]["params"]
        pooled, within = [], []
        for param, by_model in p.items():
            for v in by_model.values():
                if v["pooled_rho"] is not None:
                    pooled.append(abs(v["pooled_rho"]))
                if v["within_severity_rho"] is not None:
                    within.append(abs(v["within_severity_rho"]))
        if not pooled:
            continue
        if np.mean(pooled) < 0.15:
            lines.append(
                f"- {kind}：退化参数与正确性的合并 |ρ| 均值只有 {np.mean(pooled):.2f}，"
                f"同一严重度内 {np.mean(within):.2f}。样本级的失败几乎不能由这些可观测参数解释，"
                "说明起作用的不是“强度连续变大”，而是具体图像内容与破坏位置的组合。")
        else:
            lines.append(
                f"- {kind}：合并 |ρ| 均值 {np.mean(pooled):.2f}，同一严重度内降到 {np.mean(within):.2f}。"
                "合并相关主要来自严重度之间的差异；给定同一强度后，参数对失败样本的区分力有限。")
    # which parameter best separates failures within a severity
    rows = []
    for kind in KINDS:
        for param, by_model in out["parameters"][kind]["params"].items():
            vals = [abs(v["within_severity_rho"]) for v in by_model.values()
                    if v["within_severity_rho"] is not None]
            if vals:
                rows.append((kind, param, float(np.mean(vals))))
    if rows:
        kind, param, val = max(rows, key=lambda r: r[2])
        if val < 0.2:
            lines.append(
                f"- 所有参数中，同一严重度内区分力最强的也只是 {kind} 的 {param}"
                f"（平均 |ρ|={val:.2f}），仍然很弱。失败与否更多由图像内容决定，"
                "而不是由某一个可测量的退化强度决定。")
        else:
            lines.append(
                f"- 同一严重度内区分力最强的参数是 {kind} 的 {param}"
                f"（平均 |ρ|={val:.2f}），可作为模块的输入线索之一。")
    return lines


def module_lines(out):
    lines = []
    c = out["consensus"].get("matched_haze+glare_5")
    if c:
        sw = c["per_model"]["swin_t"]
        lines.append(
            f"- 可争取空间：五级组合下仍有 {pct(1 - c['share_all_wrong'])}% 的样本至少有一个模型答对；"
            f"Swin-Tiny 的 {pct(sw['share_errors_universal'])}% 错误属于“三模型全错”，"
            f"剩下的 {pct(1 - sw['share_errors_universal'])}% 才是模块的现实目标。")
    singles = [out["consensus"][k]["share_all_wrong"]
               for k in ("matched_haze_5", "matched_cloud_5", "matched_glare_5")
               if k in out["consensus"]]
    if singles:
        lines.append(
            f"- 单退化下三模型全错只占 {pct(min(singles))}%–{pct(max(singles))}%，"
            "说明增强训练已经把大部分单退化缺口补上；新模块的主战场是组合退化，"
            "且不能以牺牲干净图和单退化为代价。")
    lines.append(
        "- 失败与全局退化参数在同一严重度内几乎无关，"
        "“先估计雾浓度/饱和比例再复原”不是关键；更合理的是让网络判断哪些局部证据已不可信，"
        "并在聚合时降低其权重。")
    lines.append(
        "- 类别级损失集中在结构相近类别（油轮↔散货船、LPG↔驳船、客船↔散货船），"
        "与“判别性部件被遮挡/饱和”一致，而不是全局颜色或亮度偏移。")
    lines += ["",
              "第一版模块（Reliability-Gated Part Aggregation，RGPA）：在骨干最后一阶段之后，"
              "用两路信号计算每个 token 的可靠性——输入侧的局部饱和比例与局部对比度（固定算子，无参数），"
              "以及一个 1×1 卷积学到的可靠性头；用可靠性加权池化替代全局平均池化，"
              "并保留一条原始 GAP 的残差路径。参数量目标 <0.1M，不改数据配方与训练步数。",
              "",
              "验证顺序：先做同骨干、同增强、同种子的“无门控”对照，确认能复现组合增强基线；"
              "再加门控，并同时报告干净、单退化、组合退化与成本。"]
    return lines


if __name__ == "__main__":
    main()
