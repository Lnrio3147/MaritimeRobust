#!/usr/bin/env python
"""Degradation diagnostics for the repaired CDLNet and the clean baselines.

Everything is recomputed from saved per-condition predictions, so no model is
re-run. Definitions follow scripts/diagnostics_report.py and report.py:
a flip is a sample whose clean prediction was correct, a high-confidence error
is an error with max softmax >= 0.9, and intervals are grouped by source group.

Two scopes are reported: the civilian 8-class main experiment and the full
23-class research split (external-model reference).
"""
import collections
import json

import numpy as np

from maritime_robust.common import ROOT, read_jsonl, write_json
from maritime_robust.report import grouped_f1_interval

SCOPES = {
    "civilian": {
        "title": "民船主实验（8 类）",
        "cdlnet_repro": "runs_external/fgsc23_research_civilian/cdlnet_clean_s17",
        "resnet50": "runs/fgsc23_research_civilian/resnet50_clean_s17",
        "convnext_tiny": "runs/fgsc23_research_civilian/convnext_tiny_clean_s17",
        "swin_t": "runs/fgsc23_research_civilian/swin_t_clean_s17",
    },
    "all": {
        "title": "全 23 类（research split，外部模型参照）",
        "cdlnet_repro": "runs_external/fgsc23_research_all/cdlnet_clean_s17",
        "resnet50": "runs/fgsc23_research_all/resnet50_clean_s17",
        "convnext_tiny": "runs/fgsc23_research_all/convnext_tiny_clean_s17",
        "swin_t": "runs/fgsc23_research_all/swin_t_clean_s17",
    },
}
KINDS = ("haze", "cloud", "glare")
FAMILIES = {
    "clean": ["clean"],
    "matched single": [f"matched_{k}_{s}" for k in KINDS for s in range(1, 6)],
    "shifted single": [f"shifted_{k}_{s}" for k in KINDS for s in range(1, 6)],
    "matched combination": [f"matched_{k}_{s}" for k in ("haze+glare", "haze+cloud") for s in (3, 5)],
    "shifted combination": [f"shifted_{k}_{s}" for k in ("haze+glare", "haze+cloud") for s in (3, 5)],
}
FAMILY_DISPLAY = {"clean": "干净", "matched single": "单退化 matched",
                  "shifted single": "单退化 shifted", "matched combination": "组合 matched",
                  "shifted combination": "组合 shifted"}
INTEREST = ["clean", "matched_haze_5", "matched_cloud_5", "matched_glare_5",
            "matched_haze+glare_5", "matched_haze+cloud_5", "shifted_haze+cloud_5"]
CI_CONDITIONS = ("clean", "matched_haze_5", "matched_cloud_5", "matched_glare_5",
                 "matched_haze+glare_5", "matched_haze+cloud_5", "shifted_haze+cloud_5")
DISPLAY = {"cdlnet_repro": "CDLNet(修复版)", "resnet50": "ResNet-50",
           "convnext_tiny": "ConvNeXt-Tiny", "swin_t": "Swin-Tiny"}
HC = 0.9


def load(name, path):
    run = ROOT / path
    config = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "evaluation/summary.json").read_text())
    preds = {c: read_jsonl(run / "evaluation" / f"{c}.jsonl") for c in summary}
    return {"name": name, "run": path, "config": config, "summary": summary, "preds": preds}


def per_class(y, p, n):
    cm = np.zeros((n, n), dtype=int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    support = cm.sum(1)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(n, dtype=float), where=support > 0)
    return recall, support


def family_scores(model, family):
    names = FAMILIES[family]
    return {"macro_f1": float(np.mean([model["summary"][c]["macro_f1"] for c in names])),
            "accuracy": float(np.mean([model["summary"][c]["accuracy"] for c in names])),
            "conditions": names}


def retention(model, family):
    if family == "clean":
        return {"retention": None, "clean_correct_observations": 0}
    clean = {r["sample_id"]: r for r in model["preds"]["clean"]}
    survived = total = 0
    for c in FAMILIES[family]:
        for r in model["preds"][c]:
            if clean[r["sample_id"]]["prediction"] == r["label"]:
                total += 1
                survived += int(r["prediction"] == r["label"])
    return {"retention": survived / total if total else None,
            "clean_correct_observations": total}


def class_recall(model, family, n):
    recalls = []
    for c in FAMILIES[family]:
        rows = model["preds"][c]
        recalls.append(per_class([r["label"] for r in rows], [r["prediction"] for r in rows], n)[0])
    return np.mean(recalls, axis=0)


def confusion_pairs(model, condition, labels, limit=8):
    counter = collections.Counter()
    for r in model["preds"][condition]:
        if r["label"] != r["prediction"]:
            counter[(r["label"], r["prediction"])] += 1
    return [{"true": labels[str(t)], "predicted": labels[str(p)], "count": n}
            for (t, p), n in counter.most_common(limit)]


def confidence_stats(model, condition):
    rows = model["preds"][condition]
    clean = {r["sample_id"]: r for r in model["preds"]["clean"]}
    errors = [r for r in rows if r["prediction"] != r["label"]]
    high = [r for r in errors if max(r["probabilities"]) >= HC]
    flips = [r for r in rows
             if clean[r["sample_id"]]["prediction"] == r["label"] and r["prediction"] != r["label"]]
    correct = [max(r["probabilities"]) for r in rows if r["prediction"] == r["label"]]
    return {"n": len(rows), "errors": len(errors), "error_rate": len(errors) / len(rows),
            "high_confidence_errors": len(high),
            "high_confidence_share_of_errors": len(high) / len(errors) if errors else 0.0,
            "mean_confidence_correct": float(np.mean(correct)) if correct else None,
            "mean_confidence_wrong": float(np.mean([max(r["probabilities"]) for r in errors]))
            if errors else None,
            "clean_correct_to_wrong": len(flips)}


def top_prediction(model, condition, labels):
    counts = collections.Counter(r["prediction"] for r in model["preds"][condition])
    index, count = counts.most_common(1)[0]
    return {"label": labels[str(index)], "count": count,
            "share": count / len(model["preds"][condition])}


def analyze(key):
    spec = SCOPES[key]
    models = [load(name, path) for name, path in spec.items() if name != "title"]
    labels = models[0]["config"]["classes"]
    n = models[0]["config"]["num_classes"]
    out = {"title": spec["title"], "run": spec["cdlnet_repro"],
           "models": [m["name"] for m in models], "labels": labels, "num_classes": n,
           "high_confidence_threshold": HC}

    out["family_scores"] = {
        m["name"]: {f: {**family_scores(m, f), **retention(m, f)} for f in FAMILIES}
        for m in models
    }
    out["severity_curves"] = {
        m["name"]: {split: {k: [m["summary"][f"{split}_{k}_{s}"]["macro_f1"] for s in range(1, 6)]
                            for k in KINDS} for split in ("matched", "shifted")}
        for m in models
    }

    cdlnet = models[0]
    _, support = per_class([r["label"] for r in cdlnet["preds"]["clean"]],
                           [r["prediction"] for r in cdlnet["preds"]["clean"]], n)
    out["class_drops"] = {}
    for model in models:
        clean_r = class_recall(model, "clean", n)
        single_r = class_recall(model, "matched single", n)
        combo_r = class_recall(model, "matched combination", n)
        order = np.argsort(clean_r - single_r)[::-1]
        out["class_drops"][model["name"]] = [
            {"label": labels[str(i)], "support": int(support[i]),
             "clean_recall": float(clean_r[i]),
             "matched_single_recall": float(single_r[i]),
             "matched_combination_recall": float(combo_r[i]),
             "drop_matched_single": float(clean_r[i] - single_r[i]),
             "drop_matched_combination": float(clean_r[i] - combo_r[i])}
            for i in order if support[i] > 0
        ]

    out["confusions"] = {
        m["name"]: {c: confusion_pairs(m, c, labels)
                    for c in ("clean", "matched_haze+glare_5", "shifted_haze+cloud_5")}
        for m in models
    }
    out["confidence"] = {m["name"]: {c: confidence_stats(m, c) for c in INTEREST}
                         for m in models}
    out["placement"] = {m["name"]: {
        "cloud_3": {p: m["summary"][f"matched_cloud_3_{p}"]["macro_f1"] for p in ("center", "border")},
        "glare_3": {p: m["summary"][f"matched_glare_3_{p}"]["macro_f1"] for p in ("center", "border")},
    } for m in models}
    out["collapse"] = {m["name"]: {c: top_prediction(m, c, labels)
                                   for c in ("clean", "matched_haze+glare_5",
                                             "shifted_haze+cloud_5")}
                       for m in models}
    out["intervals"] = {c: grouped_f1_interval(cdlnet["preds"][c], n) for c in CI_CONDITIONS}
    out["condition_macro_f1"] = {m["name"]: {c: v["macro_f1"] for c, v in m["summary"].items()}
                                 for m in models}
    return out


def pct(x, digits=1):
    return "—" if x is None else f"{100 * x:.{digits}f}"


def table(rows, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def scope_sections(d, heading_level):
    fam = d["family_scores"]
    names = d["models"]
    h = "#" * heading_level
    clean_f1 = {m: fam[m]["clean"]["macro_f1"] for m in names}

    rows = []
    for family in FAMILIES:
        row = [FAMILY_DISPLAY[family]]
        for m in names:
            value = fam[m][family]["macro_f1"]
            drop = value - clean_f1[m]
            row.append(f"{pct(value)} ({drop * 100:+.1f})" if family != "clean" else pct(value))
        rows.append(row)
    main_table = table(rows, ["条件族（Macro-F1，括号内为相对干净的变化）"]
                       + [DISPLAY[m] for m in names])

    ret_rows = [[FAMILY_DISPLAY[f]] + [pct(fam[m][f]["retention"]) for m in names]
                for f in FAMILIES if f != "clean"]
    retention_table = table(ret_rows, ["条件族", *[DISPLAY[m] for m in names]])

    curve_rows = []
    for kind in KINDS:
        curve_rows.append([f"{kind}（matched）"] + [pct(d["severity_curves"]["cdlnet_repro"]["matched"][kind][i]) for i in range(5)])
        curve_rows.append([f"{kind}（shifted）"] + [pct(d["severity_curves"]["cdlnet_repro"]["shifted"][kind][i]) for i in range(5)])
    curve_table = table(curve_rows, ["退化", "1", "2", "3", "4", "5"])
    base_rows = []
    for m in ("resnet50", "convnext_tiny", "swin_t"):
        for kind in KINDS:
            base_rows.append([f"{DISPLAY[m]} {kind}"]
                             + [pct(d["severity_curves"][m]["matched"][kind][i]) for i in range(5)])
    base_table = table(base_rows, ["模型 / 退化（matched）", "1", "2", "3", "4", "5"])

    drops = d["class_drops"]["cdlnet_repro"]
    n_show = 8 if len(drops) > 8 else len(drops)
    class_table = table(
        [[r["label"], r["support"], pct(r["clean_recall"]), pct(r["matched_single_recall"]),
          pct(r["matched_combination_recall"]), f"{100 * r['drop_matched_single']:.1f}",
          f"{100 * r['drop_matched_combination']:.1f}"] for r in drops[:n_show]],
        ["类别", "测试数", "干净召回", "单退化均值", "组合均值", "单退化跌幅(pp)", "组合跌幅(pp)"])

    top5 = {m: [r["label"] for r in d["class_drops"][m][:5]] for m in names}
    overlap_table = table(
        [[DISPLAY[m], "、".join(top5[m]), len(set(top5[m]) & set(top5["cdlnet_repro"]))]
         for m in names if m != "cdlnet_repro"],
        ["模型", "单退化跌幅前 5 类", "与 CDLNet 前 5 的重合数"])
    overlap_txt = "、".join(
        f"{DISPLAY[m]} {len(set(top5[m]) & set(top5['cdlnet_repro']))}/5"
        for m in names if m != "cdlnet_repro")

    def confusion_table(model, condition):
        return table([[r["true"], r["predicted"], r["count"]]
                      for r in d["confusions"][model][condition]],
                     ["真实", "预测", "次数"])

    conf_rows = []
    for condition in INTEREST:
        entry = d["confidence"]["cdlnet_repro"][condition]
        base = d["confidence"]["convnext_tiny"][condition]
        conf_rows.append([condition, entry["errors"], entry["high_confidence_errors"],
                          pct(entry["high_confidence_share_of_errors"]),
                          entry["clean_correct_to_wrong"], base["clean_correct_to_wrong"]])
    conf_table = table(conf_rows, ["条件", "错误数", "高置信错误(≥0.9)", "占错误比例(%)",
                                   "干净对→错（CDLNet）", "干净对→错（ConvNeXt）"])

    collapse_rows = [[DISPLAY[m]]
                     + [f"{d['collapse'][m][c]['label']} ({pct(d['collapse'][m][c]['share'])}%)"
                        for c in ("clean", "matched_haze+glare_5", "shifted_haze+cloud_5")]
                     for m in names]
    collapse_table = table(collapse_rows, ["模型", "clean 首位预测",
                                           "matched 5 级雾+眩光首位预测", "shifted 5 级雾+云首位预测"])
    cd_col = d["collapse"]["cdlnet_repro"]["matched_haze+glare_5"]
    base_col = max(d["collapse"][m]["matched_haze+glare_5"]["share"]
                   for m in names if m != "cdlnet_repro")
    collapse_gap = 100 * (cd_col["share"] - base_col)
    shares = [d["collapse"][m]["matched_haze+glare_5"]["share"] for m in names]
    if min(shares) > 0.2:
        common_txt = (f"所有模型在该条件下的首位类别占比都在 20% 以上"
                      f"（{pct(min(shares))}%–{pct(max(shares))}%），单一类别塌缩是共性问题。")
    else:
        common_txt = (f"该条件下各模型首位类别占比为 {pct(min(shares))}%–{pct(max(shares))}%。")
    collapse_note = (
        f"matched 五级雾+眩光下，CDLNet 有 {pct(cd_col['share'])}% 的样本被预测为 {cd_col['label']}，"
        f"比基线中最高占比还高 {collapse_gap:.1f} 个百分点，出现明显的预测塌缩。" + common_txt
        if collapse_gap > 10 else
        f"matched 五级雾+眩光下，CDLNet 的首位预测为 {cd_col['label']}（{pct(cd_col['share'])}%），"
        f"与基线中最高占比相差 {collapse_gap:+.1f} 个百分点。" + common_txt)

    place_rows = [[DISPLAY[m]]
                  + [pct(d["placement"][m]["cloud_3"][p]) for p in ("center", "border")]
                  + [pct(d["placement"][m]["glare_3"][p]) for p in ("center", "border")]
                  for m in names]
    place_table = table(place_rows, ["模型", "云-中心", "云-边缘", "眩光-中心", "眩光-边缘"])

    ci_rows = [[c, pct(d["condition_macro_f1"]["cdlnet_repro"][c]),
                pct(d["intervals"][c]["low"]), pct(d["intervals"][c]["high"])] for c in CI_CONDITIONS]
    ci_table = table(ci_rows, ["条件", "Macro-F1", "2.5%", "97.5%"])

    combo_text = "、".join(f"{DISPLAY[m]} {pct(fam[m]['matched combination']['macro_f1'])}" for m in names)
    ret_ms = "、".join(f"{DISPLAY[m]} {pct(fam[m]['matched single']['retention'])}" for m in names)
    ret_mc = "、".join(f"{DISPLAY[m]} {pct(fam[m]['matched combination']['retention'])}" for m in names)
    clean_gap = 100 * (fam["cdlnet_repro"]["clean"]["macro_f1"] - fam["resnet50"]["clean"]["macro_f1"])
    combo5 = FAMILIES["matched combination"]
    s = d["condition_macro_f1"]["cdlnet_repro"]
    combo_lo = min(s[c] for c in combo5)
    combo_hi = max(s[c] for c in combo5)
    worst_c = min(s, key=lambda c: s[c])
    worst_v = s[worst_c]
    single_mean = fam["cdlnet_repro"]["matched single"]["macro_f1"]
    best_m = max((m for m in names if m != "cdlnet_repro"),
                 key=lambda m: fam[m]["matched combination"]["retention"])
    best_r = fam[best_m]["matched combination"]["retention"]
    mc_diff = 100 * (fam["cdlnet_repro"]["matched combination"]["retention"] - best_r)
    fragility = (f"即使扣掉干净集起点，CDLNet 的 matched 组合保持率仍比保持率最高的基线 {DISPLAY[best_m]}"
                 f"（{pct(best_r)}%）低 {abs(mc_diff):.1f} 个百分点"
                 if mc_diff < 0 else
                 f"扣掉干净集起点后，CDLNet 的 matched 组合保持率比保持率最高的基线 {DISPLAY[best_m]}"
                 f"（{pct(best_r)}%）高 {mc_diff:.1f} 个百分点")
    clean_conf = d["confidence"]["cdlnet_repro"]["clean"]
    hc_combo = [d["confidence"]["cdlnet_repro"][c]["high_confidence_share_of_errors"]
                for c in ("matched_haze+glare_5", "matched_haze+cloud_5")]

    return f"""{h} 总体落差（Macro-F1）

{main_table}

“组合”= 雾+眩光 / 雾+云，各 3、5 级，共 4 条。matched 组合均值：{combo_text}；
CDLNet 的干净集比同骨干 ResNet-50 基线{"低" if clean_gap < 0 else "高"} {abs(clean_gap):.1f} 个百分点。

{h} 干净正确样本的保持率

保持率 = 在干净图上预测正确的样本中，退化后仍然正确的比例（15 或 4 条条件合并计算）。

{retention_table}

这个口径把“干净起点”扣掉了。matched 单退化保持率：{ret_ms}；matched 组合保持率：{ret_mc}。

{h} 逐级曲线

CDLNet：

{curve_table}

基线对照（matched）：

{base_table}

{h} 类别级变化（CDLNet，按单退化跌幅排序）

{class_table}

小样本类别（如测试数低于 25 的类别）单类召回波动大，需结合“测试数”列解读。
各模型单退化跌幅最大的 5 类对比：

{overlap_table}

{h} 混淆结构

干净图（CDLNet）前 8 个错误对：

{confusion_table("cdlnet_repro", "clean")}

matched 五级雾+眩光（CDLNet）：

{confusion_table("cdlnet_repro", "matched_haze+glare_5")}

同一条件下 ConvNeXt-Tiny 的前 8 个错误对：

{confusion_table("convnext_tiny", "matched_haze+glare_5")}

shifted 五级雾+云（CDLNet）：

{confusion_table("cdlnet_repro", "shifted_haze+cloud_5")}

{h} 预测塌缩

各模型首位预测类别及占比：

{collapse_table}

{collapse_note}

{h} 高置信错误与翻转

高置信错误 = 预测错误且最大 softmax ≥ 0.9；“干净对→错”= 干净图正确、该条件下错误的样本数。

{conf_table}

干净图上 CDLNet 的 {clean_conf["errors"]} 个错误中有 {clean_conf["high_confidence_errors"]} 个（{pct(clean_conf["high_confidence_share_of_errors"])}%）置信度 ≥0.9；
matched 五级组合下该比例降到 {pct(min(hc_combo))}%–{pct(max(hc_combo))}%，即组合退化下的错误大多不再伴随高置信，
更接近“结构已丢失”而非“细节混淆”。

{h} 位置对照与分组 bootstrap 区间

三级位置对照（Macro-F1）：

{place_table}

CDLNet 分组 bootstrap（按 source_group_id 重采样）：

{ci_table}

{h} 该口径下的判断

- 4 条 matched 组合（3、5 级）的 Macro-F1 在 {pct(combo_lo)}–{pct(combo_hi)} 之间，全部 43 条件最差为
  {worst_c}（{pct(worst_v)}）；matched 单退化均值仍为 {pct(single_mean)}。组合退化不是单退化曲线的线性外推。
- {fragility}。这与“CDLNet 特有分支（区域选择/响应融合/门控注意力）在低对比度下失效”的假设一致，
  但本报告没有区域响应可视化，不能直接断言。
- 单退化跌幅前 5 类与各基线的重合数：{overlap_txt}。重合的类别更可能是数据集共享的细粒度瓶颈；
  只在 CDLNet 上出现的跌幅类别，才值得优先检查区域选择/响应融合分支。
- 高置信错误比例用于判断错误是否“结构已丢失”：比例越高，越不应把错误当作可学习的细节差异。
"""


def write_markdown(data):
    text = f"""# CDLNet 修复版退化诊断（research split，seed 17）

数据：CDLNet 修复版（单阶段 CE、256 输入、40 epoch）与同一 research split、同一种子 17 的干净训练基线
（ResNet-50 / ConvNeXt-Tiny / Swin-Tiny，224 输入、40 epoch）的 43 条件逐样本预测，全部指标由保存预测重算。
CDLNet 是修复版复现，不是作者原成绩（详见 `reports/CDLNET_REPRO.md`）。

主实验为民船 8 类；全 23 类作为外部模型参照单列。CDLNet 的干净集与基线接近或略低，因此同时给出
**保持率**（干净正确样本在退化后仍正确的比例），避免把“起点低”误读成“退化更敏感”。

## 民船主实验（8 类）

{scope_sections(data["civilian"], 3)}

## 全 23 类（外部模型参照）

{scope_sections(data["all"], 3)}

## 方法说明与限制

- 保持率、翻转、高置信错误的统计都要求逐样本配对，因此只使用同一条条件内全部测试样本；区间按
  `source_group_id` 做整组重采样。
- 混淆表按错误次数排序，不按比例；类别支持度差异很大，跨条件比较时需同时看“测试数”。
- 类别跌幅取 15 条 matched 单退化、4 条 matched 组合的平均召回，异常值可能来自单条条件或小样本类别。
- CDLNet 的 256 输入与基线的 224 输入不同，这是修复版复现的既定设置；分辨率差会同时影响干净与退化结果，
  不能把差值全部归给架构。
- 本页只做已知框分类的诊断，不涉及检测或跟踪，也不支持船只身份结论。

## 复现

```bash
.venv/bin/python scripts/cdlnet_report.py
.venv/bin/python scripts/cdlnet_diagnostics.py
```
"""
    (ROOT / "reports/CDLNET_DIAGNOSTICS.md").write_text(text)


def main():
    data = {key: analyze(key) for key in SCOPES}
    write_json(ROOT / "reports/cdlnet_diagnostics.json", data)
    write_markdown(data)
    for key, d in data.items():
        summary = {m: {f: round(v["macro_f1"], 4) for f, v in d["family_scores"][m].items()}
                   for m in d["models"]}
        print(key, json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
