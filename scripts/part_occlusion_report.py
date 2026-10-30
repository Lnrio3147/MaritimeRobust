#!/usr/bin/env python
"""Render reports/PART_OCCLUSION.md from reports/part_occlusion.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maritime_robust.common import ROOT as PROJECT, read_jsonl  # noqa: E402

DISPLAY = {"resnet50": "ResNet-50", "convnext_tiny": "ConvNeXt-Tiny", "swin_t": "Swin-Tiny"}
ORDER = ("resnet50", "convnext_tiny", "swin_t")
CONDS = (("clean", "干净"), ("corner_third", "角落遮挡 1/3"), ("center_third", "中心遮挡 1/3"),
         ("center_half", "中心遮挡 1/2"))


def table(rows, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def pct(x):
    return f"{100 * x:.1f}"


def main():
    data = json.loads((PROJECT / "reports/part_occlusion.json").read_text())
    rows = [r for r in read_jsonl(PROJECT / "data/manifests/fgsc23_research_civilian.jsonl")
            if r["split"] == "test"]
    names = {r["label"]: r["label_name"] for r in rows}

    main_rows = []
    for model in ORDER:
        e = data["models"][model]
        for cond, label in CONDS:
            c = e["conditions"][cond]
            main_rows.append([DISPLAY[model] if cond == "clean" else "", label,
                              pct(c["accuracy"]), pct(c["macro_f1"]),
                              f"{c['delta_accuracy_pp']:+.1f}" if cond != "clean" else "—",
                              c["wrong"]])
    position_rows = []
    for model in ORDER:
        e = data["models"][model]
        corner = -e["conditions"]["corner_third"]["delta_accuracy_pp"]
        center = -e["conditions"]["center_third"]["delta_accuracy_pp"]
        half = -e["conditions"]["center_half"]["delta_accuracy_pp"]
        position_rows.append([DISPLAY[model], f"{corner:.1f}", f"{center:.1f}",
                              f"{center / corner:.1f}×", f"{half:.1f}"])
    overlap_rows = []
    for model in ORDER:
        o = data["models"][model]["overlap"]
        center_wrong = data["models"][model]["conditions"]["center_third"]["wrong"]
        overlap_rows.append([DISPLAY[model], center_wrong,
                             f"{100 * o['center_errors_also_degraded_wrong']:.1f}",
                             f"{100 * o['degraded_errors_also_center_wrong']:.1f}",
                             f"{o['center_vs_haze_glare_5_jaccard']:.2f}"])
    confusion_rows = []
    for model in ORDER:
        e = data["models"][model]
        center = "；".join(f"{a}→{b}×{n}" for a, b, n in e["confusions"]["center_third"][:3])
        degraded = "；".join(f"{a}→{b}×{n}" for a, b, n in e["degraded_reference"]["confusions"][:3])
        confusion_rows.append([DISPLAY[model], center, degraded])
    class_rows = []
    for model in ("swin_t", "resnet50"):
        e = data["models"][model]
        for label in range(8):
            class_rows.append([DISPLAY[model] if label == 0 else "", names[label],
                               pct(e["conditions"]["clean"]["score"]["per_class_recall"][label]),
                               pct(e["conditions"]["center_third"]["score"]["per_class_recall"][label]),
                               pct(e["conditions"]["center_half"]["score"]["per_class_recall"][label])])
    degraded_acc = {m: data["models"][m]["degraded_reference"]["accuracy"] for m in ORDER}

    text = f"""# 遮挡干预：中心证据被破坏能否解释退化错误

## 0. 结论

能做的是**图像区域遮挡**，不是船体部件 mask：FGSC-23 没有框/多点标注（清单无 `target_mask`，
原始发布只有按类别分目录的图像），FGSRCS 有定向框但类别体系与域都不同。因此本实验用固定
灰块在 letterbox 之前遮挡图像区域，回答两个可检验的问题：误差是否位置特异、混淆结构是否
与严重退化一致。

结果支持“中心证据被破坏”这一层解释：

- 同面积灰块，中心区造成的准确率下降是角落区的 3.4–6.3 倍（第 3 节），说明主要代价来自
  被遮挡的中心内容，而不只是灰块本身；角落同面积遮挡只有 1.7–9.3 pp。
- 遮挡面积从 1/3 增到 1/2，三个模型都继续下降，呈剂量反应。
- ResNet-50 对中心遮挡最敏感（−31.4 pp），与它在五级雾+眩光下最差（47.0）一致；
  Swin-Tiny 对中心遮挡最不敏感（−10.6 pp），但退化下（63.1）略低于 ConvNeXt-Tiny（65.7），
  说明还存在中心遮挡以外的因素。
- Swin-Tiny 在中心遮挡下的前三混淆对与五级雾+眩光几乎重合（油轮→散货船 12 vs 13、
  LPG→驳船 9 vs 13、集装箱船→散货船 6 vs 6）。
- 中心遮挡错误中 50.7%–69.4% 同时也是雾+眩光错误；反向覆盖 41.4%–54.4%。说明中心证据
  损失能解释退化错误的一大块，但不是全部——退化还包含非中心的全局光度与边缘影响。

但本实验**不能**定位船首/船尾/上层建筑：中心方块不是船体 mask，也无法区分“船体”与
“画面中央的任何结构”。要回答部件级问题，需要 FGSC-23 之外的带框数据或人工标注。

## 1. 方法

- 数据：FGSC-23 民船 8 类测试集 236 张，与原评测完全相同的图像。
- 条件：干净；中心正方形，面积占图 1/3；同尺寸正方形放在四角中梯度能量最低的一角
  （同面积控制）；中心正方形，面积占图 1/2（剂量）。
- 灰块颜色 `(124,116,104)`，与 letterbox 填充色一致；遮挡发生在 `contain` 之后、`pad` 之前。
- 模型：组合增强基线的 seed 17 三个架构，checkpoint 与 43 条件评测完全相同。
- 管线校验：三个模型在“干净”条件下的逐样本预测与既有 `evaluation/clean.jsonl` 完全一致，
  否则脚本直接报错。

指标：准确率与 Macro-F1；`Δ` 为相对该模型干净条件的百分点变化；错误数分母为 236。
“重叠”定义：中心遮挡错误集合与五级雾+眩光错误集合的交集；Jaccard = 交集 / 并集。

## 2. 主结果

{table(main_rows, ["模型", "条件", "准确率", "Macro-F1", "Δ准确率(pp)", "错误数"])}

## 3. 位置特异性与剂量

下降幅度按“干净 − 条件”计，倍数为中心下降 / 角落下降：

{table(position_rows, ["模型", "角落下降(pp)", "中心下降(pp)", "中心/角落", "中心 1/2 下降(pp)"])}

角落遮挡本身也有 1.7–9.3 pp 的代价，说明灰块对模型是分布外刺激；但中心遮挡远大于此。
敏感度排序为 ResNet-50 > ConvNeXt-Tiny > Swin-Tiny，ResNet-50 的最差项与退化一致，
但 Swin/ConvNeXt 的相对位置在退化下互换，不能把退化完全等同于中心遮挡。

## 4. 与五级雾+眩光的误差重叠

{table(overlap_rows, ["模型", "中心遮挡错误数", "其中也是雾+眩光错误(%)", "雾+眩光错误中也是中心错误(%)", "Jaccard"])}

## 5. 混淆结构

各模型错误最多的三个“真值→预测”对：

{table(confusion_rows, ["模型", "中心遮挡 1/3", "五级雾+眩光"])}

Swin-Tiny 的两组混淆几乎一一对应；ResNet-50 与 ConvNeXt-Tiny 也共享油轮→散货船、
客船→散货船等主要去向，但具体排序不同。这与“同类结构混淆”一致，而不是随机崩溃。

## 6. 类别级召回（Swin-Tiny 与 ResNet-50）

{table(class_rows, ["模型", "类别", "干净", "中心遮挡 1/3", "中心遮挡 1/2"])}

中心遮挡对 LPG、驳船、客船、油轮的打击最大，正是退化诊断中损失集中的类别；散货船与
集装箱船相对稳。中心 1/2 下 ResNet-50 的 LPG 召回从 100 降到 14、客船从 94 降到 6。

## 7. 解释与边界

- 支持：严重退化下的错误有相当一部分可以由“中心判别证据被遮挡/饱和”复现，且位置特异、
  有剂量反应、混淆结构一致。这与 RGPA 的失败也自洽：破坏发生在 7×7 token 内部，池化层
  的重加权无法恢复。
- 不支持“全部解释”：中心遮挡错误中约一半到七成同时也是退化错误（50.7%–69.4%），反向
  约四到五成；Jaccard 0.31–0.44。退化还有非中心的光度与边缘效应，不能全部归因于局部遮挡。
- 限制：图像区域不是船体部件；中心方块会覆盖部分背景；灰块是合成分布外刺激；单种子、
  236 张测试图；没有显著性检验，所有差值都是描述性的。
- 下一步若要部件级因果结论：需要带定向框/分割的数据（FGSRCS 有框但类别与域不同），或对
  FGSC-23 子集人工标注船体 mask；在此之前不应使用“船首/船尾/上层建筑”这类表述。

## 复现

```bash
.venv/bin/python scripts/part_occlusion.py
.venv/bin/python scripts/part_occlusion_report.py
```
"""
    (PROJECT / "reports/PART_OCCLUSION.md").write_text(text)
    print(json.dumps({"models": list(data["models"]), "n": data["n"]}))


if __name__ == "__main__":
    main()
