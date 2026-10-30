#!/usr/bin/env python
"""Render reports/ROUTE_AB_ROUND1.md from restoration_probe.json / local_recovery.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maritime_robust.common import ROOT as PROJECT  # noqa: E402

DISPLAY = {"resnet50": "ResNet-50", "convnext_tiny": "ConvNeXt-Tiny", "swin_t": "Swin-Tiny"}
ORDER = ("resnet50", "convnext_tiny", "swin_t")
COND_LABEL = {"clean": "干净", "matched_haze_5": "雾 5", "matched_glare_5": "眩光 5",
              "matched_haze+glare_5": "雾+眩光 5", "matched_haze+cloud_5": "雾+云 5",
              "shifted_haze+glare_5": "shifted 雾+眩光 5"}


def table(rows, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def pct(x):
    return f"{100 * x:.1f}"


def main():
    a = json.loads((PROJECT / "reports/restoration_probe.json").read_text())
    b = json.loads((PROJECT / "reports/local_recovery.json").read_text())

    a_rows = []
    for model in ORDER:
        for cond in a["conditions"]:
            v = a["models"][model]["conditions"][cond]
            ident = v["methods"]["identity"]
            cells = []
            for method in ("clahe", "satfill", "fusion_clahe", "fusion_satfill"):
                m = v["methods"][method]
                cells.append(f"{pct(m['accuracy'])} ({m['delta_accuracy_pp']:+.1f}; "
                             f"{m['rescued']}/{m['hurt']})")
            a_rows.append([DISPLAY[model] if cond == a["conditions"][0] else "",
                           COND_LABEL.get(cond, cond), pct(ident["accuracy"]),
                           pct(ident["macro_f1"]), *cells])

    b_rows = []
    grids = {}
    for model in ORDER:
        for cond, v in b["models"][model]["conditions"].items():
            cells = v["cells"]
            center = [c for c in cells if c["row"] == 1 and c["col"] == 1][0]
            off = [c for c in cells if not (c["row"] == 1 and c["col"] == 1)]
            off_mean = sum(c["rescue_rate_recoverable"] for c in off) / len(off)
            gap = v["clean_accuracy"] - v["degraded_accuracy"]
            closure = ((center["accuracy"] - v["degraded_accuracy"]) / gap) if gap else 0.0
            b_rows.append([DISPLAY[model] if cond.startswith("matched_haze_5") else "",
                           COND_LABEL.get(cond, cond), v["recoverable"],
                           pct(v["degraded_accuracy"]), pct(v["clean_accuracy"]),
                           pct(center["rescue_rate_recoverable"]),
                           pct(off_mean), pct(center["accuracy"]),
                           pct(closure), pct(center["error_induction_rate_stable"]),
                           f"{center['margin_delta_recoverable']:+.3f}"])
            if cond == "matched_haze+glare_5":
                grids[model] = [[pct(cells[r * 3 + c]["rescue_rate_recoverable"])
                                 for c in range(3)] for r in range(3)]

    grid_text = "\n\n".join(
        f"**{DISPLAY[m]}**\n\n" + table(g, ["左", "中", "右"])
        for m, g in grids.items())

    text = f"""# 轮次报告：复原探测（A）与局部反事实恢复（B）

## 0. 结论

- 传统全局复原（CLAHE、饱和像素填充）及原图/复原概率融合，在六类条件下都没有同时满足
  “退化增益且干净不降”。最好的一组是 CLAHE 对雾+云五级的 +3.4 pp（ResNet-50）与
  shifted 雾+眩光五级的 +3.8 pp（Swin-Tiny），但同一方法在干净图下降 3.4–3.8 pp；
  概率融合只能减小损害，不能消除。
- 眩光五级平均有 37.0% 像素在 250 阈值下饱和，雾+眩光五级为 62.1%。饱和像素的纹理已经
  被裁剪，最近邻填充只能补颜色，不能补结构；ConvNeXt-Tiny 在眩光五级因此下降 9.7 pp。
  这是“复原目标 ≠ 判别目标”的直接证据，也说明任何只做光度校正的方法上限有限。
- 局部反事实恢复给出清晰的位置结构：把 3×3 网格中**中心一格**换回真实干净内容，可恢复
  54.7%/62.3%/75.4% 的可恢复错误（ResNet-50/ConvNeXt-Tiny/Swin-Tiny，雾+眩光五级），
  其余八格平均只有 20.3%/28.9%/33.1%；中心格把该条件的准确率从 47.0/65.7/63.1 提到
  70.3/80.1/83.1，补上干净-退化缺口的 54.4%/65.5%/81.3%。
- 但这是**用真实干净内容替换**得到的上界，不是任何算法可自动达到的结果；中心格外推仍会
  引入 0.7%–6.6% 的稳定样本错误诱导。它证明“中心区域含有可恢复证据”，不证明现有复原器
  能恢复它。
- 因此下一批不做“先把复原网络接上分类器”的串联方案；优先做 C1（局部退化增强）与
  C2/C3（带塌缩防护的联合训练消融），并把 A3/A5 的学习式复原限制为零训练、同数据对照。

## 1. 指标定义

- 准确率：正确样本数 / 236。Macro-F1：8 类 F1 的未加权平均，分母为各类。
- `Δ准确率(pp)`：条件准确率 − 同模型同条件 identity 准确率，乘 100。
- `rescued`：identity 预测错误、该方法预测正确的样本数；`hurt`：identity 正确、该方法
  错误的样本数。分母分别为 identity 错误数与正确数。
- 可恢复集合（B）：退化预测错误且干净预测正确的样本；雾+眩光五级为 106/61/65 个。
- `rescue@cell`：可恢复集合中，把该格替换为真实干净内容后预测正确的比例。
- `error induction`：退化与干净都正确、替换后错误的样本 / 两者都正确的样本。
- `margin`：softmax 首位概率减次位概率；`margin_delta_recoverable` 只在可恢复集合上平均。

## 2. A：传统复原与预测融合

格式为“准确率 (Δ pp; rescued/hurt)”。`identity` 在干净条件已逐样本对齐既有评测
（脚本断言，不一致即终止）；四个退化条件的 identity 预测也额外与既有评测逐样本比对，
3 个模型 × 4 个条件共 12 组全部 0 差异，说明探测管线与训练/评测管线一致。

{table(a_rows, ["模型", "条件", "identity", "Macro-F1", "CLAHE", "饱和填充", "融合 CLAHE", "融合填充"])}

观察：

- CLAHE 在雾+云五级（ResNet-50 +3.4）和 shifted 雾+眩光五级（Swin +3.8）有帮助，但在
  干净图稳定下降 3.4–3.8 pp，在 matched 雾/眩光上多为负。
- 饱和填充在眩光相关条件下伤害最大（ConvNeXt 眩光 −9.7、雾+眩光 −10.6），因为它把
  40%–62% 的已裁剪区域换成邻域颜色，破坏了剩余结构。
- 概率融合通常把 CLAHE 的损害减半，但无法把净收益稳定转正；干净条件仍有 −0.4 至 −2.1。
- 结论：这两种方法都不能作为新模块的组件；它们只能作为“光度校正类复原”的下界对照。

## 3. B：局部反事实恢复

中心格为 3×3 网格的正中一格（约内容面积 1/9）。中心格与其余八格均值为：

{table(b_rows, ["模型", "条件", "可恢复数", "退化", "干净", "中心 rescue", "其余均值", "中心后准确率", "缺口闭合", "中心 error induction", "中心 margin Δ"])}

3×3 恢复率网格（雾+眩光五级，行从上到下、列从左到右）：

{grid_text}

- 中心格的优势在三个架构上一致，且随退化类型变化：雾+眩光最强（2.2–2.8 倍于其余均值），
  眩光次之，雾最弱且可恢复样本少（6–23 个，噪声大）。
- 中心格替换后准确率大幅回升，但离干净仍有距离（ResNet-50 70.3 vs 89.8），说明中心之外
  还有可恢复信息，或者中心格只覆盖了船体的一部分。
- 中心格替换对稳定样本的错误诱导为 0.7%–6.6%，低于其带来的恢复量；但这是同图真实内容，
  不含复原伪影。

## 4. 机制解释与边界

- 眩光/雾+眩光的饱和比例说明大量像素的信息已被裁剪；PSNR/SSIM 可以变好而分类不变或变差，
  因此 A 线只以分类增益为准。
- B 的位置结构遮挡实验互相印证：中心遮挡伤害最大，中心恢复收益最大。两者都指向“中心
  判别证据”而不是全局色彩统计。
- 局限：3×3 网格是图像区域，不是船体部件；中心格也覆盖背景；替换使用真实干净像素，是
  分布外刺激与上界，不是可实现方法；单种子、236 张测试图，未做显著性检验。
- 测试集已多次用于诊断。后续 C 线的方法选择只在验证集做，测试集冻结到方法定型后再用
  三种子评测。

## 5. 对下一批的含义

1. C1：在组合增强中加入 center/border 局部退化，直接针对中心证据损失；与组合基线同数据、
   同种子配对比较。
2. C2：干净/退化预测一致性必须带 margin 与类别分布监控，防止“全部预测同一类”的平凡解。
3. C3：特征一致性只在未饱和区域加权，权重由输入饱和图下采样得到，不用可学习门控。
4. A3/A5（DehazeFormer/PromptIR 等）只做零训练推理对照；若不能在干净不降的前提下超过
   A 线融合对照，不进入训练。

## 复现

```bash
.venv/bin/python scripts/restoration_probe.py
.venv/bin/python scripts/local_recovery.py
.venv/bin/python scripts/route_ab_report.py
```
"""
    (PROJECT / "reports/ROUTE_AB_ROUND1.md").write_text(text)
    print(json.dumps({"restoration_conditions": len(a["conditions"]),
                      "local_conditions": len(b["conditions"]), "models": len(ORDER)}))


if __name__ == "__main__":
    main()
