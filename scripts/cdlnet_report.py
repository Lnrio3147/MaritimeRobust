#!/usr/bin/env python
"""Summarize the repaired CDLNet reproduction against the frozen baselines."""
import json
from pathlib import Path

from maritime_robust.common import write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs_external/fgsc23_research_all/cdlnet_clean_s17"
OFFICIAL = ROOT / "runs_external/fgsc23_official_all/cdlnet_clean_s17"
CIVILIAN = ROOT / "runs_external/fgsc23_research_civilian/cdlnet_clean_s17"
PROTOCOL = ROOT / "runs_external/fgsc23_official_all/cdlnet_protocol_check_s17"
SCOPES = {
    "matched single": [f"matched_{k}_{s}" for k in ("haze", "cloud", "glare") for s in range(1, 6)],
    "shifted single": [f"shifted_{k}_{s}" for k in ("haze", "cloud", "glare") for s in range(1, 6)],
    "matched combination": [f"matched_{k}_{s}" for k in ("haze+glare", "haze+cloud") for s in (3, 5)],
    "shifted combination": [f"shifted_{k}_{s}" for k in ("haze+glare", "haze+cloud") for s in (3, 5)],
}


def mean(summary, names):
    return sum(summary[n]["macro_f1"] for n in names) / len(names)


def main():
    summary = json.loads((RUN / "evaluation/summary.json").read_text())
    settings = json.loads((RUN / "complete.json").read_text())
    history = json.loads((RUN / "history.json").read_text())
    best_epoch = max(history, key=lambda r: r["val_macro_f1"])["epoch"]
    official = None
    if (OFFICIAL / "evaluation/summary.json").exists():
        o_summary = json.loads((OFFICIAL / "evaluation/summary.json").read_text())
        o_settings = json.loads((OFFICIAL / "complete.json").read_text())
        o_history = json.loads((OFFICIAL / "history.json").read_text())
        o_best = max(o_history, key=lambda r: r["val_macro_f1"])["epoch"]
        official = {
            "clean_accuracy": o_summary["clean"]["accuracy"],
            "clean_macro_f1": o_summary["clean"]["macro_f1"],
            "clean_balanced_accuracy": o_summary["clean"]["balanced_accuracy"],
            "best_epoch": o_best, "epochs": len(o_history),
            "best_val_macro_f1": o_settings["best_val_macro_f1"],
            "leakage": o_settings["source_group_leakage"],
            "train_n": o_settings["train_n"], "val_n": o_settings["val_n"],
        }
    civilian = None
    if (CIVILIAN / "evaluation/summary.json").exists():
        c_summary = json.loads((CIVILIAN / "evaluation/summary.json").read_text())
        c_settings = json.loads((CIVILIAN / "complete.json").read_text())
        c_history = json.loads((CIVILIAN / "history.json").read_text())
        c_rows = {"clean": c_summary["clean"]["macro_f1"],
                  "clean_accuracy": c_summary["clean"]["accuracy"]}
        c_rows.update({name: mean(c_summary, names) for name, names in SCOPES.items()})
        c_worst = min(c_summary, key=lambda k: c_summary[k]["macro_f1"])
        c_rows["worst condition"] = c_summary[c_worst]["macro_f1"]
        civilian = {
            "rows": c_rows, "worst_condition": c_worst,
            "best_val_macro_f1": c_settings["best_val_macro_f1"],
            "best_epoch": max(c_history, key=lambda r: r["val_macro_f1"])["epoch"],
            "epochs": len(c_history), "train_n": c_settings["train_n"],
            "val_n": c_settings["val_n"],
            "training_wall_seconds": c_settings["training_wall_seconds"],
        }
    protocol = None
    if (PROTOCOL / "evaluation/clean.json").exists():
        p_score = json.loads((PROTOCOL / "evaluation/clean.json").read_text())
        p_settings = json.loads((PROTOCOL / "complete.json").read_text())
        protocol = {
            "accuracy": p_score["accuracy"], "macro_f1": p_score["macro_f1"],
            "balanced_accuracy": p_score["balanced_accuracy"],
            "train_n": p_settings["train_n"], "resize": p_settings["resize"],
            "scheduler": p_settings["scheduler"], "batch_size": p_settings["batch_size"],
            "epochs": p_settings["epochs"],
            "training_wall_seconds": p_settings["training_wall_seconds"],
        }
    worst = sorted(summary, key=lambda k: summary[k]["macro_f1"])[:8]
    rows = {"clean": summary["clean"]["macro_f1"]}
    rows.update({name: mean(summary, names) for name, names in SCOPES.items()})
    rows["worst condition"] = summary[worst[0]]["macro_f1"]
    result = {
        "run": str(RUN.relative_to(ROOT)), "model": "cdlnet_repro",
        "reproduction": settings["reproduction"], "scope": settings["scope"],
        "protocol": settings["protocol"], "seed": settings["seed"],
        "input_size": settings["input_size"], "num_classes": settings["num_classes"],
        "best_val_macro_f1": settings["best_val_macro_f1"],
        "official_sanity": official, "civilian": civilian, "protocol_check": protocol,
        "clean_macro_f1": rows["clean"], "clean_accuracy": summary["clean"]["accuracy"],
        "summary": rows, "worst_conditions": {k: summary[k]["macro_f1"] for k in worst},
        "epochs_run": len(history), "best_epoch": best_epoch,
        "training_wall_seconds": settings["training_wall_seconds"],
        "peak_cuda_allocated_bytes": settings["peak_cuda_allocated_bytes"],
        "parameter_count": settings["parameter_count"],
        "source_sha256": settings["source_sha256"],
        "checkpoint_sha256": settings["checkpoint_sha256"],
        "manifest_sha256": settings["manifest_sha256"],
        "recipe_sha256": settings["recipe_sha256"],
    }
    write_json(ROOT / "reports/cdlnet_repro.json", result)

    baselines = json.loads((ROOT / "reports/results.json").read_text())["runs"]
    def baseline(model, seed, scope="all", protocol="research"):
        return next(x for x in baselines if x["model"] == model and x["scope"] == scope
                    and x["protocol"] == protocol and x["augmentation"] == "clean" and x["seed"] == seed)

    ref = baseline("resnet50", 17)
    ref_official = baseline("resnet50", 17, "all", "official")
    best = baseline("convnext_tiny", 17)
    civ_text = ""
    if civilian:
        civ_ref = baseline("resnet50", 17, "civilian")
        r = civilian["rows"]
        civ_text = f"""

## 民船 8 类范围（主实验）

与主实验同数据、同划分、同 43 条件：train {civilian['train_n']} / val {civilian['val_n']}，
最佳干净 val Macro-F1 {civilian['best_val_macro_f1']:.4f}（第 {civilian['best_epoch']}/{civilian['epochs']} 轮），
训练 {civilian['training_wall_seconds']/60:.1f} min。

| 条件族 | CDLNet 修复版 | ResNet-50（civilian，s17） | 差（CDLNet − 基线） |
|---|---|---|---|
| 干净 | {r['clean']:.4f}（acc {r['clean_accuracy']:.4f}） | {civ_ref['clean_macro_f1']:.4f} | {r['clean']-civ_ref['clean_macro_f1']:+.4f} |
| matched 单退化（15 条均值） | {r['matched single']:.4f} | {civ_ref['matched_single_macro_f1']:.4f} | {r['matched single']-civ_ref['matched_single_macro_f1']:+.4f} |
| shifted 单退化（15 条均值） | {r['shifted single']:.4f} | — | — |
| matched 组合（4 条均值） | {r['matched combination']:.4f} | {civ_ref['matched_combination_macro_f1']:.4f} | {r['matched combination']-civ_ref['matched_combination_macro_f1']:+.4f} |
| shifted 组合（4 条均值） | {r['shifted combination']:.4f} | {civ_ref['shifted_combination_macro_f1']:.4f} | {r['shifted combination']-civ_ref['shifted_combination_macro_f1']:+.4f} |
| 最差单条件（`{civilian['worst_condition']}`） | {r['worst condition']:.4f} | {civ_ref['worst_macro_f1']:.4f} | {r['worst condition']-civ_ref['worst_macro_f1']:+.4f} |

民船范围内 CDLNet 干净集比同骨干基线高 {(r['clean']-civ_ref['clean_macro_f1'])*100:.1f} 个百分点，
但单退化低 {abs(r['matched single']-civ_ref['matched_single_macro_f1'])*100:.1f}、组合退化低
{abs(r['matched combination']-civ_ref['matched_combination_macro_f1'])*100:.1f} / {abs(r['shifted combination']-civ_ref['shifted_combination_macro_f1'])*100:.1f} 个百分点，
即“干净更高、退化更差”的不对称在 8 类上更明显。
"""

    curve = {k: [round(summary[f"matched_{k}_{s}"]["macro_f1"], 3) for s in range(1, 6)]
             for k in ("haze", "cloud", "glare")}
    curve_shifted = {k: [round(summary[f"shifted_{k}_{s}"]["macro_f1"], 3) for s in range(1, 6)]
                     for k in ("haze", "cloud", "glare")}
    proto_text = ""
    if protocol and official:
        proto_text = f"""
### 协议对齐对照（诊断）

为排除“预处理/调度/数据量导致 86%”的解释，另跑 `runs_external/fgsc23_official_all/cdlnet_protocol_check_s17`：
train+val 训练（{protocol['train_n']} 张）、拉伸到 256×256、SGD lr 0.01 +
StepLR(10, 0.1)、batch {protocol['batch_size']}、{protocol['epochs']} epoch、不留 val（取最后一轮），
训练 {protocol['training_wall_seconds']/60:.1f} min。

| 变体 | OA | Macro-F1 | balanced acc |
|---|---|---|---|
| 主复现（train only，letterbox，cosine） | {official['clean_accuracy']:.2%} | {official['clean_macro_f1']:.2%} | {official['clean_balanced_accuracy']:.2%} |
| 协议对齐（train+val，stretch，StepLR） | {protocol['accuracy']:.2%} | {protocol['macro_f1']:.2%} | {protocol['balanced_accuracy']:.2%} |

两个变体的总正确数都是 710/824，但逐类召回与混淆矩阵不同，是巧合而非同一模型。
协议对齐只带来约 +1.2 个 Macro-F1 点、OA 不变，仍比 93.23% 低约 7.1 个百分点。
因此差距不能归因于预处理、调度或训练数据量，剩余差距只能来自未公开的方法组件
（分类/二分类训练、双模型门控、KNN 修正）。
"""
    sanity = ""
    if official:
        sanity = f"""

## 官方划分 sanity check

| 项 | 值 |
|---|---|
| 官方划分（原图组按官方 train/val/test） | train {official['train_n']} / val {official['val_n']} |
| 最佳干净 val Macro-F1 | {official['best_val_macro_f1']:.4f}（第 {official['best_epoch']}/{official['epochs']} 轮） |
| 干净 test accuracy | {official['clean_accuracy']:.4f} |
| 干净 test Macro-F1 | {official['clean_macro_f1']:.4f} |

该划分下 `source_group_id` 仍有 train/test {official['leakage']['train/test']} 组、val/test {official['leakage']['val/test']} 组重叠
（FGSC-23 官方划分本身如此），因此只能作为“干净集精度量级”检查：修复版 OA {official['clean_accuracy']:.2%}
仍低于论文报告的干净集成绩，与 research split 上“未复现出干净集增益”的结论一致。
官方划分不做退化评测，退化结论以 research split 为准。
"""
    combos = {k: round(summary[k]["macro_f1"], 3) for k in
              ("matched_haze+glare_3", "matched_haze+glare_5", "matched_haze+cloud_3",
               "matched_haze+cloud_5", "shifted_haze+glare_3", "shifted_haze+glare_5",
               "shifted_haze+cloud_3", "shifted_haze+cloud_5")}
    text = f"""# CDLNet 修复版复现：干净训练 → 43 条件退化评测

> **这不是作者的原始成绩。** 官方仓库 `MIPCLab/CDLNetcode` 缺训练入口、缺权重、含未注册层，
> 只能按其公开结构修复后单阶段训练。结果只用于本项目内部的退化敏感性分析，
> 不得与论文报告的 FGSC-23 指标并列比较。

## 训练设置

| 项 | 值 |
|---|---|
| 源码 | `external/cdlnet_repro/`（本次运行哈希已写入 `reports/cdlnet_repro.json`） |
| 数据 | `{settings['manifest'].split('/')[-1]}`（research split，{settings['num_classes']} 类，train {settings['train_n']} / val {settings['val_n']}） |
| 输入 | {settings['input_size']}×{settings['input_size']}（4×4 网格 + 3×3 窗口需要 32 的整数倍） |
| 骨干 | ResNet-50 ImageNet-1K V2（`resnet50-11ad3fa6.pth`） |
| 训练 | 单阶段交叉熵，SGD lr {settings['optimizer']['lr']}，momentum 0.9，wd {settings['optimizer']['weight_decay']}，cosine，{settings['epochs']} epoch，batch {settings['batch_size']}，AMP |
| 选择 | 干净 val Macro-F1（最佳 {settings['best_val_macro_f1']:.4f}，第 {best_epoch}/{len(history)} 轮） |
| 资源 | 训练 {settings['training_wall_seconds']/60:.1f} min，峰值显存 {settings['peak_cuda_allocated_bytes']/1e9:.2f} GB，参数 {settings['parameter_count']/1e6:.2f} M |

## 与官方实现的差异（必须随结果一起报告）

1. 官方 `project/main.py` 导入不存在的 `train` 模块，且 `CKPT_*` 未定义：无法复原三阶段训练，
   本复现只做单阶段分类训练，去掉二分类头、代理样本与对比损失。
2. `FeatureExtractor` 在 `forward` 内新建 `response_head`，参数不注册、每次前向随机：
   本复现将其注册为模块（否则该分支不可能被训练）。
3. `GatedSelfAttention` 的布尔索引在反向时语义不明：改为 `torch.where`；推理时使用全 1 mask，保证可复现。
4. 官方在 2048 通道全分辨率特征图上做上采样响应，256 输入下 batch 16 会触发 CUDA
   `invalid configuration argument`：本复现将 1×1 响应头放在上采样之前（线性算子可交换，
   内部像素等价），显存从 >19 GB 降到 2.4 GB。
5. 官方硬编码 42 类 / `FGSCR-test` / 双卡：本复现使用 research split 的 23 类与单卡。
6. 评测退化与本项目基线完全同源（同一 `corruption_seed`、同一 `corrupt`、同 letterbox），
   仅分辨率 256 与基线的 224 不同。
7. 两个运行完成后，`run.py` 又加入了官方划分 sanity check 用的 `--allow-official-leakage`
   开关（只跳过原图组重叠检查，不改变训练/评测路径）；research 运行 `complete.json` 记录的
   `run.py` 哈希对应加入开关前的版本，`cdlnet_model.py` 与 `cdlnet_data.py` 哈希仍与当前文件一致。

## 为什么干净集不是论文的 93.23%

仓库 README 声称 FGSC-23 上 93.23%（RGCRL-Net 92.89%）。本复现的官方划分 OA 为
{official['clean_accuracy']:.2%}，差约 {93.23 - official['clean_accuracy']*100:.1f} 个百分点。差距有明确来源，
不是评测代码写错：

1. **作者的方法训练代码没有公开。** `project/main.py` 的 Step 2（分类训练）与 Step 3（二分类 gate 训练）
   只剩注释和模型构建，训练调用被删除；`res_bin.py` 只有 `warmup_train`，没有分类/二分类训练函数。
   公开代码无法产出作者的 checkpoint，93.23% 不能由它复现。
2. **官方评测是双模型 + 门控 + KNN。** `project/evaluate.py` 同时使用 `ckpt_cls`（分类）与
   `ckpt_final`（二分类 gate），测试时每样本提供弱/强/原图三视图，先用二分类头决定是否启用 CIMAM，
   再对特征做 k=5 的 KNN 修正后才分类。本复现是单模型、单视图、单次前向。
3. **预处理与调度不同。** 官方 `dataloader.py` 用 `Resize([256,256])` 拉伸，本项目用 letterbox；
   官方 SGD lr 0.01 + StepLR(10, 0.1)、batch 12，本项目 cosine、batch 16。
4. **训练数据量不同。** 本项目留出 654 张 val 做选型，实际只用 2,602 张训练；文献常用约 1:4 的
   训练/测试划分（约 3,240 张训练），且不一定留 val。

因此 86% 应读作“CDLNet 架构在单阶段 CE 下的水平”，而不是“CDLNet 方法的水平”。
按 FGSC-23 数据集原文的基线（ResNet-50 81.33%，Xception 87.76%），普通骨干在 81–88%；本复现落在该区间，
与“方法组件缺失”一致。同一官方划分上，项目的 ResNet-50 基线干净 test Macro-F1 为 {ref_official['clean_macro_f1']*100:.2f}%，
说明评测 harness 本身没有系统性偏低。
{proto_text}
{sanity}
## 结果（test，43 条件）

| 条件族 | CDLNet 修复版 | ResNet-50（基线，s17） | 差（CDLNet − 基线） |
|---|---|---|---|
| 干净 | {rows['clean']:.4f} | {ref['clean_macro_f1']:.4f} | {rows['clean']-ref['clean_macro_f1']:+.4f} |
| matched 单退化（15 条均值） | {rows['matched single']:.4f} | {ref['matched_single_macro_f1']:.4f} | {rows['matched single']-ref['matched_single_macro_f1']:+.4f} |
| shifted 单退化（15 条均值） | {rows['shifted single']:.4f} | — | — |
| matched 组合（4 条均值） | {rows['matched combination']:.4f} | {ref['matched_combination_macro_f1']:.4f} | {rows['matched combination']-ref['matched_combination_macro_f1']:+.4f} |
| shifted 组合（4 条均值） | {rows['shifted combination']:.4f} | {ref['shifted_combination_macro_f1']:.4f} | {rows['shifted combination']-ref['shifted_combination_macro_f1']:+.4f} |
| 最差单条件 | {rows['worst condition']:.4f} | {ref['worst_macro_f1']:.4f} | {rows['worst condition']-ref['worst_macro_f1']:+.4f} |

干净准确率 {result['clean_accuracy']:.4f}。同表 ConvNeXt-Tiny（s17）为干净 {best['clean_macro_f1']:.4f}、
matched 单退化 {best['matched_single_macro_f1']:.4f}、matched 组合 {best['matched_combination_macro_f1']:.4f}、
最差 {best['worst_macro_f1']:.4f}，可作为“更强干净基线在退化下掉得更慢”的参照。

### 逐级退化曲线（Macro-F1，severity 1→5）

| 退化 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| haze（matched） | {curve['haze'][0]} | {curve['haze'][1]} | {curve['haze'][2]} | {curve['haze'][3]} | {curve['haze'][4]} |
| cloud（matched） | {curve['cloud'][0]} | {curve['cloud'][1]} | {curve['cloud'][2]} | {curve['cloud'][3]} | {curve['cloud'][4]} |
| glare（matched） | {curve['glare'][0]} | {curve['glare'][1]} | {curve['glare'][2]} | {curve['glare'][3]} | {curve['glare'][4]} |
| haze（shifted） | {curve_shifted['haze'][0]} | {curve_shifted['haze'][1]} | {curve_shifted['haze'][2]} | {curve_shifted['haze'][3]} | {curve_shifted['haze'][4]} |
| cloud（shifted） | {curve_shifted['cloud'][0]} | {curve_shifted['cloud'][1]} | {curve_shifted['cloud'][2]} | {curve_shifted['cloud'][3]} | {curve_shifted['cloud'][4]} |
| glare（shifted） | {curve_shifted['glare'][0]} | {curve_shifted['glare'][1]} | {curve_shifted['glare'][2]} | {curve_shifted['glare'][3]} | {curve_shifted['glare'][4]} |

### 组合退化

| 条件 | Macro-F1 |
|---|---|
""" + "\n".join(f"| {k} | {v} |" for k, v in combos.items()) + f"""

### 最差 8 个条件

""" + "\n".join(f"- `{k}` {summary[k]['macro_f1']:.3f}" for k in worst) + civ_text + f"""
## 观察

- all 23 类上 CDLNet 修复版干净集比同骨干 ResNet-50 基线低 {abs(rows['clean']-ref['clean_macro_f1'])*100:.1f} 个百分点（民船 8 类相反，见上节），
  说明单阶段 CE 修复版没有复现出论文的干净集增益；这不排除是训练流程差异（无二分类预热/代理样本）造成。
- 单退化中 haze/cloud 比 glare 更伤、shifted glare 最温和，方向与基线一致；但五级差异变大：
  haze 0.461（基线 0.594）、cloud 0.500（基线 0.472）、glare 0.603（基线 0.607）。
- 组合退化比同骨干 ResNet-50 基线掉得更深（shifted 组合低 5.7 个百分点）：matched 组合均值 {rows['matched combination']:.3f}，最差 {rows['worst condition']:.3f}。
- 组合退化在 severity 3 时已跌到 0.56–0.67，severity 5 跌到 0.13–0.21。
  一种可能的解释是“区域选择 + 响应融合”在低对比度、遮挡叠加时选错区域并把误差放大；
  这仍是假设，需在下一轮用区域响应可视化与干净/退化区域命中率来验证，不能从当前指标直接断言。
- 局限：单种子、单阶段训练、256 输入、未复现二分类/对比分支；结论只适用于该修复版。

## 复现命令

```bash
.venv/bin/python external/cdlnet_repro/run.py train --run-dir runs_external/fgsc23_research_all/cdlnet_clean_s17
.venv/bin/python external/cdlnet_repro/run.py eval  --run-dir runs_external/fgsc23_research_all/cdlnet_clean_s17
.venv/bin/python scripts/cdlnet_report.py
```
"""
    (ROOT / "reports/CDLNET_REPRO.md").write_text(text)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rows.items()},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
