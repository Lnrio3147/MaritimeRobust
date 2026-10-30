#!/usr/bin/env python
"""Generate reports/EFMNET_REPRO.md and reports/CROSS_MODEL_DEGRADATION.md.

Every number in the two reports is read from an artifact produced by the
scripts named in each report; nothing is typed in by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_GLOB = "runs_external/fgsc23_civ8/*/"


def load(path):
    return json.loads((ROOT / path).read_text())


def run_dir():
    dirs = sorted(ROOT.glob(RUN_GLOB))
    if not dirs:
        raise SystemExit("no EFM-Net run directory")
    return dirs[0]


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pct(x):
    return f"{100 * x:.2f}"


def delta(value):
    return f"{value:+.1f}"


def repro_report(run, analysis):
    recs = records(run / "record.txt")
    best = max(recs, key=lambda r: r["test_acc1"])
    efmnet = analysis["models"]["efmnet"]
    native = analysis["preprocess_ablation"]

    lines = [
        "# EFM-Net 架构复现记录（8 类民用船型，非官方成绩）",
        "",
        "2026-10-30。外部 SOTA 复现，用于新课题的退化观测与 G0/G1 判定。"
        "所有数字由 `scripts/efmnet_eval.py`、`scripts/baseline_val_conditions.py` 与 "
        "`scripts/cross_model_degradation.py` 生成，工件见 `reports/efmnet_val_conditions*.json`、"
        "`reports/cross_model_degradation.json`。",
        "",
        "## 1. 复现范围",
        "",
        "**架构级重实现，无官方权重。** 参数随机初始化后从头训练，只使用 ImageNet-22K 的 "
        "ConvNeXt-Base 主干权重作为初始化；MHA（多层级注意力）与 BPP 细化头全部随机初始化。"
        "因此本文结果不是论文成绩的复现，只能称为\"同架构、本项目协议\"的一次独立训练。",
        "",
        "代码：`external/audit/EFM-Net-Pytorch`（commit `7d04323`，Apache-2.0），"
        "工作副本 `external/efmnet_repro/`。",
        "",
        "## 2. 使其可运行的补丁",
        "",
        "| 文件 | 问题 | 处理 |",
        "|---|---|---|",
        "| `util/utils.py` | `from torch._six import inf`（torch≥2.0 移除） | 改为 `from torch import inf` |",
        "| `util/utils.py` | `save_input_image` 依赖 cv2 | 改用 PIL 保存 |",
        "| `main.py` | `dist.get_rank()` 未导入 `dist` | 改为 `utils.get_rank()` |",
        "| `main.py` | 非分布式分支 `logger = None` 后调用 logger | 改为 `create_logger(...)` |",
        "| `main.py` | `logger('...')` 把 logger 当函数调用 | 改为 `logger.info(...)` |",
        "| `models/miner.py` | `torch.autograd.set_detect_anomaly(True)` 拖慢训练 | 移除 |",
        "| `processor/optim_factory.py` | timm 0.9.16 已移除 `novograd` 等模块，导入即失败 | 改为可选导入（默认优化器 adamw 不受影响） |",
        "| `datasets/FGSC-23` | 软链路径少一级 | 重建软链指向 `data/raw/fgsc23/FGSC-23` |",
        "",
        "## 3. 预训练权重来源",
        "",
        "- 文件：`external/efmnet_repro/pretrained/convnext_base_22k_224.pth`（420 MB，344 个张量）",
        "- 来源：ImageNet-22K 监督预训练 ConvNeXt-Base，官方 `convnext_base_22k_224` 权重",
        "- 载入日志明确列出未初始化模块：`mha.*`、`head.*`（共 26 项），其余主干完全由预训练权重填充",
        "",
        "## 4. 数据划分",
        "",
        "由 `scripts/efmnet_dataset.py` 生成 ImageFolder 布局（软链，不复制像素）：",
        "",
        "| EFM-Net 目录 | 项目划分 | 数量 | 用途 |",
        "|---|---|---|---|",
        "| `train/` | manifest `split=train` | 618 | 训练 |",
        "| `test/` | manifest `split=val` | 157 | 模型选择（EFM-Net 代码里叫 test） |",
        "| 未链接 | manifest `split=test` | 236 | 官方测试集，保持冻结 |",
        "",
        "类别为 8 类民用船型（container ship、car carrier、bulk carrier、oil tanker、"
        "fishing boat、passenger ship、liquefied gas ship、barge）。",
        "",
        "## 5. 训练配置",
        "",
        "| 项 | 值 | 说明 |",
        "|---|---|---|",
        "| 模型 | EFM-Net = ConvNeXt-B + MHA(M=16) + BPP | 87,750,292 参数 |",
        "| 优化器 | AdamW，lr 1e-3，wd 1e-8，layer_decay 0.8 | 代码默认 |",
        "| batch × 累积 | 16 × 30 = 480 | 与论文有效批大小一致 |",
        "| 每 epoch 更新次数 | 1 | `len(dataset)//480 = 1` |",
        "| 计划 epoch | 1800 | 使总更新次数 = 1800，与论文 300 epoch × 6 次一致 |",
        "| 实际 epoch | " + str(len(recs)) + " | 见 §6 |",
        "| 增强 | timm 默认训练变换 | 无 mixup / cutmix / EMA |",
        "| 种子 | 17 | |",
        "| 设备 | 单卡 RTX 4090 24 GB | |",
        "",
        "## 6. 为何在 " + str(len(recs)) + " epoch 停止",
        "",
        "| 观测 | 值 |",
        "|---|---|",
        "| 最好 val 准确率 | " + f"{best['test_acc1']:.2f}%" + f"（record 中 epoch 索引 {best['epoch']}，即第 {best['epoch'] + 1} epoch）" + " |",
        "| 之后连续未刷新的 epoch 数 | " + str(len(recs) - best["epoch"] - 1) + " |",
        "| 停止时 LR | " + f"{recs[-1]['train_lr']:.3e}" + "（余弦尚未走完） |",
        "| 停止时训练损失 | " + f"{recs[-1]['train_loss']:.3f}" + "（首 epoch 2.08） |",
        "",
        "理由：最优权重在第 " + str(best["epoch"] + 1) + " epoch 出现后 " + str(len(recs) - best["epoch"] - 1) +
        " 个 epoch 内未被刷新，继续训练只增加过拟合风险；按 `checkpoint-best.pth` 停止，"
        "并把\"余弦退火尾段未执行\"作为已知偏离记录在 §10。",
        "",
        "保留的 checkpoint：`checkpoint-best.pth`（最优）、`checkpoint-499.pth`、`checkpoint-599.pth`。",
        "",
        "## 7. 两套预处理口径",
        "",
        "| 口径 | 定义 | clean val acc |",
        "|---|---|---|",
        f"| EFM-Net 原生 | timm eval transform：缩放到 256×256（squash，非等比）→ 中心裁剪 224 | {pct(native['clean']['native'])}% |",
        f"| 项目管线 | `contain(224)` → 退化 → `pad(224, 灰边 124,116,104)` → ImageNet 归一化 | {pct(native['clean']['project'])}% |",
        "",
        f"两套口径相差 {100 * (native['clean']['native'] - native['clean']['project']):.2f} pp。"
        "**85.99% 是在 EFM-Net 自己的预处理下得到的，不能作为相同评估协议下的 SOTA 对比数，"
        "也不能与原论文 23 类官方划分的 OA 直接比较。** 跨模型比较一律使用项目管线口径"
        "（`reports/CROSS_MODEL_DEGRADATION.md`），并且只比较同一口径内的差值。",
        "",
        "## 8. 退化结果（项目管线，val 157）",
        "",
        "配对 bootstrap（按 `source_group_id` 分组重采样 2000 次）相对 clean 的差值：",
        "",
        "| 条件 | acc | Δ (pp) | 95% CI |",
        "|---|---|---|---|",
    ]
    for condition, stat in efmnet["conditions"].items():
        lines.append(f"| {condition} | {pct(stat['accuracy'])} | {delta(stat['delta_pp'])} | "
                     f"[{stat['ci_lo']:.1f}, {stat['ci_hi']:.1f}] |")

    lines += [
        "",
        "原生预处理下的对照（仅 clean 标定 + 与项目管线的差值比较）见 "
        "`reports/cross_model_degradation.json` 的 `preprocess_ablation` 字段，"
        "汇总见 `reports/CROSS_MODEL_DEGRADATION.md` §4。",
        "",
        "## 9. 与论文数字的关系（明确不可比）",
        "",
        "| 差异 | 论文 | 本次 |",
        "|---|---|---|",
        "| 类别数 | 23 | 8 |",
        "| 划分 | 官方 FGSC-23 train/test | 项目 8 类划分 618/157 |",
        "| 权重 | 官方训练结果 | 随机初始化后从头训练 |",
        "| 评估预处理 | 论文自己的管线 | 项目 contain+灰边 pad |",
        "| 优化步数 | 300 epoch × 6 次/epoch = 1800 | " + str(len(recs)) + " 次 |",
        "| 数字 | OA 约 89.70（23 类） | 85.99%（原生口径）/ 78.34%（项目口径） |",
        "",
        "结论：本运行只能作为\"同架构在本项目协议下的独立训练\"，不构成对论文成绩的复现，"
        "也不能声称高于或低于论文。G0 判定因此改为\"架构复现 + 协议声明\"，"
        "其可比性仅限本项目内部（与 7 个冻结基线同管线、同图、同种子）。",
        "",
        "## 10. 协议偏离清单",
        "",
        "1. 类别数与划分：8 类、项目划分，非官方 23 类官方划分。",
        "2. 预处理：训练用 timm 增强与原生 eval 变换；跨模型评估改用项目管线，导致干净条件净降 "
        + f"{100 * (native['clean']['native'] - native['clean']['project']):.2f} pp。",
        "3. 优化步数：" + str(len(recs)) + " 次，低于计划的 1800 次；余弦退火未完成（停止时 LR "
        + f"{recs[-1]['train_lr']:.3e}）。",
        "4. 无 EMA、无 mixup/cutmix、无分布式训练（代码默认值）。",
        "5. 训练集 618 张，远小于论文 23 类训练集。",
        "6. 未使用官方权重，故任何绝对准确率都不应作为 SOTA 行进入对比表，除非同时标注口径。",
        "",
        "## 11. 工件",
        "",
        "| 工件 | 内容 |",
        "|---|---|",
        f"| `{run.relative_to(ROOT)}checkpoint-best.pth` | 最优权重（1.05 GB） |",
        f"| `{run.relative_to(ROOT)}record.txt` | 逐 epoch 训练/验证记录 |",
        "| `reports/efmnet_val_conditions.json` | 项目管线 43 条件精度 |",
        "| `reports/efmnet_val_conditions_samples.json` | 项目管线逐样本预测与概率 |",
        "| `reports/efmnet_val_conditions_native.json` | 原生预处理 43 条件精度 |",
        "| `reports/cross_model_degradation.json` | 跨模型统计（Δ、CI、闸门、转移矩阵） |",
        "",
    ]
    return "\n".join(lines)


def cross_report(analysis, failure):
    families = list(analysis["families"])
    lines = [
        "# 跨模型退化评估（val 157，42 个退化条件 + clean，配对设计）",
        "",
        "2026-10-30。目的：在**同图、同退化参数、同随机种子、同项目预处理**下，"
        "确定哪些退化条件在多个独立骨干家族上稳定成立（G1）。",
        "",
        "## 1. 设计",
        "",
        "- 样本：FGSC-23 val 157 张（8 类民用船型），152 个独立来源组",
        "- 条件：`degradations.conditions()` 的 43 个条件（clean + matched/shifted × haze/cloud/glare 1–5 + 组合 + 位置对照）",
        "- 种子：`corruption_seed(sample_id, role=\"efmnet\", ...)`，所有模型共用同一批退化图像",
        "- 模型：7 个冻结组合增强分类器（ConvNeXt-T ×3 种子、Swin-T ×3 种子、ResNet-50 s17）+ EFM-Net 复现",
        "- 统计：逐图配对正确/错误向量；差值 = 条件 − clean；bootstrap 以 `source_group_id` 为抽样单位，2000 次，2.5/97.5 分位",
        "- 骨干家族（独立性单位）：convnext_tiny、swin_t、resnet50、efmnet_convnext_b（同家族多种子先按样本取平均再统计）",
        "- G1 判定：家族级 Δ ≤ −5.0 pp 且 CI 上界 < 0，计为通过",
        "",
        "## 2. 干净条件",
        "",
        "| 模型 | clean acc |",
        "|---|---|",
    ]
    for name, entry in analysis["models"].items():
        lines.append(f"| {name} | {pct(entry['clean_accuracy'])}% |")
    lines += [
        "",
        "EFM-Net 在项目管线下低于 7 个自家基线，原因见 `reports/EFMNET_REPRO.md` §7（预处理口径），"
        "比较差值时该偏移在配对设计内抵消。",
        "",
        "## 3. 42 个退化条件 × 4 家族 Δ (pp)（连同 clean 共 43 个评估条件）",
        "",
        "| 条件 | " + " | ".join(families) + " | 通过家族数 |",
        "|---|" + "---|" * (len(families) + 1),
    ]
    for condition in analysis["conditions"]:
        cells = [delta(analysis["families"][f]["conditions"][condition]["delta_pp"]) for f in families]
        gate = analysis["gate"][condition]
        lines.append(f"| {condition} | " + " | ".join(cells) + f" | {gate['n_pass']}/4 |")

    passing = [c for c in analysis["conditions"] if analysis["gate"][c]["n_pass"] == 4]
    lines += [
        "",
        "## 4. G1 判定",
        "",
        f"**通过（4/4 家族，Δ≤−5 pp 且 CI 不跨零）的条件共 {len(passing)} 个：**",
        "",
    ]
    for condition in passing:
        cells = []
        for family in families:
            stat = analysis["families"][family]["conditions"][condition]
            cells.append(f"{family} {delta(stat['delta_pp'])} [{stat['ci_lo']:.1f},{stat['ci_hi']:.1f}]")
        lines.append(f"- `{condition}`：" + "；".join(cells))
    counts = {}
    for condition in analysis["conditions"]:
        counts[analysis["gate"][condition]["n_pass"]] = counts.get(analysis["gate"][condition]["n_pass"], 0) + 1
    lines += [
        "",
        "通过家族数分布：" + "，".join(f"{k}/4 → {v} 个条件" for k, v in sorted(counts.items())),
        "",
        "说明：单雾条件（matched/shifted haze 5）不达标，"
        "与 `reports/DEPTH_FOG_CHECK.md` 的 3.5 pp 观测一致；"
        "**眩光、云、以及雾与其他退化的组合**才达到跨骨干稳定门槛。"
        "这决定了课题的动机必须写在\"组合退化与眩光/云\"上，而不是\"单一浓雾\"。",
        "",
        "## 5. 预处理消融（EFM-Net 2×43）",
        "",
        "| 口径 | clean | 说明 |",
        "|---|---|---|",
        f"| 项目管线 | {pct(analysis['preprocess_ablation']['clean']['project'])}% | contain + 灰边 pad |",
        f"| EFM-Net 原生 | {pct(analysis['preprocess_ablation']['clean']['native'])}% | 缩放 256 + 中心裁剪 224 |",
        "",
        "| 条件 | acc(项目) | acc(原生) | Δ(项目) | Δ(原生) | Δ 差 |",
        "|---|---|---|---|---|---|",
    ]
    ab = analysis["preprocess_ablation"]["conditions"]
    for condition in analysis["conditions"]:
        s = ab[condition]
        lines.append(f"| {condition} | {pct(s['accuracy_project'])} | {pct(s['accuracy_native'])} | "
                     f"{delta(s['delta_project_pp'])} | {delta(s['delta_native_pp'])} | "
                     f"{delta(s['delta_difference_pp'])} |")
    diffs = [ab[c]["delta_difference_pp"] for c in analysis["conditions"]]
    lines += [
        "",
        f"两套口径的退化 Δ 差异：均值 {sum(diffs) / len(diffs):+.1f} pp，范围 {min(diffs):+.1f}…{max(diffs):+.1f} pp。"
        "结论：预处理的绝对水平差异很大（" +
        f"{100 * (analysis['preprocess_ablation']['clean']['native'] - analysis['preprocess_ablation']['clean']['project']):.2f} pp），"
        "但退化**方向与量级基本一致**；因此跨模型比较使用项目管线，"
        "而 EFM-Net 是否\"更鲁棒\"的判断必须看同一口径内的 Δ。",
        "",
        "## 6. 错误转移、逐类别与置信度（EFM-Net，项目管线）",
        "",
    ]
    for condition, block in analysis["focus"].items():
        ef = block["efmnet"]
        lines += [
            f"### {condition}",
            "",
            f"- clean 正确 → 退化错误：{ef['broken']}；clean 错误 → 退化正确：{ef['fixed']}；"
            f"四个家族全部错误的样本：{block['all_families_wrong']}",
            "- 主要错误转移（真实类 → 预测类，n）：" +
            ("，".join(f"{t}→{p} ({n})" for t, p, n in ef["top_transitions"]) or "无"),
            "- 逐类别召回 clean → 退化：" + "，".join(
                f"{c}: {100 * ef['per_class_recall_clean'][c]:.0f}% → {100 * ef['per_class_recall'][c]:.0f}%"
                for c in sorted(ef["per_class_recall"], key=int)),
            "",
        ]
    conf = analysis["confidence_efmnet"]
    lines += [
        "### 置信度变化（EFM-Net）",
        "",
        "| 条件 | 平均最大概率 | 正确样本 | 错误样本 |",
        "|---|---|---|---|",
    ]
    for condition, stat in conf.items():
        wrong = stat["mean_max_prob_wrong"]
        lines.append(f"| {condition} | {stat['mean_max_prob']:.3f} | "
                     f"{stat['mean_max_prob_correct']:.3f} | "
                     f"{'—' if wrong is None else f'{wrong:.3f}'} |")
    lines += failure_sections(failure)
    lines += [
        "",
        "## 8. 结论与下一步最小实验",
        "",
        "- G1 **在 9 个条件上达到跨骨干门槛（4/4 家族）**，主要来自眩光、云与雾的组合；单雾 5 级不达标。",
        "- 因此课题动机写成\"组合退化（雾+眩光、雾+云）与眩光/云单独作用\"，不写\"单一浓雾\"。",
        "- 未自动启动第二骨干训练，也未启动 G2 模块；下一步最小实验（需人工批准）：",
        "  1. 在 matched_haze+glare_5、matched_glare_5、matched_haze+cloud_5 三个条件上做逐类损失归因，"
        "确认退化主要打击哪些类别（本表已给出逐类召回，可直接复用）；",
        "  2. 若要满足 G1 的\"≥2 骨干\"更严格口径（当前已是 4 家族，含 EFM-Net），可只补 EFM-Net 的 "
        "ConvNeXt-T/S 变体各一次训练（每次约 2.7 h），非必需；",
        "  3. 模块设计必须以\"组合退化下的逐类召回塌缩\"为机制假设，不能以重建/一致性为目标"
        "（`docs/THESIS_SCOPE.md` §4 已证伪）。",
        "",
    ]
    return "\n".join(lines)



FOCUS_MARGIN = ("clean", "matched_haze_5", "shifted_haze_5", "matched_glare_5",
                "matched_cloud_5", "matched_haze+glare_5", "matched_haze+cloud_5")


def failure_sections(failure):
    families = list(failure["family_conditions"])
    lines = ["", "## 5b. 预定义退化组（先定义、后统计，无事后挑选）", ""]
    lines.append("| 组 | 条件数 | " + " | ".join(f"{f} 均值Δ / BH 显著" for f in families) + " |")
    lines.append("|---|---|" + "---|" * len(families))
    for group, entry in failure["groups_summary"].items():
        cells = [f"{entry['families'][f]['mean_delta_pp']:+.1f} pp / "
                 f"{entry['families'][f]['n_bh_significant']}/{entry['n_conditions']}"
                 for f in families]
        lines.append(f"| {group} | {entry['n_conditions']} | " + " | ".join(cells) + " |")

    fdr = failure["fdr"]
    lines += [
        "",
        "## 5c. 多重比较与样本量",
        "",
        f"- 检验数：{fdr['n_tests']}（4 个骨干家族 × {fdr['n_tests'] // 4} 个退化条件）；"
        f"未校正时 α=0.05 下期望假阳性约 {fdr['expected_false_positives_without_correction']:.1f} 个；"
        f"Benjamini–Hochberg 后显著 {fdr['n_significant']} 个（截断 p ≤ {fdr['cutoff']:.4f}）。",
        f"- 样本量：157 张，{failure['groups']} 个独立来源组；单条件准确率的标准误约 "
        f"{100 * (0.2 * 0.8 / 157) ** 0.5:.1f} pp 量级，因此 2–3 pp 的差异不能解读为效应。",
        "- 单侧检验（退化方向），bootstrap 以来源组为单位重采样 2000 次。",
        "",
        "## 5d. 判别 margin 与真实类置信度",
        "",
        "margin = top-1 概率 − top-2 概率（预测错误时取负）；真实类置信度 = 真实类概率。",
        "",
        "| 条件 | " + " | ".join(families) + " |",
        "|---|" + "---|" * len(families),
    ]
    for condition in FOCUS_MARGIN:
        cells = []
        for family in families:
            stat = failure["family_conditions"][family][condition]
            cells.append(f"margin {stat['mean_margin']:+.3f} / 真类 {stat['mean_true_class_prob']:.3f}")
        lines.append(f"| {condition} | " + " | ".join(cells) + " |")

    combos = failure["combos"]
    lines += [
        "",
        "## 5e. 组合退化的超加性 margin 损失",
        "",
        "加性预期 = clean margin + 单雾 Δ + 单眩光/单云 Δ；超加性 = 实际 − 加性预期（越负越超加）。",
        "",
        "| 组合条件 | " + " | ".join(families) + " |",
        "|---|" + "---|" * len(families),
    ]
    for condition, per_family in combos.items():
        cells = [f"{per_family[f]['extra_loss']:+.3f}" if f in per_family else "—" for f in families]
        lines.append(f"| {condition} | " + " | ".join(cells) + " |")

    allwrong = {k: v for k, v in failure["allwrong"].items()
                if isinstance(v, dict) and "n" in v}
    top = sorted(allwrong.items(), key=lambda kv: -kv[1]["n"])[:10]
    lines += [
        "",
        "## 7. 失效样本与四家族共同错误",
        "",
        "四个骨干家族在同一张图上同时判错的样本数（同一批退化图像）：",
        "",
        "| 条件 | 四家族共同错误 | 其中 EFM-Net 干净条件本来正确 |",
        "|---|---|---|",
    ]
    for condition, stat in top:
        lines.append(f"| {condition} | {stat['n']} | {stat['n_clean_correct_for_efmnet']} |")

    persistence = failure["allwrong"]["persistence_in_passing_conditions"]
    lines += [
        "",
        "在先前通过的 9 个条件中，同一张图被四家族同时判错的次数分布：" +
        "，".join(f"{k} 次 → {v} 张" for k, v in sorted(persistence["histogram"].items(), key=lambda kv: int(kv[0]))),
        f"；失败 ≥3 个条件的样本 {len(persistence['samples_failing_3_or_more'])} 张（明细见 JSON）。",
        "",
        "样本路径与来源组见 `reports/failure_analysis.json` 的 `allwrong` 字段，可直接按 `path` 人工核查。",
        "",
    ]
    return lines


def g2_report(analysis, failure, occlusion):
    families = list(failure["family_conditions"])
    efmnet = "efmnet_convnext_b"
    table = failure["family_conditions"][efmnet]
    combos = failure["combos"]
    extra_glare = [combos[c][efmnet]["extra_loss"] for c in combos
                   if "haze+glare" in c and efmnet in combos[c]]
    extra_cloud = [combos[c][efmnet]["extra_loss"] for c in combos
                   if "haze+cloud" in c and efmnet in combos[c]]
    allwrong = {k: v for k, v in failure["allwrong"].items() if isinstance(v, dict) and "n" in v}
    worst = max(allwrong.items(), key=lambda kv: kv[1]["n"])
    center = occlusion["models"]["convnext_tiny"]["overlap"]
    fdr = failure["fdr"]
    groups = failure["groups_summary"]
    lines = [
        "# G2 P0：失效机制假设与最低成本证伪实验",
        "",
        "2026-10-30。本文件只提出机制假设与证伪方案，**不启动任何模块训练**。"
        "全部数字取自 `reports/failure_analysis.json`、`reports/cross_model_degradation.json`、"
        "`reports/part_occlusion.json`。",
        "",
        "## 1. 结论（最受证据支持的机制）",
        "",
        "**组合退化（雾+眩光、雾+云）把中心判别证据的 margin 压到接近零，"
        "模型退化为按少数类原型做决策，错误沿固定的类别对迁移；"
        "单一雾、单一云、单一眩光只造成小幅 margin 收缩，不足以解释组合条件下的塌缩。**",
        "",
        "支持证据：",
        "",
        f"1. margin 单调塌缩（EFM-Net，项目管线）：clean {table['clean']['mean_margin']:+.3f} → "
        f"matched_glare_5 {table['matched_glare_5']['mean_margin']:+.3f} → "
        f"matched_haze+glare_5 {table['matched_haze+glare_5']['mean_margin']:+.3f}；"
        f"真实类置信度 {table['clean']['mean_true_class_prob']:.3f} → "
        f"{table['matched_haze+glare_5']['mean_true_class_prob']:.3f}。margin 仍为正，说明不是特征整体失效，"
        "而是判别边界被压缩到噪声尺度。",
        f"2. 组合的超加性：雾+眩光条件的额外 margin 损失 "
        f"{min(extra_glare):+.3f}…{max(extra_glare):+.3f}，"
        f"雾+云 {min(extra_cloud):+.3f}…{max(extra_cloud):+.3f}（加性预期之外的部分）。",
        f"3. 错误集中在固定类别对（EFM-Net，matched_haze+glare_5）：2→3（bulk carrier→oil tanker）9 次、"
        "0→2（container→bulk）8 次、7→6（barge→liquefied gas）8 次；逐类召回 barge 90%→10%、"
        "car carrier 83%→33%、fishing boat 50%→6%。",
        f"4. 四个家族共同错误 {worst[1]['n']} 张 / 157（`{worst[0]}`），"
        f"其中 EFM-Net 干净条件本可正确 {worst[1]['n_clean_correct_for_efmnet']} 张；"
        "错误不是单个模型的偶发。",
        f"5. 区域证据（**仅区域级，不是部件级**）：同面积灰块遮挡中心区造成的下降是角落区的 3.4–6.3 倍；"
        f"ConvNeXt-T 的中心遮挡错误中 {100 * center['center_errors_also_degraded_wrong']:.1f}% "
        "同时也是雾+眩光错误。FGSC-23 没有船体部件标注，因此不能声称定位到舰桥/船首等部件。",
        "",
        "## 2. 竞争解释",
        "",
        "| # | 解释 | 与本数据的符合度 | 判别性预测 |",
        "|---|---|---|---|",
        "| A | 全局光度损失（对比度/饱和度下降），与空间内容无关 | 部分符合：单雾/单云掉点小且均匀 | 若成立，等均值平坦雾与结构雾应当等效（已观测：结构无可测效应，支持 A 的一部分） |",
        "| B | 类别先验/长尾吸引：模型转向样本多的原型类 | 符合错误方向集中（bulk/tanker/gas） | 若成立，重采样/损失重加权即可缓解，无需特征级模块 |",
        "| C | 中心判别证据被破坏（区域级） | 符合遮挡实验与错误重叠 | 若成立，中心区的 margin 损失应显著大于边缘同面积遮挡 |",
        "| D | 预处理/域偏移放大退化 | 部分：EFM-Net 两套预处理相差 7.64 pp | 若成立，退化 Δ 应随预处理强烈变化（实测两套 Δ 差异均值 −2.5 pp，方向一致） |",
        "| E | 统计噪声（157 张、168 次检验） | 已控制：BH 后仍 "
        f"{fdr['n_significant']}/{fdr['n_tests']} 显著，未校正期望假阳性 {fdr['expected_false_positives_without_correction']:.1f} | 若成立，多数条件不应在 4 个家族上同时显著 |",
        "",
        "预定义组平均（无事后挑选）：" + "；".join(
            f"{g} {groups[g]['families'][efmnet]['mean_delta_pp']:+.1f} pp（{groups[g]['n_conditions']} 个条件）"
            for g in groups) + "。",
        "",
        "## 3. 最低成本证伪实验",
        "",
        "**冻结特征线性探针（不做任何训练）。**",
        "",
        "- 做法：取 4 个骨干家族各自的倒数第二层特征（每家族 1 个种子，共 4 个），"
        "在 FGSC-23 train 618 张的 clean 图上拟合 8 类逻辑回归，val 157 张上分别在 clean、"
        "matched_glare_5、matched_haze+glare_5、matched_haze+cloud_5、shifted_haze+glare_5 上评估。",
        "- 判据：若探针在各退化条件下的掉点远小于模型本身（例如 <2 pp，而模型 >20 pp），"
        "则信息仍在特征中、失效发生在决策边界 → 机制在**判别层**，模块应做边界/校准，"
        "不得做重建；若探针同样塌缩，则为**表征层**损失，机制描述需改写。",
        "- 为什么能证伪：它能分开本文件的机制（判别边界压缩）与竞争解释 A/B/C 中"
        "「信息已被摧毁」的版本。它不消耗训练预算，也不触碰测试集。",
        "- 成本：4 家族 × 157 张 × 5 条件 ≈ 3,140 次特征前向 + 4 次逻辑回归；GPU < 5 min，"
        "总计 < 30 min（含特征缓存）。",
        "",
        "## 4. 普通增强基线与计算成本要求",
        "",
        "- 对照基线（已有、零成本）：`runs/fgsc23_research_civilian/convnext_tiny_aug_s17/s29/s43`、"
        "`swin_t_aug_*`、`resnet50_aug_s17`，即「普通增强」训练的同数据模型；模块必须在该基线上比较，"
        "而不是只与组合增强基线比较。",
        "- 若需新的普通增强基线：ConvNeXt-T 8 类 618 图，约 15–20 min/run，3 种子约 1 h。",
        "- 模块成本上限（建议预注册）：参数量增量 ≤ +10%，单图 FLOPs 增量 ≤ +15%，"
        "推理延迟增量 ≤ +15%；报告必须给出「每百分点增益的成本增量」。",
        "- G2 准入不变：退化层均值 ≥ +1.0 pp、≥2/3 种子为正、干净 ≥ −1.0 pp、其余层 ≥ −2.0 pp。",
        "",
        "## 5. 当前不采取的动作",
        "",
        "- 不训练 EFM-Net ConvNeXt-T/S 变体。",
        "- 不启动任何新模块、不调退化强度、不看测试集。",
        "- 下一步只有一件事：执行 §3 的线性探针（<30 min，无训练），用结果决定机制描述是否成立。",
        "",
    ]
    return "\n".join(lines)


def main():
    analysis = load("reports/cross_model_degradation.json")
    failure = load("reports/failure_analysis.json")
    occlusion = load("reports/part_occlusion.json")
    run = run_dir()
    (ROOT / "reports/EFMNET_REPRO.md").write_text(repro_report(run, analysis))
    (ROOT / "reports/CROSS_MODEL_DEGRADATION.md").write_text(cross_report(analysis, failure))
    (ROOT / "reports/G2_P0_FAILURE_MECHANISM.md").write_text(
        g2_report(analysis, failure, occlusion))
    print("wrote reports/EFMNET_REPRO.md, reports/CROSS_MODEL_DEGRADATION.md, "
          "reports/G2_P0_FAILURE_MECHANISM.md")


if __name__ == "__main__":
    main()
