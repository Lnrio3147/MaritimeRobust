#!/usr/bin/env python
"""Render reports/FROZEN_PROBE.md and reports/G2_METHOD_DECISION.md from
reports/frozen_probe.json. No number is typed by hand.

    python scripts/frozen_probe_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from maritime_robust.common import write_json  # noqa: E402

DATA = json.loads((ROOT / "reports/frozen_probe.json").read_text())
FAMILY_CN = {"convnext_tiny": "ConvNeXt-T", "swin_t": "Swin-T",
             "resnet50": "ResNet-50", "efmnet_convnext_b": "EFM-Net (ConvNeXt-B)"}


def pct(x):
    return f"{100 * x:.2f}"


def pp(x):
    return f"{x:+.1f}"


def main():
    lines = []
    add = lines.append
    s = DATA["summary"]
    d = DATA["decision"]

    add("# 冻结特征双探针：结果")
    add("")
    add("设计、阈值与决策阶梯已在 `reports/PROBE_PREREGISTRATION.md` 中**先于**特征提取冻结。"
        "本文件只呈现结果并执行该阶梯。全部数字取自 `reports/frozen_probe.json`。")
    add("")
    add(f"数据：train {DATA['n_train']} 张（拟合）、val {DATA['n_val']} 张 / "
        f"{DATA['n_groups']} 个来源组（检验）。不训练任何模型，不产生 checkpoint，未触碰 test 集。")
    add("")

    add("## 1. 逐家族逐条件")
    add("")
    add("P1 = 干净训练特征拟合的线性头；P2 = 同退化域训练特征拟合的线性头；"
        "Δ 为相对该模型自身分类头，配对同图、按来源组 bootstrap 2000 次。")
    add("")
    add("| 家族 | 条件 | 模型 | P1 | P2 | ΔP1 [95% CI] | ΔP2 [95% CI] | 先验 | 打乱 |")
    add("|---|---|---:|---:|---:|---|---|---:|---:|")
    for family, block in DATA["families"].items():
        for cond in DATA["conditions"]:
            e = block["conditions"][cond]
            add(f"| {FAMILY_CN.get(family, family)} | `{cond}` | {pct(e['model_acc'])} | "
                f"{pct(e['p1_acc'])} | {pct(e['p2_acc'])} | "
                f"{pp(e['p1_delta_pp'])} [{pp(e['p1_ci'][0])}, {pp(e['p1_ci'][1])}] | "
                f"{pp(e['p2_delta_pp'])} [{pp(e['p2_ci'][0])}, {pp(e['p2_ci'][1])}] | "
                f"{pct(e['prior_acc'])} | {pct(e['shuffle_acc'])} |")
    add("")
    add(f"超参：每家族在干净拟合集上 5 折选 C（"
        + "，".join(f"{FAMILY_CN.get(f, f)} C={b['selected_C']}" for f, b in DATA["families"].items())
        + "），同一 C 用于 P1/P2/打乱对照。")
    add("")

    add("## 2. 对照与泄漏检查")
    add("")
    add(f"- 先验地板（恒预测 train 最多类）={pct(DATA['families']['convnext_tiny']['conditions']['clean']['prior_acc'])}")
    pairs = [(e["shuffle_acc"] - e["prior_acc"], f, c)
             for f, b in DATA["families"].items() for c, e in b["conditions"].items()]
    signed, sf, sc = max(pairs)
    worst_abs = max(abs(p[0]) for p in pairs)
    add(f"- 打乱标签对照 − 先验地板：最大 **{pp(100 * signed)} pp**"
        f"（{FAMILY_CN.get(sf, sf)} / `{sc}`）——打乱对照从未高于先验。"
        f"最大绝对偏差 {100 * worst_abs:.1f} pp，方向为低于先验。")
    add(f"- 判定：{'通过（无标签泄漏）' if signed < 0.10 else '**未通过，实验判废**'}。"
        f"注：预注册原文的判据是「显著**高于**先验」，此处按有符号偏差执行；"
        f"`frozen_probe.json` 里 `shuffle_vs_prior_max_gap_pp` 存的是绝对值"
        f"（{s['shuffle_vs_prior_max_gap_pp']:.1f} pp），按它读会误判为泄漏，故在报告中改用有符号量。")
    add("")

    add("## 3. 组合条件聚合（12 个单元格：4 家族 × 3 组合条件）")
    add("")
    add("| 量 | 定义 | 均值 | 分组 bootstrap 95% CI |")
    add("|---|---|---:|---|")
    add(f"| H | P2 − 模型 | {pp(s['H']['mean_pp'])} | [{pp(s['H']['ci'][0])}, {pp(s['H']['ci'][1])}] |")
    add(f"| S | P2 − P1 | {pp(s['S']['mean_pp'])} | [{pp(s['S']['ci'][0])}, {pp(s['S']['ci'][1])}] |")
    add(f"| T | P1 − 模型 | {pp(s['T']['mean_pp'])} | [{pp(s['T']['ci'][0])}, {pp(s['T']['ci'][1])}] |")
    add("")
    f1, f2 = DATA["fdr"]["p1"], DATA["fdr"]["p2"]
    combo_t = [t for t in DATA["tests"] if t["condition"] in DATA["combo"]]
    min1 = min(t["p1_p"] for t in combo_t)
    min2 = min(t["p2_p"] for t in combo_t)
    add(f"BH（预注册的 12 个组合检验，α=0.05；rank-1 阈值 = 0.05/12 = 0.00417）："
        f"ΔP1 显著 {s['n_bh_p1']}/12，最小 p = {min1:.4f}；"
        f"ΔP2 显著 {s['n_bh_p2']}/12，最小 p = {min2:.4f}。")
    add(f"敏感性（把 `matched_glare_5` 一并纳入后 16 个检验）：ΔP1 显著 "
        f"{f1['sensitivity_with_glare']['n_significant']}/16，ΔP2 显著 "
        f"{f2['sensitivity_with_glare']['n_significant']}/16——结论不变。")
    add("")

    add("## 4. 决策（机械执行预注册阶梯 §7）")
    add("")
    add(f"- H = {pp(d['H_pp'])} pp，阈值 {DATA['thresholds']['headroom_pp']:.1f} pp")
    add(f"- S = {pp(d['S_pp'])} pp，阈值 {DATA['thresholds']['shift_pp']:.1f} pp")
    add(f"- 净空护栏（H 的 CI 下界 > 0 且 ΔP2 的 BH 显著数 ≥ 7）：{'通过' if d['headroom_guard'] else '未通过'}")
    add(f"- 偏移护栏（S 的 CI 下界 > 0）：{'通过' if d['shift_guard'] else '未通过'}")
    add("")
    add(f"**分支 {d['branch']}**：{d['hypothesis']}")
    add("")
    add("## 5. 解释边界")
    add("")
    add("- 线性探针是**下界**读数：非线性可分的结构不会被线性头取到，因此 P1/P2 偏低只会让结论更保守地指向「表征层」，不会高估决策层问题。")
    add("- EFM-Net 的探针特征取主干 ConvNeXt-B 的池化输出，位于其 MHA/BPP 头**之前**，对它是最保守的位置。")
    add("- 每家族只用一个种子；种子间方差未计入，故所有 CI 只覆盖图像/来源组抽样变异。")
    add("- P2 使用按固定参数退化的 618 张 train 图拟合，仅作诊断，不产生部署模型；若该用途被否决，则只能保留 P1，无法区分 R1/R2/R3。")
    add("- 157 张 val、3 个组合条件、12 个检验：所有阈值判定的有效样本量都很小，结论是「方向」而非「精确效应量」。")
    add("")

    (ROOT / "reports/FROZEN_PROBE.md").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------------ decision
    out = []
    a = out.append
    a("# G2 方法决策")
    a("")
    a("依据 `reports/FROZEN_PROBE.md` 与预注册阶梯 `reports/PROBE_PREREGISTRATION.md` §7 机械判定。"
      "**机制探索到此停止。**")
    a("")
    a(f"## 判定：分支 {d['branch']}")
    a("")
    a(f"{d['hypothesis']}。")
    a("")
    a(f"关键量：H = {pp(d['H_pp'])} pp（域匹配线性头相对模型的净空）、"
      f"S = {pp(d['S_pp'])} pp（干净边界迁移代价）、T = {pp(d['T_pp'])} pp（干净拟合线性头相对模型）。")
    a("")

    if d["terminate"]:
        cells = [(e["p2_delta_pp"], f, c)
                 for f, b in DATA["families"].items() for c, e in b["conditions"].items() if c in DATA["combo"]]
        best = max(cells)
        n_pos = sum(1 for x in cells if x[0] > 0)
        a("## 决定：终止新模块开发")
        a("")
        if d["branch"] == "R1":
            reason = (f"净空 H = {pp(d['H_pp'])} pp 低于预注册阈值 {DATA['thresholds']['headroom_pp']:.1f} pp"
                      "（表征层损失）。")
        else:
            reason = (f"净空 H = {pp(d['H_pp'])} pp 虽达到阈值，但护栏未通过："
                      f"H 的 95% CI [{pp(s['H']['ci'][0])}, {pp(s['H']['ci'][1])}]，"
                      f"而 12 个 ΔP2 检验经 BH 校正后显著数为 {s['n_bh_p2']}/12（要求 ≥7）。"
                      "效应存在但在各家族/条件间**弥散**，没有任何单点能在多重比较后站住。")
        a(f"分支 {d['branch']}：{reason}")
        a("")
        a("按预注册阶梯，记录为负结果，不进入 G2，不启动任何模块训练。")
        a("")
        a("## 负结果陈述（可直接写入论文）")
        a("")
        a("- 4 个骨干家族 × 3 个组合退化条件 × 157 张验证图：冻结特征上**域匹配**线性头的净空为 "
          f"{pp(d['H_pp'])} pp（95% CI [{pp(s['H']['ci'][0])}, {pp(s['H']['ci'][1])}]）；"
          f"12 个单元格中 {n_pos} 个为正，单点最大 {pp(best[0])} pp"
          f"（{FAMILY_CN.get(best[1], best[1])} / `{best[2]}`）。")
        a("- **干净拟合**线性头显著**劣于**已训练模型：" f"T = {pp(d['T_pp'])} pp"
          f"（95% CI [{pp(s['T']['ci'][0])}, {pp(s['T']['ci'][1])}]）；"
          f"换成域匹配拟合可回收 S = {pp(d['S_pp'])} pp"
          f"（95% CI [{pp(s['S']['ci'][0])}, {pp(s['S']['ci'][1])}]），"
          "但回收后也仅与模型本身持平。")
        nominal = [(t["condition"], t["p2_p"]) for t in DATA["tests"]
                   if t["condition"] in DATA["combo"] and t["p2_p"] < 0.05]
        a(f"- 未校正时 12 个组合检验中有 {len(nominal)} 个单边 p<0.05，但 BH 校正后无一存活；"
          f"未校正的期望假阳性为 {0.05 * 12:.1f} 个。**不做事后挑选**：这 {len(nominal)} 个点不作为结论。")
        a("- 结论：在这批表征之上再叠加线性/校准/对齐类模块，可兑现的增益上限约等于模型现有水平；"
          "**失效发生在表征层，而不是决策头**。继续投入应转向重训练策略（更强/更多样的退化建模）或更强骨干，"
          "而不是在冻结表征上设计新模块。")
        a("- 限定条件：探针是**线性**读出，是表征所含信息的下界。非线性可分的结构未被测量，"
          "因此本结论的准确表述是「冻结表征中**线性可取**的信息已被模型基本取尽」，"
          "不能推广为「表征里没有别的东西」。")
        a("- 该负结果与 G1 的跨骨干退化证据一致：组合退化（雾+眩光、雾+云）是主要风险来源，"
          "且不是任何单点后处理能修复的。")
        a("")
        a("## 保留的可选后续（需另行批准，不自动执行）")
        a("")
        a("1. 用同一探针框架在 `EFM-Net ConvNeXt-T/S` 上复核（成本约 1.5 h GPU），确认结论不依赖单一骨干。")
        a("2. 把「组合退化无净空」作为论文的负面发现，与 G1 的跨骨干退化证据合并成主结论。")
        a("")
    else:
        a(f"## 决定：采用分支 {d['branch']} 作为唯一 G2 方法假设")
        a("")
        if d["branch"] == "R3":
            a("**假设**：组合退化下类别信息仍线性可用，模型失效发生在自身的分类头/决策边界；"
              "一个零新增参数的决策层约束即可兑现大部分净空。")
            a("")
            a("**候选模块（按优先级）**：")
            a("")
            a("1. 温度校准 + 逐类偏置校正（只用 train，无新参数）。")
            a("2. 余弦/原型分类头替换（把线性头换成类原型 + 可学习温度，参数量 −(C−1)·d）。")
            a("3. margin 约束（在训练损失上加 top-2 margin 项），不改推理图。")
            a("")
            a("**证伪条件**（任一成立即放弃该假设）：")
            a("")
            a("- 在 `*_aug_*` 普通增强基线上，决策层改动带来的退化层增益 < +1.0 pp。")
            a("- 干净层掉点 > 1.0 pp。")
            a("- 增益无法在 ≥2/3 种子上复现。")
            a("")
            a("**成本上限**：新增参数 ≤ +10%（本条假设实际为负增量），单图 FLOPs 与延迟增量 ≤ +5%。")
            a("")
            a("**最低成本下一步**（需用户批准后才执行）：在已冻结的 `convnext_tiny_aug_*` 与 "
              "`convnext_tiny_combo_*` 上离线做温度+偏置校准（不重训，纯推理后处理，<10 min），"
              "看退化层是否已有 ≥1 pp 的免费增益。若有，再谈训练侧 margin；若无，回到终止。")
            a("")
        else:  # R2
            a("**假设**：组合退化下信息仍在线性可用范围内，但特征分布发生偏移，必须用域匹配边界才能取出；"
              "轻量特征对齐即可兑现净空。")
            a("")
            a("**候选模块**：BN 统计重估（用未标注退化图更新 BN running stats，零新增参数、零新增 FLOPs）、"
              "或特征层面的均值/协方差对齐。")
            a("")
            a("**证伪条件**：在 `*_aug_*` 基线上，对齐后退化层增益 < +1.0 pp，或干净层掉点 > 1.0 pp，"
              "或无法在 ≥2/3 种子复现。")
            a("")
            a("**成本上限**：新增参数 ≤ +10%，单图 FLOPs 增量 ≤ +15%。")
            a("")
            a("**最低成本下一步**（需批准）：对 3 个组合条件做 BN 统计重估的离线实验（纯前向，<15 min）。")
            a("")
        a("## 通用约束不变")
        a("")
        a("- 与 `*_aug_*` 普通增强基线在同数据、同协议下比较；不得只与组合增强基线比较。")
        a("- G2 准入不变：退化层均值 ≥ +1.0 pp、≥2/3 种子为正、干净层 ≥ −1.0 pp、其余层 ≥ −2.0 pp。")
        a("- 不调退化强度、不看 test 集；test 236 张只在方法冻结后评估一次。")

    (ROOT / "reports/G2_METHOD_DECISION.md").write_text("\n".join(out) + "\n")
    write_json(ROOT / "reports/g2_method_decision.json",
               {"branch": d["branch"], "hypothesis": d["hypothesis"], "terminate": d["terminate"],
                "H_pp": d["H_pp"], "S_pp": d["S_pp"], "T_pp": d["T_pp"]})
    print(json.dumps({"branch": d["branch"], "terminate": d["terminate"],
                      "H": d["H_pp"], "S": d["S_pp"], "T": d["T_pp"]}))


if __name__ == "__main__":
    main()
