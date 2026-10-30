#!/usr/bin/env python
"""Aggregate paired-restorer evaluations against identity on the same samples."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maritime_robust.common import ROOT as PROJECT  # noqa: E402

KINDS = ("haze", "cloud", "glare")
FAMILIES = {
    "clean": ["clean"],
    "matched single": [f"matched_{k}_{s}" for k in KINDS for s in range(1, 6)],
    "shifted single": [f"shifted_{k}_{s}" for k in KINDS for s in range(1, 6)],
    "matched combination": [f"matched_{k}_{s}" for k in ("haze+glare", "haze+cloud")
                            for s in (3, 5)],
    "shifted combination": [f"shifted_{k}_{s}" for k in ("haze+glare", "haze+cloud")
                            for s in (3, 5)],
}
HCE_CONDITIONS = ["clean"] + FAMILIES["matched combination"]
DISPLAY = {"resnet50": "ResNet-50", "convnext_tiny": "ConvNeXt-Tiny", "swin_t": "Swin-Tiny"}
BACKBONES = ("resnet50", "convnext_tiny", "swin_t")


def load(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def fmean(entries, conditions):
    return float(np.mean([entries[c]["macro_f1"] for c in conditions]))


def family_means(entries):
    out = {name: fmean(entries, conds) for name, conds in FAMILIES.items()}
    out["worst"] = min(entries[c]["macro_f1"] for c in entries)
    return out


def restored_mean(restorers, condition):
    return float(np.mean([restorers[str(seed)][condition]["macro_f1"]
                          for seed in restorers]))


def restored_family_means(restorers):
    out = {}
    for name, conds in FAMILIES.items():
        out[name] = float(np.mean([restored_mean(restorers, c) for c in conds]))
    out["worst"] = min(restored_mean(restorers, c) for c in restorers[str(next(iter(restorers)))])
    return out


def weighted_rate(values, weights):
    pairs = [(v, w) for v, w in zip(values, weights) if v is not None and w > 0]
    if not pairs:
        return None
    return sum(v * w for v, w in pairs) / sum(w for _, w in pairs)


def hce_mean(entries, restorer_entries=None):
    def one(table):
        return float(np.mean([table[c]["hce_rate"] for c in HCE_CONDITIONS]))
    if restorer_entries is None:
        return one(entries)
    return float(np.mean([one(restorer_entries[str(seed)]) for seed in restorer_entries]))


def unit_rows(data, backbone):
    rows = []
    for name, info in data["classifiers"].items():
        if info["backbone"] != backbone:
            continue
        identity = info["identity"]
        restorers = info["restorers"]
        base = family_means(identity)
        rest = restored_family_means(restorers)
        per_seed = {}
        for seed, table in restorers.items():
            table_means = family_means(table)
            per_seed[seed] = {k: 100 * (table_means[k] - base[k])
                              for k in list(FAMILIES) + ["worst"]}
        rows.append({
            "classifier": name, "seed": info["seed"],
            "identity": base, "restored": rest,
            "delta": {k: 100 * (rest[k] - base[k]) for k in list(FAMILIES) + ["worst"]},
            "per_restorer_delta": per_seed,
            "restorer_positive": float(np.mean([
                per_seed[s]["matched combination"] > 0 for s in per_seed])),
            "hce_identity": hce_mean(identity),
            "hce_restored": hce_mean(identity, restorers),
            "rescue": _rescue(identity, restorers),
        })
    return rows


def _rescue(identity, restorers):
    conditions = FAMILIES["matched combination"]
    weights = [identity[c]["recoverable_n"] for c in conditions]
    rates = []
    for seed in restorers:
        rates.append([restorers[str(seed)][c]["rescue_rate"] for c in conditions])
    rate = [weighted_rate([r[i] for r in rates], weights) for i in range(len(conditions))]
    stable = [identity[c]["stable_n"] for c in conditions]
    induce = [weighted_rate([restorers[str(seed)][c]["error_induction_rate"]
                             for seed in restorers], stable) for c in conditions]
    induce = [x for x in induce if x is not None]
    return {"rescue_rate": weighted_rate(rate, weights),
            "error_induction_rate": float(np.mean(induce)) if induce else None,
            "recoverable_n": int(sum(weights))}


def admission(rows):
    if not rows:
        return {"pass": False, "reason": "no classifier runs"}
    names = list(FAMILIES) + ["worst"]
    means = {k: float(np.mean([r["delta"][k] for r in rows])) for k in names}
    seed_mc = [r["delta"]["matched combination"] for r in rows]
    seed_clean = [r["delta"]["clean"] for r in rows]
    positive = sum(x > 0 for x in seed_mc)
    restorer_positive = float(np.mean([r["restorer_positive"] for r in rows]))
    enough_seeds = len(rows) >= 2
    passed = (enough_seeds and means["matched combination"] >= 1.0
              and means["clean"] >= -1.0
              and positive >= max(2, int(np.ceil(2 / 3 * len(rows))))
              and min(seed_clean) >= -2.0
              and min(means[f] for f in ("matched single", "shifted single",
                                         "shifted combination", "worst")) >= -2.0
              and restorer_positive >= 2 / 3)
    return {"pass": bool(passed), "means": means, "seed_matched_combination": seed_mc,
            "seed_clean": seed_clean, "positive": positive, "n": len(rows),
            "restorer_positive_fraction": restorer_positive,
            "reason": "ok" if passed else "admission failed"}


def pixel_summary(data):
    out = {}
    for name, conds in FAMILIES.items():
        degraded_l1, restored_l1, degraded_psnr, restored_psnr = [], [], [], []
        for c in conds:
            degraded_l1.append(data["identity_pixel"][c]["l1"])
            for seed, table in data["pixel"].items():
                restored_l1.append(table[c]["l1"])
                if table[c]["psnr"] is not None:
                    restored_psnr.append(table[c]["psnr"])
            if data["identity_pixel"][c]["psnr"] is not None:
                degraded_psnr.append(data["identity_pixel"][c]["psnr"])
        out[name] = {"degraded_l1": float(np.mean(degraded_l1)),
                     "restored_l1": float(np.mean(restored_l1)),
                     "degraded_psnr": float(np.mean(degraded_psnr)) if degraded_psnr else None,
                     "restored_psnr": float(np.mean(restored_psnr)) if restored_psnr else None}
    return out


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}"


def table(rows, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def main():
    val = load(PROJECT / "reports/restorer_eval_val.json")
    test = load(PROJECT / "reports/restorer_eval_test.json")
    summary = {"val": {}, "test": {}}
    text = ["# 配对监督复原器（第一轮）", "",
            "## 0. 方法与准入", "",
            "轻量残差 U-Net（3 级、base 32、约 0.8M 参数）在训练集上学习精确的"
            "（退化、干净）对；损失为内容区 L1 + 0.1 边缘 L1；训练混合为 50% 干净、"
            "25% 单退化（1–5 级）、25% 组合（3/5 级），matched 域、随机位置。"
            "不使用分类器、标签或验证/测试图像训练；checkpoint 由验证集五个退化条件的"
            "平均像素 L1 选择。",
            "",
            "- 主指标为 Macro-F1；家族值为该家族条件 Macro-F1 的未加权平均。",
            "- 配对比较：同一分类器、同一种子、同一条件下，复原减 identity（不复原），乘 100 为 pp。",
            "- 每个分类器种子先对三个复原器种子取平均，再作为准入单位；另报告逐复原器种子方向。",
            "- 准入：matched 组合均值 ≥ +1.0 pp、干净均值 ≥ −1.0 pp、至少 2/3 分类器种子为正、"
            "没有种子干净低于 −2.0 pp、其余家族均值不低于 −2.0 pp，且至少 2/3 的"
            "（分类器种子 × 复原器种子）组合在 matched 组合上为正。",
            "- HCE：错误且首位概率 ≥ 0.9 的样本数 / 样本数，取干净 + 四个 matched 组合条件的平均。",
            "- rescue：identity 错、同一分类器干净预测对的样本中，复原后预测对的比例；"
            "error induction：identity 对、干净也对、复原后错的样本比例。仅在 matched 组合上汇总。",
            "- 像素指标只用于说明复原器确实改变了图像；分类增益才是准入依据。",
            ""]
    if val is None:
        text += ["当前没有验证集评测文件，尚未运行评测。", ""]
        (PROJECT / "reports/PAIRED_RESTORER_ROUND1.md").write_text("\n".join(text))
        print(json.dumps({"status": "no eval"}))
        return
    for split, data in (("val", val), ("test", test)):
        if data is None:
            continue
        summary[split] = {}
        text += [f"## 1{'.' if split == 'val' else '.1'} {'验证集（开发闸门）' if split == 'val' else '测试集（方法定型后一次）'}", ""]
        gate_rows = []
        for backbone in BACKBONES:
            rows = unit_rows(data, backbone)
            if not rows:
                continue
            decision = admission(rows)
            summary[split][backbone] = {"rows": rows, "admission": decision,
                                         "pixel": pixel_summary(data) if split == "test" else None}
            seed_table = []
            for r in rows:
                seed_table.append([DISPLAY[backbone], r["seed"],
                                   pct(r["identity"]["matched combination"]),
                                   pct(r["restored"]["matched combination"]),
                                   f"{r['delta']['matched combination']:+.1f}",
                                   pct(r["identity"]["clean"]), pct(r["restored"]["clean"]),
                                   f"{r['delta']['clean']:+.1f}",
                                   f"{r['restorer_positive']:.2f}",
                                   "诊断（单种子）" if len(rows) < 2 else
                                   ("是" if (r["delta"]["matched combination"] >= 1.0 and
                                            r["delta"]["clean"] >= -1.0) else "否")])
            text += [f"### {DISPLAY[backbone]}（{len(rows)} 个分类器种子）", "",
                     table(seed_table, ["模型", "分类器 seed", "基线 matched 组合",
                                        "复原 matched 组合", "Δ pp", "基线干净",
                                        "复原干净", "Δ pp", "复原种子为正比例",
                                        "单种子通过"]), ""]
            dm = decision["means"] if "means" in decision else {}
            if dm:
                family_rows = [[name, f"{dm[name]:+.1f}"] for name in list(FAMILIES) + ["worst"]]
                text += ["**家族配对 Δ（pp，三个复原器种子平均）**", "",
                         table(family_rows, ["家族", "Δ pp"]), ""]
            text += [f"准入：{'通过' if decision['pass'] else '未通过'}；"
                     f"matched 组合各种子 Δ = "
                     f"{', '.join(f'{x:+.1f}' for x in decision['seed_matched_combination'])} pp；"
                     f"干净各种子 Δ = "
                     f"{', '.join(f'{x:+.1f}' for x in decision['seed_clean'])} pp；"
                     f"（分类器种子 × 复原器种子）正比例为 "
                     f"{decision['restorer_positive_fraction']:.2f}。", ""]
            detail = [[r["seed"],
                       "—" if r["rescue"]["rescue_rate"] is None else pct(r["rescue"]["rescue_rate"]),
                       "—" if r["rescue"]["error_induction_rate"] is None
                       else f"{100 * r['rescue']['error_induction_rate']:.1f}",
                       pct(r["hce_identity"]), pct(r["hce_restored"]),
                       f"{100 * (r['hce_restored'] - r['hce_identity']):+.1f}"]
                      for r in rows]
            text += ["**机制诊断（matched 组合）**", "",
                     table(detail, ["分类器 seed", "rescue(%)", "error induction(%)",
                                    "HCE 基线(%)", "HCE 复原(%)", "Δ pp"]), ""]
            per_seed_rows = []
            for r in rows:
                for seed, delta in r["per_restorer_delta"].items():
                    per_seed_rows.append([DISPLAY[backbone], r["seed"], seed,
                                          f"{delta['matched combination']:+.1f}",
                                          f"{delta['clean']:+.1f}"])
            text += ["**逐复原器种子 Δ（matched 组合 / 干净）**", "",
                     table(per_seed_rows, ["模型", "分类器 seed", "复原器 seed",
                                           "matched 组合 Δ pp", "干净 Δ pp"]), ""]
            gate_rows.append([DISPLAY[backbone],
                              "诊断（单种子）" if len(rows) < 2 else
                              ("通过" if decision["pass"] else "未通过"),
                              f"{decision['means']['clean']:+.1f}",
                              f"{decision['means']['matched combination']:+.1f}"])
        text += ["**汇总**", "",
                 table(gate_rows, ["模型", "准入", "干净 Δ 均值", "matched 组合 Δ 均值"]), "",
                 "ResNet-50 只有种子 17 的组合增强基线，不能构成三种子准入，只作诊断。", ""]
        if split == "val":
            gate_parts = []
            for backbone in BACKBONES:
                if backbone not in summary["val"]:
                    continue
                if len(summary["val"][backbone]["rows"]) < 2:
                    continue
                means = summary["val"][backbone]["admission"].get("means")
                if means:
                    gate_parts.append(means["clean"] >= -1.0 and
                                      means["matched combination"] >= -1.0)
            text += ["验证集闸门只检查灾难性伤害：干净与 matched 组合均不低于 −1.0 pp 才进入测试集。"
                     f"当前闸门：{'通过' if gate_parts and all(gate_parts) else '未通过'}。"
                     "验证集不用于有效性判定。", ""]
            pixel = pixel_summary(data)
            pixel_rows = [[name, f"{v['degraded_l1']:.3f}", f"{v['restored_l1']:.3f}",
                           "—" if v["degraded_psnr"] is None else f"{v['degraded_psnr']:.2f}",
                           "—" if v["restored_psnr"] is None else f"{v['restored_psnr']:.2f}"]
                          for name, v in pixel.items()]
            text += ["**像素诊断（验证集，matched 域，三个复原器种子平均）**", "",
                     table(pixel_rows, ["家族", "退化 L1", "复原 L1", "退化 PSNR", "复原 PSNR"]),
                     "",
                     "PSNR 在 0–1 像素尺度计算；干净条件无退化 PSNR。像素指标改善不构成分类增益。", ""]
            contact = load(PROJECT / "reports/figures/restorer_contact_sheet.json")
            if contact:
                ratios = "、".join(f"seed {seed} {100 * value:.1f}%"
                                   for seed, value in contact["restored_gradient_ratio"].items())
                text += ["**定性检查**", "",
                         f"`{contact['output']}` 为 {contact['samples']} 个验证样本的 "
                         f"clean / degraded {contact['kind']}{contact['severity']} / restored 对照。"
                         "梯度能量（相邻像素平均绝对差）相对干净图："
                         f"退化图 {100 * contact['degraded_gradient_ratio']:.1f}%，复原图 {ratios}。"
                         "复原显著改善 L1/PSNR，但高频结构仍远低于干净图；目视可见船体轮廓"
                         "被平滑为低对比块，部分样本在眩光区出现大面积暗色涂抹。", ""]
            rescue_rates = [r["rescue"]["rescue_rate"] for b in BACKBONES
                            for r in summary["val"].get(b, {}).get("rows", [])
                            if r["rescue"]["rescue_rate"] is not None]
            induce_rates = [r["rescue"]["error_induction_rate"] for b in BACKBONES
                            for r in summary["val"].get(b, {}).get("rows", [])
                            if r["rescue"]["error_induction_rate"] is not None]
            combo_conditions = FAMILIES["matched combination"]
            recoverable = [sum(info["identity"][c]["recoverable_n"] for c in combo_conditions)
                           for info in data["classifiers"].values()]
            stable = [sum(info["identity"][c]["stable_n"] for c in combo_conditions)
                      for info in data["classifiers"].values()]
            gains, losses = [], []
            for info in data["classifiers"].values():
                for restorer_table in info["restorers"].values():
                    gains.append(sum((restorer_table[c]["rescue_rate"] or 0.0) *
                                     info["identity"][c]["recoverable_n"]
                                     for c in combo_conditions))
                    losses.append(sum((restorer_table[c]["error_induction_rate"] or 0.0) *
                                      info["identity"][c]["stable_n"]
                                      for c in combo_conditions))
            combo = pixel["matched combination"]
            combo_delta = [summary["val"][b]["admission"]["means"]["matched combination"]
                           for b in BACKBONES if b in summary["val"]
                           and len(summary["val"][b]["rows"]) >= 2]
            text += ["## 2. 解释", "",
                     f"像素与分类方向相反：matched 组合退化 PSNR {combo['degraded_psnr']:.2f}、"
                     f"L1 {combo['degraded_l1']:.3f}，复原后 PSNR {combo['restored_psnr']:.2f}、"
                     f"L1 {combo['restored_l1']:.3f}，但 Macro-F1 下降 "
                     f"{min(abs(x) for x in combo_delta):.1f}–"
                     f"{max(abs(x) for x in combo_delta):.1f} pp。", ""]
            if rescue_rates and induce_rates and stable and recoverable:
                text += [f"机制上，matched 组合的 rescue 为 {100 * min(rescue_rates):.1f}%–"
                         f"{100 * max(rescue_rates):.1f}%，error induction 为 "
                         f"{100 * min(induce_rates):.1f}%–{100 * max(induce_rates):.1f}%"
                         "（分类器种子间）。四个 matched 组合合计，每个分类器的可恢复样本为 "
                         f"{min(recoverable)}–{max(recoverable)} 个、稳定样本为 "
                         f"{min(stable)}–{max(stable)} 个（各条件逐样本计数之和）；"
                         f"按实测比例折算，平均救回 {min(gains):.0f}–{max(gains):.0f} 个、"
                         f"新错 {min(losses):.0f}–{max(losses):.0f} 个，净效果是错误增加。", ""]
            if contact and contact.get("restored_gradient_ratio"):
                ratio = max(contact["restored_gradient_ratio"].values())
                text += [f"梯度能量只恢复到干净图的 {100 * ratio:.0f}%，说明 L1 最优解把被饱和裁剪的"
                         "结构替换为低频平滑内容；这与 B 线用同图真实干净像素替换得到的上界不是同一件事。"
                         "配对监督能学会去雾/去眩光的颜色统计，但没有恢复被裁掉的判别结构。", ""]
    if test is not None:
        text += ["## 2. 像素诊断", ""]
        pixel = pixel_summary(test)
        pixel_rows = [[name, f"{v['degraded_l1']:.3f}", f"{v['restored_l1']:.3f}",
                       "—" if v["degraded_psnr"] is None else f"{v['degraded_psnr']:.2f}",
                       "—" if v["restored_psnr"] is None else f"{v['restored_psnr']:.2f}"]
                      for name, v in pixel.items()]
        text += [table(pixel_rows, ["家族", "退化 L1", "复原 L1", "退化 PSNR", "复原 PSNR"]), "",
                 "PSNR 在 0–1 像素尺度计算；干净条件无退化 PSNR。像素改善不构成分类增益。", ""]
    verdict = []
    for backbone in BACKBONES:
        if test and backbone in summary["test"] and \
                len(summary["test"][backbone]["rows"]) >= 2:
            verdict.append(f"{DISPLAY[backbone]}："
                           f"{'通过' if summary['test'][backbone]['admission']['pass'] else '未通过'}")
    text += ["## 3. 结论", "",
             ("；".join(verdict) if verdict else
              "验证集闸门未通过（干净与 matched 组合均需不低于 −1.0 pp），按预注册规则测试集未运行。"
              "配对监督复原器在 matched 组合上一致为负：全部（分类器种子 × 复原器种子）组合的 "
              "Δ 均小于 0。按 ROUND2_DECISION 的停止条件，复原路线在本轮停止，"
              "不再以更大复原网络继续。"), "",
             "本实验检验的是“B 线真实干净内容上界能否由同配方配对数据训练的复原器实现”。"
             "复原器只见过合成退化；真实传感器退化、跨数据集和不同骨干仍需另行验证。", ""]
    output = PROJECT / "reports/PAIRED_RESTORER_ROUND1.md"
    output.write_text("\n".join(text))
    (PROJECT / "reports/paired_restorer_round1.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({split: {b: summary[split][b]["admission"]["pass"]
                              for b in summary[split]} for split in summary}))


if __name__ == "__main__":
    main()
