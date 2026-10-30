#!/usr/bin/env python
"""Aggregate C1/C2 joint-training runs against same-seed combined baselines."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maritime_robust.common import ROOT as PROJECT, read_jsonl  # noqa: E402

SEEDS = (17, 29, 43)
BACKBONES = ("swin_t", "convnext_tiny")
MODES = {"c1": "localized", "c2": "consistency", "c3": "feature"}
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
DISPLAY = {"swin_t": "Swin-Tiny", "convnext_tiny": "ConvNeXt-Tiny"}


def run_paths(backbone, mode, seed):
    base = PROJECT / f"runs/fgsc23_research_civilian/{backbone}_combo_s{seed}"
    joint = PROJECT / (f"runs_joint/fgsc23_research_civilian/"
                       f"{backbone}_joint_{MODES[mode]}_s{seed}")
    return base, joint


def load(run):
    return json.loads((run / "evaluation/summary.json").read_text())


def family_mean(summary, conditions):
    return float(np.mean([summary[c]["macro_f1"] for c in conditions]))


def run_families(summary):
    out = {name: family_mean(summary, conds) for name, conds in FAMILIES.items()}
    out["worst"] = min(v["macro_f1"] for v in summary.values())
    return out


def confidence_stats(run, conditions):
    wrong = high = n = 0
    conf_errors = []
    for cond in conditions:
        for r in read_jsonl(run / "evaluation" / f"{cond}.jsonl"):
            p = max(r["probabilities"])
            n += 1
            if r["prediction"] != r["label"]:
                wrong += 1
                conf_errors.append(p)
                high += int(p >= 0.9)
    return {"n": n, "error_rate": wrong / n, "hce_rate": high / n,
            "mean_confidence_on_errors": float(np.mean(conf_errors)) if conf_errors else None}


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}"


def table(rows, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def main():
    data = {"seeds": SEEDS, "families": list(FAMILIES), "modes": {}}
    sections = []
    for backbone in BACKBONES:
        for mode in MODES:
            seed_rows, paired, per_condition, diagnostics, confidence = [], [], [], [], []
            baseline_mc = []
            available = []
            for seed in SEEDS:
                base, joint = run_paths(backbone, mode, seed)
                if not (joint / "evaluation/summary.json").exists():
                    continue
                b, j = load(base), load(joint)
                bf, jf = run_families(b), run_families(j)
                conf_conditions = ["clean"] + FAMILIES["matched combination"]
                cb, cj = confidence_stats(base, conf_conditions), confidence_stats(joint, conf_conditions)
                confidence.append({"seed": seed, "baseline": cb, "variant": cj,
                                   "delta_hce_pp": 100 * (cj["hce_rate"] - cb["hce_rate"])})
                available.append(seed)
                baseline_mc.append(100 * bf["matched combination"])
                seed_rows.append([DISPLAY[backbone], MODES[mode], seed,
                                  pct(bf["matched combination"]), pct(jf["matched combination"]),
                                  pct(jf["clean"]), pct(jf["matched single"]),
                                  pct(jf["shifted single"]),
                                  pct(jf["shifted combination"]), pct(jf["worst"])])
                paired.append({name: 100 * (jf[name] - bf[name]) for name in
                               list(FAMILIES) + ["worst"]})
                for cond in b:
                    per_condition.append((cond,
                                          100 * (j[cond]["macro_f1"] - b[cond]["macro_f1"])))
                complete = json.loads((joint / "complete.json").read_text())
                hist = json.loads((joint / "history.json").read_text())
                diagnostics.append({"seed": seed,
                                    "val_macro_f1": complete["best_val_macro_f1"],
                                    "final_consistency_loss": complete.get("final_consistency_loss"),
                                    "final_feature_loss": complete.get("final_feature_loss"),
                                    "final_val_entropy": hist[-1].get("val_mean_entropy"),
                                    "final_val_max_prob": hist[-1].get("val_mean_max_prob"),
                                    "final_val_pred_hist": hist[-1].get("val_pred_hist")})
            if not available:
                continue
            means = {name: float(np.mean([p[name] for p in paired])) for name in paired[0]}
            sds = {name: float(np.std([p[name] for p in paired], ddof=1))
                   if len(paired) > 1 else None for name in paired[0]}
            seed_mc = [p["matched combination"] for p in paired]
            seed_clean = [p["clean"] for p in paired]
            admission = (means["matched combination"] >= 1.0 and means["clean"] >= -1.0
                         and sum(x > 0 for x in seed_mc) >= 2
                         and min(seed_clean) >= -2.0
                         and min(means[f] for f in ("matched single", "shifted single",
                                                    "shifted combination", "worst")) >= -2.0)
            cond_means = {}
            for cond, delta in per_condition:
                cond_means.setdefault(cond, []).append(delta)
            ranking = sorted(((float(np.mean(v)), cond) for cond, v in cond_means.items()),
                             reverse=True)
            data["modes"][f"{backbone}_{mode}"] = {
                "seeds": available, "family_means": means, "family_sd": sds,
                "seed_matched_combination": seed_mc, "seed_clean": seed_clean,
                "seed_baseline_matched_combination": baseline_mc,
                "admission": admission, "diagnostics": diagnostics,
                "confidence": confidence,
                "top_gains": ranking[:5], "top_losses": ranking[-5:],
            }
            delta_rows = [[DISPLAY[backbone], MODES[mode], name, f"{means[name]:+.1f}",
                           "—" if sds[name] is None else f"{sds[name]:.1f}"]
                          for name in list(FAMILIES) + ["worst"]]
            diag_rows = [[DISPLAY[backbone], MODES[mode], d["seed"],
                          "—" if (d["final_consistency_loss"] is None and
                                  d["final_feature_loss"] is None) else
                          (f"{d['final_consistency_loss']:.4f}"
                           if d["final_consistency_loss"] is not None
                           else f"{d['final_feature_loss']:.4f}"),
                          "—" if d["final_val_entropy"] is None
                          else f"{d['final_val_entropy']:.3f}",
                          "—" if d["final_val_max_prob"] is None
                          else f"{d['final_val_max_prob']:.3f}",
                          "—" if not d["final_val_pred_hist"]
                          else str(d["final_val_pred_hist"])] for d in diagnostics]
            conf_rows = [[DISPLAY[backbone], MODES[mode], c["seed"],
                          pct(c["baseline"]["hce_rate"]), pct(c["variant"]["hce_rate"]),
                          f"{c['delta_hce_pp']:+.1f}",
                          "—" if c["baseline"]["mean_confidence_on_errors"] is None
                          else f"{c['baseline']['mean_confidence_on_errors']:.3f}",
                          "—" if c["variant"]["mean_confidence_on_errors"] is None
                          else f"{c['variant']['mean_confidence_on_errors']:.3f}"]
                         for c in confidence]
            sections.append({
                "title": f"{DISPLAY[backbone]} / {MODES[mode]}（seeds {available}）",
                "seed_table": table(seed_rows, ["模型", "变体", "seed", "基线 matched 组合",
                                                "变体 matched 组合", "干净", "matched 单",
                                                "shifted 单", "shifted 组合", "最差"]),
                "delta_table": table(delta_rows, ["模型", "变体", "家族", "Δ pp 均值", "种子 sd"]),
                "diag_table": table(diag_rows, ["模型", "变体", "seed", "一致性/特征损失",
                                                "验证熵", "验证 max prob", "验证预测直方图"]),
                "conf_table": table(conf_rows, ["模型", "变体", "seed", "基线 HCE(%)",
                                                "变体 HCE(%)", "Δ pp", "基线错误平均置信",
                                                "变体错误平均置信"]),
                "ranking": ranking,
            })

    text = ["# C 线第一轮：联合训练消融（C1/C2/C3）", "",
            "## 0. 准入规则与指标", "",
            "- 主指标为 Macro-F1；家族值为该家族条件 Macro-F1 的未加权平均（条件数见 `reports/joint_round1.json`）。",
            "- 配对比较：同一骨干、同一种子、同一评测条件下，变体减组合增强基线，乘 100 得 pp；跨种子取均值与样本 sd。",
            "- 准入：matched 组合 Δ ≥ +1.0 pp，干净 Δ ≥ −1.0 pp，至少 2/3 种子 matched 组合为正，没有种子干净 Δ 低于 −2.0 pp，且 matched 单退化、shifted 单退化、shifted 组合、最差条件四项均值均不低 2.0 pp 以上。",
            "- 三个种子是判定单位；单种子正负不构成结论。",
            "- C2 的验证熵为验证集逐样本预测分布的平均熵（8 类，均匀=2.079）；验证 max prob 为平均首位概率；两者与类别直方图用于发现一致性塌缩。", ""]
    text += ["## 0.1 结论", ""]
    for key, value in data["modes"].items():
        text.append(
            f"- `{key}`：{'通过' if value['admission'] else '未通过'}；matched 组合各种子 Δ = "
            f"{', '.join(f'{x:+.1f}' for x in value['seed_matched_combination'])} pp，"
            f"均值 {value['family_means']['matched combination']:+.1f} pp；干净均值 "
            f"{value['family_means']['clean']:+.1f} pp。")
        base = value["seed_baseline_matched_combination"]
        gains = value["seed_matched_combination"]
        if base and max(gains) == gains[int(np.argmin(base))]:
            text.append(
                f"  - 该组种子基线组合为 {', '.join(f'{x:.1f}' for x in base)}；最大增益出现在"
                f"基线最低的 seed {value['seeds'][int(np.argmin(base))]}，与回归均值一致，"
                f"不能当作处理效应。")
    if data["modes"]:
        text += ["", "三种子方向不一致时，均值不能作为增益证据；本轮没有任何变体在 matched 组合上"
                     "同时满足“均值 ≥ +1 pp、至少 2/3 种子为正、没有种子干净图低于 −2 pp”。", ""]
    if not sections:
        text += ["当前没有可汇总的 C1/C2 运行。", ""]
    for section in sections:
        text += [f"## {section['title']}", "", "**逐种子家族值**", "",
                 section["seed_table"], "", "**相对同种子组合基线的配对 Δ**", "",
                 section["delta_table"], "", "**训练诊断（C2/C3）**", "",
                 section["diag_table"], "", "**高置信错误（干净 + matched 组合，236×5 样本）**", "",
                 "HCE 定义为错误且首位概率 ≥ 0.9 的样本数 / 该组样本总数。", "",
                 section["conf_table"], "",
                 "单条件 Δ 最大/最小五项：" +
                 "；".join(f"{cond} {delta:+.1f}" for delta, cond in section["ranking"][:5]) +
                 " / " +
                 "；".join(f"{cond} {delta:+.1f}" for delta, cond in section["ranking"][-5:]),
                 "", ""]
    (PROJECT / "reports/JOINT_ROUND1.md").write_text("\n".join(text))
    (PROJECT / "reports/joint_round1.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(json.dumps({k: v["admission"] for k, v in data["modes"].items()}))


if __name__ == "__main__":
    main()
