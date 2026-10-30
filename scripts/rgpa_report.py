#!/usr/bin/env python
"""Round-1 report for Reliability-Gated Part Aggregation (RGPA).

Compares four runs trained through the same harness on the civilian research
manifest, seed 17, combined augmentation:
  plain    frozen engine model (harness parity check against swin_t_combo_s17)
  control  RGPA head with plain GAP (same pipeline, different head init)
  learned  token reliability from features only
  full     reliability plus input saturation/low-contrast cues

Also probes the trained gate (attention temperature / residual share) to test
whether the learned attention direction is useful at all.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from maritime_robust.common import ROOT as PROJECT, Budget, read_jsonl, write_json  # noqa: E402
from maritime_robust.engine import ShipDataset, metrics  # noqa: E402
from modules.rgpa import build_model, input_cues  # noqa: E402

RUNS = {
    "plain": "runs_modules/fgsc23_research_civilian/swin_t_rgpa_plain_s17",
    "control": "runs_modules/fgsc23_research_civilian/swin_t_rgpa_control_s17",
    "learned": "runs_modules/fgsc23_research_civilian/swin_t_rgpa_learned_s17",
    "full": "runs_modules/fgsc23_research_civilian/swin_t_rgpa_gate_s17",
}
BASELINE = "runs/fgsc23_research_civilian/swin_t_combo_s17"
PLAIN_SEEDS = (17, 29, 43)


def plain_path(seed):
    return f"runs_modules/fgsc23_research_civilian/swin_t_rgpa_plain_s{seed}"
DISPLAY = {"plain": "plain（基线模型）", "control": "control（GAP 头）",
           "learned": "learned（仅特征门控）", "full": "full（特征+输入线索）"}
KINDS = ("haze", "cloud", "glare")
FAMILIES = {
    "clean": ["clean"],
    "matched single": [f"matched_{k}_{s}" for k in KINDS for s in range(1, 6)],
    "shifted single": [f"shifted_{k}_{s}" for k in KINDS for s in range(1, 6)],
    "matched combination": [f"matched_{k}_{s}" for k in ("haze+glare", "haze+cloud") for s in (3, 5)],
    "shifted combination": [f"shifted_{k}_{s}" for k in ("haze+glare", "haze+cloud") for s in (3, 5)],
}
PROBE = [("clean", {"kind": "clean", "severity": 0, "domain": "matched", "placement": "random"}),
         ("haze_5", {"kind": "haze", "severity": 5, "domain": "matched", "placement": "random"}),
         ("glare_5", {"kind": "glare", "severity": 5, "domain": "matched", "placement": "random"}),
         ("haze+glare_5", {"kind": "haze+glare", "severity": 5, "domain": "matched", "placement": "random"}),
         ("haze+cloud_5", {"kind": "haze+cloud", "severity": 5, "domain": "matched", "placement": "random"})]


def load(rel):
    run = PROJECT / rel
    summary = json.loads((run / "evaluation/summary.json").read_text())
    preds = {c: read_jsonl(run / "evaluation" / f"{c}.jsonl") for c in summary}
    complete = json.loads((run / "complete.json").read_text())
    return {"run": run, "summary": summary, "preds": preds, "complete": complete}


def family_mean(summary, conditions):
    return float(np.mean([summary[c]["macro_f1"] for c in conditions]))


def retention(preds, conditions):
    clean = {r["sample_id"]: r for r in preds["clean"]}
    total = survived = 0
    for c in conditions:
        for r in preds[c]:
            if clean[r["sample_id"]]["prediction"] == r["label"]:
                total += 1
                survived += int(r["prediction"] == r["label"])
    return survived / total if total else None


def pct(x, digits=1):
    return "—" if x is None else f"{100 * x:.{digits}f}"


def table(rows, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def gate_stats(mode, rows):
    ck = torch.load(PROJECT / RUNS[mode] / "best.pt",
                    map_location="cpu", weights_only=False)
    model = build_model("swin_t", 8, mode, False)
    model.load_state_dict(ck["model"])
    model = model.cuda().eval()
    out = {"reliability_weight_norm": float(model.rgpa.reliability.weight.norm()),
           "cue_weight_norm": float(model.rgpa.cue.weight.norm())}
    cond = dict(PROBE[3][1])
    ds = ShipDataset(rows, condition=cond)
    x = torch.stack([ds[i][0] for i in range(len(rows))]).cuda()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        f = model.norm(model.features(x)).permute(0, 3, 1, 2).float()
        logits = model.rgpa.reliability(f)
        if mode == "full":
            logits = logits + model.rgpa.cue(input_cues(x, f.shape[-2:]))
        w = torch.softmax(logits.flatten(2), dim=2).view_as(logits)
    entropy = (-(w.clamp_min(1e-12).log() * w).sum((2, 3))).mean() / np.log(w.shape[-1] * w.shape[-2])
    saturation = input_cues(x, f.shape[-2:])[:, 0].float()
    out.update(logit_std=float(logits.std()), entropy_ratio=float(entropy),
               max_token_share_mean=float(w.flatten(1).max(1).values.mean()),
               attention_saturation_corr=float(np.corrcoef(
                   w.flatten(1).float().cpu().numpy().ravel(),
                   saturation.flatten(1).cpu().numpy().ravel())[0, 1]))
    return out, model


def probe(model, rows):
    """Macro-F1 under attention temperature tau and residual share alpha."""
    y = torch.tensor([r["label"] for r in rows])
    result = {}
    for name, cond in PROBE:
        ds = ShipDataset(rows, condition=cond)
        x = torch.stack([ds[i][0] for i in range(len(rows))]).cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            f = model.norm(model.features(x)).permute(0, 3, 1, 2).float()
            logits = model.rgpa.reliability(f) + model.rgpa.cue(input_cues(x, f.shape[-2:]))
        row = {}
        for tau in (0.5, 1.0):
            w = torch.softmax((logits / tau).flatten(2), dim=2).view_as(logits)
            for alpha in (0.5, 1.0):
                out = model.rgpa.classifier(alpha * (f * w).sum((2, 3)) + (1 - alpha) * f.mean((2, 3)))
                r = metrics(y.tolist(), out.argmax(1).cpu().tolist(), 8)
                row[f"tau{tau}_alpha{alpha}"] = 100 * r["macro_f1"]
        result[name] = row
    return result


def main():
    data = {k: load(v) for k, v in RUNS.items()}
    baseline = load(BASELINE)
    parity = {
        "identical_predictions": all(
            [r["prediction"] for r in data["plain"]["preds"][c]] ==
            [r["prediction"] for r in baseline["preds"][c]] for c in baseline["summary"]),
        "identical_probabilities": all(
            np.array_equal([r["probabilities"] for r in data["plain"]["preds"][c]],
                           [r["probabilities"] for r in baseline["preds"][c]])
            for c in baseline["summary"]),
        "max_macro_f1_delta_pp": 100 * max(
            abs(data["plain"]["summary"][c]["macro_f1"] - baseline["summary"][c]["macro_f1"])
            for c in baseline["summary"]),
    }

    rows = [r for r in read_jsonl(PROJECT / "data/manifests/fgsc23_research_civilian.jsonl")
            if r["split"] == "test"]
    diagnostics, probe_table = {}, {}
    with Budget("evaluation"):
        for mode in ("learned", "full"):
            stats, model = gate_stats(mode, rows)
            diagnostics[mode] = stats
            if mode == "full":
                probe_table = probe(model, rows)

    summary = {"parity": parity, "runs": {}, "gate": diagnostics, "probe": probe_table}
    for name, d in data.items():
        entry = {"path": RUNS[name], "clean": family_mean(d["summary"], ["clean"]),
                 "clean_accuracy": d["summary"]["clean"]["accuracy"],
                 "worst": min(d["summary"], key=lambda c: d["summary"][c]["macro_f1"]),
                 "worst_macro_f1": min(v["macro_f1"] for v in d["summary"].values()),
                 "val_macro_f1": d["complete"]["best_val_macro_f1"],
                 "minutes": d["complete"]["training_wall_seconds"] / 60,
                 "parameters": d["complete"]["parameter_count"],
                 "gate_parameters": d["complete"]["gate_parameter_count"]}
        for family, conds in FAMILIES.items():
            if family != "clean":
                entry[family] = family_mean(d["summary"], conds)
                entry[f"{family} retention"] = retention(d["preds"], conds)
        summary["runs"][name] = entry

    seed_entries = {}
    for seed in PLAIN_SEEDS:
        rel = plain_path(seed)
        if not (PROJECT / rel / "evaluation/summary.json").exists():
            continue
        d = load(rel)
        seed_entries[seed] = {
            "clean": family_mean(d["summary"], ["clean"]),
            "matched single": family_mean(d["summary"], FAMILIES["matched single"]),
            "matched combination": family_mean(d["summary"], FAMILIES["matched combination"]),
            "shifted combination": family_mean(d["summary"], FAMILIES["shifted combination"]),
            "worst": min(v["macro_f1"] for v in d["summary"].values()),
            "val": d["complete"]["best_val_macro_f1"]}
    fields = ("clean", "matched single", "matched combination", "shifted combination", "worst", "val")
    noise = {}
    for field in fields:
        values = np.array([e[field] for e in seed_entries.values()])
        noise[field] = {"mean": float(values.mean()),
                        "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
                        "min": float(values.min()), "max": float(values.max())}
    summary["plain_seeds"] = {str(k): v for k, v in seed_entries.items()}
    summary["noise"] = noise
    write_json(PROJECT / "reports/rgpa_round1.json", summary)

    main_rows = []
    for name in RUNS:
        e = summary["runs"][name]
        main_rows.append([DISPLAY[name], pct(e["clean"]), pct(e["matched single"]),
                          pct(e["shifted single"]), pct(e["matched combination"]),
                          pct(e["shifted combination"]), pct(e["matched combination retention"]),
                          f"{e['worst']} ({pct(e['worst_macro_f1'])})",
                          pct(e["val_macro_f1"])])
    delta_rows = []
    base = summary["runs"]["plain"]
    for name in ("control", "learned", "full"):
        e = summary["runs"][name]
        delta_rows.append([DISPLAY[name],
                           f"{100 * (e['clean'] - base['clean']):+.1f}",
                           f"{100 * (e['matched single'] - base['matched single']):+.1f}",
                           f"{100 * (e['matched combination'] - base['matched combination']):+.1f}",
                           f"{100 * (e['matched combination retention'] - base['matched combination retention']):+.1f}",
                           f"{100 * (e['shifted combination'] - base['shifted combination']):+.1f}"])
    gate_rows = [[DISPLAY[m], f"{s['reliability_weight_norm']:.3f}", f"{s['cue_weight_norm']:.4f}",
                  f"{s['logit_std']:.3f}", f"{s['entropy_ratio']:.4f}",
                  f"{s['max_token_share_mean']:.4f}（均匀 {1/49:.4f}）",
                  f"{s['attention_saturation_corr']:+.3f}"]
                 for m, s in diagnostics.items()]
    probe_rows = [[name, *[f"{probe_table[name][k]:.1f}" for k in
                           ("tau1.0_alpha0.5", "tau1.0_alpha1.0", "tau0.5_alpha0.5")]]
                  for name, _ in PROBE]
    control_entry = summary["runs"]["control"]
    seed_rows = [[str(seed), pct(e["clean"]), pct(e["matched single"]), pct(e["matched combination"]),
                  pct(e["shifted combination"]), pct(e["worst"]), pct(e["val"])]
                 for seed, e in seed_entries.items()]
    seed_rows.append(["均值 ± sd"] + [
        f"{100 * noise[f]['mean']:.1f} ± {100 * noise[f]['sd']:.1f}" for f in fields])
    seed_rows.append(["范围"] + [
        f"{100 * noise[f]['min']:.1f}–{100 * noise[f]['max']:.1f}" for f in fields])
    noise_rows = []
    for name in ("control", "learned", "full"):
        e = summary["runs"][name]
        cells = []
        for field in ("clean", "matched single", "matched combination"):
            delta = e[field] - noise[field]["mean"]
            cells.append(f"{100 * delta:+.1f}（{delta / noise[field]['sd']:+.1f} sd）")
        noise_rows.append([DISPLAY[name], *cells])

    text = f"""# RGPA 第一轮：可靠性门控部件聚合

## 0. 结论先说

本轮模块**没有通过**项目规则“新模块必须在相同数据上打赢退化增强基线”。对照用的是
`plain`（冻结工程的原始 Swin-Tiny，同数据、同增强）的三个种子（17/29/43）。
没有一档变体在干净、单退化、组合退化上同时达到基线水平：`full` 与 `control` 的
matched 组合均值略高，但干净与单退化低 1–2 个百分点；`learned` 三项都低。
详见第 3 节的三种子噪声对照。诊断显示门控基本没有学起来，
且“更集中地加权 token”的方向本身有害（见第 5 节）。因此不进入多骨干扩展，也不据此
调整数据集或配方。

## 1. 复现检验

`plain` 模式直接调用冻结工程的 `build_model`，通过新脚本的同一训练循环训练。
在全部 43 个评测条件下，它与既有基线 `swin_t_combo_s17` 的逐样本预测
{"完全一致" if parity["identical_predictions"] else "不一致"}，Macro-F1 最大差
{parity["max_macro_f1_delta_pp"]:.3f} pp，概率
{"逐元素相同" if parity["identical_probabilities"] else "存在差异"}。训练/评估管线的复现成立，
下面的差异只可能来自模型本身。

Metric 定义（首次出现）：组合保持率 = P(退化后答对 | 干净答对)，分母为
该模型干净集答对样本数 × 4 个 matched 组合条件；分母随模型不同，不是跨模型配对检验。

## 2. 主结果

{table(main_rows, ["运行", "干净", "单退化 matched", "单退化 shifted", "组合 matched", "组合 shifted", "组合保持率", "最差条件", "最佳 val"])}

相对 `plain` 的变化（pp）：

{table(delta_rows, ["运行", "Δ干净", "Δ单退化", "Δ组合", "Δ组合保持率", "Δshifted 组合"])}

注意上表的差值都以 seed 17 的 `plain` 为参照。第 3 节表明组合退化的种子间波动远大于
这些差值，因此增减几个百分点不能直接读作模块效果。

## 3. 种子噪声基线（plain，17/29/43）

同一模型、同一配方、只换随机种子。这是判断任何模块差值是否可解释的前提。

{table(seed_rows, ["seed", "干净", "单退化 matched", "组合 matched", "组合 shifted", "最差条件", "最佳 val"])}

三档变体（均为 seed 17）相对 `plain` 三种子均值的差，括号内为以种子标准差为单位的距离：

{table(noise_rows, ["运行", "Δ干净", "Δ单退化", "Δ组合"])}

种子噪声在组合退化上最大：sd {100 * noise['matched combination']['sd']:.1f} pp、范围
{100 * noise['matched combination']['min']:.1f}–{100 * noise['matched combination']['max']:.1f}，
而干净与单退化的 sd 约 {100 * noise['clean']['sd']:.1f} pp。由此：

- `full` 的组合 +1.4 pp 只有 {abs((summary['runs']['full']['matched combination'] - noise['matched combination']['mean']) / noise['matched combination']['sd']):.1f} sd，
  不能算收益；它在干净/单退化上的负值反而更远离均值。
- `control` 与 `learned` 在干净、单退化上低 1.0–3.0 pp（1.9–5.1 sd），属于无收益或负面。
- 组合退化的 6.1 pp 种子极差比任何模块差异都大，单种子组合结论不可用。
- 选型指标（干净 val）与组合测试表现不同序：seed 17 val 最低但组合最高，seed 43 相反。
  这不是模块问题，但说明按干净 val 选 checkpoint 会放大组合评估的种子噪声。

三个种子只能给出粗略量级，sd 本身也有很大不确定性；但它足以否定 <2 pp 的单种子“收益”。

## 4. 门控是否学起来

在全部 236 张测试图、五级雾+眩光上统计（注意力按 softmax 计算）：

{table(gate_rows, ["运行", "‖W_rel‖", "‖W_cue‖", "logit 标准差", "权重熵比（1=均匀）", "最大 token 权重均值", "corr(注意力, 饱和)"])}

可靠性权重范数只有 ~0.10（初始为 0），输入线索卷积范数 ~0.009；49 个 token 的权重熵比
约 0.98，最大 token 权重约为均匀值的 2.2 倍。门控有方向（与饱和区域负相关），但幅度很小，
功能上仍接近全局平均池化。

## 5. 注意力方向本身是否有用

对已训练好的 `full` 模型，在测试时改变注意力温度 τ（越小越集中）与注意力占比 α
（α=1 表示完全丢掉 GAP 残差），报告 Macro-F1：

{table(probe_rows, ["条件", "τ=1, α=0.5（训练配置）", "τ=1, α=1（纯注意力）", "τ=0.5, α=0.5（更集中）"])}

α 从 0.5 提到 1.0 时，干净与雾+眩光同时下降；τ 变小也没有一致收益。也就是说，问题不是
“门控没训够”，而是学到的 token 重加权不如均匀平均。注意力与饱和区域确实负相关
（见门控诊断表最后一列），但压低这些 token 的同时也丢掉了其中仍可用的目标轮廓信息。

## 6. 为什么这一版不成立

- 骨干最后一层是 7×7 token（每 token 约 32×32 像素），而组合退化的破坏是局部的；
  一个 token 内既有被毁的部件也有幸存的部件，池化层面的重加权无法把它们分开。
- 残差 GAP 让初始函数等于基线，但也把注意力效应减半；训练信号弱，40 epoch 内门控停在
  近均匀状态，新增参数实际未参与决策。
- 诊断页已经指出失败与全局退化参数在同严重度内几乎无关；本模块仍只依据“局部是否像被
  破坏”来加权，等于用一个弱代理去近似一个内容相关的决策，信息量不足。

## 7. 下一步（需先满足同一门槛）

- 噪声基线已补齐（第 3 节）。后续任何模块都必须以三种子均值和 sd 报告结果，
  单种子差值不再作为结论；组合退化上的“收益”至少要超过 3 pp（约 1 sd）才值得复跑。
- 若继续做模块，优先选择能保留局部证据而不是只重加权的机制（例如多尺度池化或
  测试时多视图融合），并且仍以“同数据打赢组合增强基线”为唯一准入条件。
- 不再往 RGPA 这一族叠加参数；其失败模式（池化层看不到 token 内部）不是调参能解决的。

## 复现

```bash
.venv/bin/python scripts/train_rgpa.py --manifest data/manifests/fgsc23_research_civilian.jsonl --backbone swin_t --mode plain
.venv/bin/python scripts/train_rgpa.py --manifest data/manifests/fgsc23_research_civilian.jsonl --backbone swin_t --mode control
.venv/bin/python scripts/train_rgpa.py --manifest data/manifests/fgsc23_research_civilian.jsonl --backbone swin_t --mode learned
.venv/bin/python scripts/train_rgpa.py --manifest data/manifests/fgsc23_research_civilian.jsonl --backbone swin_t --mode full
.venv/bin/python scripts/rgpa_report.py
```
"""
    (PROJECT / "reports/RGPA_ROUND1.md").write_text(text)
    print(json.dumps({"parity": parity,
                      "runs": {k: round(v["clean"], 4) for k, v in summary["runs"].items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
