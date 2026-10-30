#!/usr/bin/env python
"""Report the CDLNet branch ablations against the full repaired model.

Reads saved per-condition predictions only. Answers three questions:
  1. does removing the gated self-attention help or hurt?
  2. does removing the crop-refinement branch help or hurt?
  3. does the plain ResNet-50 under the same 256 letterbox pipeline differ
     from the project's 224 baseline (input-protocol control)?
"""
import json

import numpy as np

from maritime_robust.common import ROOT, read_jsonl, write_json

SCOPES = {
    "research_all": "全 23 类",
    "research_civilian": "民船 8 类",
}
KINDS = ("haze", "cloud", "glare")
MATCHED_SINGLE = [f"matched_{k}_{s}" for k in KINDS for s in range(1, 6)]
SHIFTED_SINGLE = [f"shifted_{k}_{s}" for k in KINDS for s in range(1, 6)]
MATCHED_COMBO = [f"matched_{k}_{s}" for k in ("haze+glare", "haze+cloud") for s in (3, 5)]
SHIFTED_COMBO = [f"shifted_{k}_{s}" for k in ("haze+glare", "haze+cloud") for s in (3, 5)]
FAMILIES = {"clean": ["clean"], "matched single": MATCHED_SINGLE,
            "shifted single": SHIFTED_SINGLE, "matched combination": MATCHED_COMBO,
            "shifted combination": SHIFTED_COMBO}
DISPLAY = {"full": "CDLNet 完整", "no_gate": "去门控", "no_region": "去裁剪分支",
           "no_branch": "去两者", "resnet50_256": "ResNet-50(256)",
           "resnet50_224": "ResNet-50(224 基线)"}
ORDER = ["full", "no_gate", "no_region", "no_branch", "resnet50_256", "resnet50_224"]


SPLIT = {"research_all": "all", "research_civilian": "civilian"}


def run_path(scope, key):
    split = SPLIT[scope]
    if key == "resnet50_224":
        return f"runs/fgsc23_research_{split}/resnet50_clean_s17"
    if key == "full":
        return f"runs_external/fgsc23_research_{split}/cdlnet_clean_s17"
    if key == "resnet50_256":
        key = "resnet50"  # queue writes ablation_resnet50_s17
    return f"runs_external/fgsc23_research_{split}/ablation_{key}_s17"


def load(path):
    run = ROOT / path
    if not (run / "evaluation/summary.json").exists():
        return None
    summary = json.loads((run / "evaluation/summary.json").read_text())
    complete = json.loads((run / "complete.json").read_text()) if (run / "complete.json").exists() else {}
    preds = {c: read_jsonl(run / "evaluation" / f"{c}.jsonl") for c in summary}
    return {"summary": summary, "complete": complete, "preds": preds, "path": path}


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


def analyze(scope):
    out = {"scope": scope, "models": {}}
    for key in ORDER:
        data = load(run_path(scope, key))
        if data is None:
            continue
        summary = data["summary"]
        entry = {"path": data["path"],
                 "clean": family_mean(summary, ["clean"]),
                 "worst": min(summary, key=lambda c: summary[c]["macro_f1"]),
                 "worst_macro_f1": min(v["macro_f1"] for v in summary.values())}
        for name, conds in FAMILIES.items():
            if name == "clean":
                continue
            entry[name] = family_mean(summary, conds)
            entry[f"{name} retention"] = retention(data["preds"], conds)
        complete = data["complete"]
        entry["active_parameters"] = complete.get("active_parameter_count",
                                                  complete.get("parameter_count"))
        entry["parameter_count"] = complete.get("parameter_count")
        entry["best_val_macro_f1"] = complete.get("best_val_macro_f1")
        entry["training_minutes"] = (complete.get("training_wall_seconds") or 0) / 60
        out["models"][key] = entry
    return out


def pct(x, digits=1):
    return "—" if x is None else f"{100 * x:.{digits}f}"


def table(rows, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def scope_sections(data, level):
    m = data["models"]
    h = "#" * level
    rows = []
    for key in ORDER:
        e = m.get(key)
        if not e:
            continue
        rows.append([DISPLAY[key],
                     f"{e['active_parameters'] / 1e6:.1f}M" if e.get("active_parameters") else "—",
                     pct(e["clean"]), pct(e["matched single"]), pct(e["shifted single"]),
                     pct(e["matched combination"]), pct(e["shifted combination"]),
                     pct(e["matched combination retention"]),
                     f"{e['worst']} ({pct(e['worst_macro_f1'])})"])
    main = table(rows, ["变体", "激活参数", "干净", "单退化 matched", "单退化 shifted",
                        "组合 matched", "组合 shifted", "组合保持率", "最差条件"])

    rows = []
    for key in ORDER[1:]:
        if key not in m or "full" not in m:
            continue
        e, full = m[key], m["full"]
        rows.append([DISPLAY[key],
                     f"{100 * (e['clean'] - full['clean']):+.1f}",
                     f"{100 * (e['matched single'] - full['matched single']):+.1f}",
                     f"{100 * (e['matched combination'] - full['matched combination']):+.1f}",
                     f"{100 * (e['matched combination retention'] - full['matched combination retention']):+.1f}"])
    delta = table(rows, ["变体", "Δ干净(pp)", "Δ单退化(pp)", "Δ组合(pp)", "Δ组合保持率(pp)"])

    return f"""{h} {SCOPES[data['scope']]}

{main}

相对完整 CDLNet 的变化：

{delta}
"""


def main():
    data = {scope: analyze(scope) for scope in SCOPES}
    write_json(ROOT / "reports/cdlnet_ablation.json", data)

    all_m = data["research_all"]["models"]
    civ_m = data["research_civilian"]["models"]

    def delta(scope_models, a, b, field):
        if a not in scope_models or b not in scope_models:
            return None
        return 100 * (scope_models[a][field] - scope_models[b][field])

    field_cn = {"clean": "干净集", "matched single": "单退化",
                "matched combination": "组合退化"}

    def fact(models, a, b, field):
        d = delta(models, a, b, field)
        if d is None:
            return f"{DISPLAY[a]} 与 {DISPLAY[b]} 的差距无法计算（运行缺失）。"
        better = "高" if d > 0 else "低"
        return f"{DISPLAY[a]}的{field_cn[field]} Macro-F1 比 {DISPLAY[b]}{better} {abs(d):.1f} 个百分点"

    text = f"""# CDLNet 分支消融（research split，seed 17）

全部运行使用同一 CDLNet 数据管线（256×256 letterbox、同一几何增强）、SGD lr 0.01、
cosine 40 epoch、batch 16、同一种子 17；唯一差别是架构。`resnet50_256` 是同一管线下
的纯 ResNet-50 对照，用来分离输入协议与新增分支；`ResNet-50(224 基线)` 是项目原有 224
管线（batch 32、项目自带增强，与 CDLNet 管线不同），只能作为整体协议对照，不能把差值
单独归因于输入分辨率。

说明：`no_gate`/`no_region`/`no_branch` 的 checkpoint 里仍保留未使用模块的参数，
因此表中“激活参数”才是实际参与前向的参数量。组合保持率以各模型干净集正确样本为分母，
分母随模型不同，只能用于同一模型内部的趋势比较，不是严格的跨模型配对检验。

{scope_sections(data['research_all'], 2)}

{scope_sections(data['research_civilian'], 2)}

## 判断

- 23 类：{fact(all_m, 'no_gate', 'full', 'clean')}；{fact(all_m, 'no_gate', 'full', 'matched combination')}。
  {fact(all_m, 'no_region', 'full', 'clean')}；{fact(all_m, 'no_region', 'full', 'matched combination')}。
- 8 类：{fact(civ_m, 'no_gate', 'full', 'clean')}；{fact(civ_m, 'no_gate', 'full', 'matched combination')}。
  {fact(civ_m, 'no_region', 'full', 'clean')}；{fact(civ_m, 'no_region', 'full', 'matched combination')}。
- 输入/训练协议对照（256 vs 224，batch 与增强实现也不同，不能单独归因分辨率）：23 类 {fact(all_m, 'resnet50_256', 'resnet50_224', 'clean')}；
  8 类 {fact(civ_m, 'resnet50_256', 'resnet50_224', 'clean')}。
- 去两者与 ResNet-50(256) 同为 256 管线、唯一架构差别是分类头（GroupFC vs 单层线性）：
  23 类 {fact(all_m, 'no_branch', 'resnet50_256', 'clean')}；8 类 {fact(civ_m, 'no_branch', 'resnet50_256', 'clean')}。
  两个口径方向相反，说明这一差距落在单种子噪声内，既不能归因于输入协议，也不能据此判断分类头优劣。

以上差值都是单种子、单次运行的观测，不能当作显著性结论；组合退化上的差异需结合诊断页的
分组 bootstrap 区间解读。

## 复现

```bash
bash scripts/run_cdlnet_ablation.sh
.venv/bin/python scripts/cdlnet_ablation_report.py
```
"""
    (ROOT / "reports/CDLNET_ABLATION.md").write_text(text)
    for scope, d in data.items():
        print(scope, {k: round(v["clean"], 4) for k, v in d["models"].items()})


if __name__ == "__main__":
    main()
