import collections
import csv
import json
from pathlib import Path

import numpy as np

from .common import ROOT, digest, read_jsonl, write_json
from .engine import metrics


def grouped_f1_interval(records,n,repeats=300,seed=20261027):
    groups=collections.defaultdict(list)
    for r in records:
        groups[r["source_group_id"]].append(r)
    matrices=[]
    for members in groups.values():
        matrix=np.zeros((n,n),dtype=float)
        for r in members:
            matrix[r["label"],r["prediction"]]+=1
        matrices.append(matrix)
    matrices=np.asarray(matrices)
    rng=np.random.default_rng(seed)
    scores=[]
    for _ in range(repeats):
        count=np.bincount(rng.integers(len(matrices),size=len(matrices)),minlength=len(matrices))
        cm=np.einsum("g,gij->ij",count,matrices)
        denom=cm.sum(0)+cm.sum(1)
        f1=np.divide(2*cm.diagonal(),denom,out=np.zeros(n),where=denom>0)
        scores.append(float(f1.mean()))
    return {"low":float(np.quantile(scores,.025)),"high":float(np.quantile(scores,.975)),
            "repeats":repeats,"group_count":len(groups),"group_unit":"source_group_id (duplicate component if scene unavailable)",
            "method":"unstratified cluster percentile bootstrap, fixed label set"}


def summarize():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs=[]
    for complete in sorted((ROOT/"runs").glob("fgsc23_*/*/complete.json")):
        run=complete.parent
        config=json.loads(complete.read_text())
        summaries=run/"evaluation/summary.json"
        if not summaries.exists():
            continue
        scores=json.loads(summaries.read_text())
        clean=scores["clean"]["macro_f1"]
        single=[v["macro_f1"] for k,v in scores.items() if v["condition"]["domain"]=="matched" and v["condition"]["kind"] in ("haze","cloud","glare") and v["condition"]["placement"]=="random"]
        combination=[v["macro_f1"] for k,v in scores.items() if v["condition"]["domain"]=="matched" and "+" in v["condition"]["kind"]]
        shifted_combination=[v["macro_f1"] for k,v in scores.items() if v["condition"]["domain"]=="shifted" and "+" in v["condition"]["kind"]]
        augmentation=config.get("augmentation") or ("single" if config["augmented"] else "clean")
        item={"run":str(run.relative_to(ROOT)),"model":config["model"],"scope":config["scope"],
              "protocol":config["protocol"],"augmented":config["augmented"],"augmentation":augmentation,"seed":config["seed"],
              "clean_macro_f1":clean,"matched_single_macro_f1":float(np.mean(single)) if single else None,
              "matched_combination_macro_f1":float(np.mean(combination)) if combination else None,
              "shifted_combination_macro_f1":float(np.mean(shifted_combination)) if shifted_combination else None,
              "worst_macro_f1":min(v["macro_f1"] for v in scores.values()),
              "gpu_wall_hours":config["training_wall_seconds"]/3600}
        history_path=run/'history.json'
        if history_path.exists():item['epoch_loop_wall_hours']=sum(e['seconds'] for e in json.loads(history_path.read_text()))/3600
        costs=run/"cost.json"
        if costs.exists():
            item.update(json.loads(costs.read_text()))
        runs.append(item)
        diagnoses={}
        clean_preds=read_jsonl(run/"evaluation/clean.jsonl")
        clean_by_id={r["sample_id"]:r for r in clean_preds}
        for name,score in scores.items():
            preds=read_jsonl(run/f"evaluation/{name}.jsonl")
            rebuilt=metrics([r["label"] for r in preds],[r["prediction"] for r in preds],config["num_classes"])
            if abs(rebuilt["macro_f1"]-score["macro_f1"])>1e-12:
                raise AssertionError("Metrics do not match saved predictions")
            flips=[r for r in preds if clean_by_id[r["sample_id"]]["prediction"]==r["label"] and r["prediction"]!=r["label"]]
            high=[r for r in preds if r["prediction"]!=r["label"] and max(r["probabilities"])>=.9]
            correlations={}
            wrong=np.asarray([r["prediction"]!=r["label"] for r in preds],float)
            for field in ("saturated_fraction","center_proxy_affected_fraction"):
                values=np.asarray([r["degradation"].get(field) or 0 for r in preds])
                correlations[field]=float(np.corrcoef(values,wrong)[0,1]) if values.std()>1e-8 and wrong.std()>1e-8 else None
            sizes=np.asarray([r["original_width"]*r["original_height"] for r in preds],float)
            correlations["log_original_pixel_area"]=float(np.corrcoef(np.log1p(sizes),wrong)[0,1]) if sizes.std()>1e-8 and wrong.std()>1e-8 else None
            cm=np.asarray(score["confusion_matrix"])
            confusions=[{"true_label":int(i),"predicted_label":int(j),"count":int(cm[i,j])}
                        for i,j in np.ndindex(cm.shape) if i!=j and cm[i,j]>0]
            confusions.sort(key=lambda r:-r["count"])
            diagnoses[name]={"macro_f1_drop":clean-score["macro_f1"],
                             "macro_f1_95ci":grouped_f1_interval(preds,config["num_classes"]),
                             "clean_correct_to_wrong":len(flips),"high_confidence_errors":len(high),
                             "error_correlations_descriptive_only":correlations,
                             "largest_confusions":confusions[:10],
                             "examples":[r["sample_id"] for r in sorted(high,key=lambda r:-max(r["probabilities"]))[:12]]}
            by_size={}
            for label,low,high_bound in [('short_side_lt96',0,96),('short_side_96_191',96,192),('short_side_ge192',192,float('inf'))]:
                subset=[r for r in preds if low<=min(r['original_width'],r['original_height'])<high_bound]
                by_size[label]={'n':len(subset),'accuracy':float(np.mean([r['label']==r['prediction'] for r in subset])) if subset else None}
            diagnoses[name]['crop_resolution_proxy_bins']=by_size
            cm=np.array(score['confusion_matrix']);np.fill_diagonal(cm,0)
            top=np.argsort(cm.ravel())[::-1][:5]
            diagnoses[name]['top_confusions']=[{'true':config['classes'].get(str(int(index//len(cm))),str(index//len(cm))),
                                                'predicted':config['classes'].get(str(int(index%len(cm))),str(index%len(cm))),
                                                'count':int(cm.ravel()[index])} for index in top if cm.ravel()[index]>0]
        write_json(run/"diagnostics.json",diagnoses)
        fig,ax=plt.subplots(figsize=(6,4))
        for kind in ("haze","cloud","glare"):
            vals=[scores.get(f"matched_{kind}_{i}",{}).get("macro_f1",np.nan) for i in range(1,6)]
            ax.plot(range(6),[clean]+vals,marker="o",label=kind)
        ax.set(xlabel="Severity (0 = clean)",ylabel="Macro-F1",ylim=(0,1),title=f"{config['model']} / {config['scope']} / {augmentation} / seed {config['seed']}")
        ax.legend();fig.tight_layout();fig.savefig(run/"degradation_curves.png",dpi=160);plt.close(fig)
        dcp_summary=run/"evaluation_dcp/summary.json"
        if dcp_summary.exists():
            dscores=json.loads(dcp_summary.read_text())
            dcombination=[v["macro_f1"] for v in dscores.values() if v["condition"]["domain"]=="matched" and "+" in v["condition"]["kind"]]
            dshifted=[v["macro_f1"] for v in dscores.values() if v["condition"]["domain"]=="shifted" and "+" in v["condition"]["kind"]]
            ditem={"run":str(run.relative_to(ROOT))+"/DCP","model":config["model"],"scope":config["scope"],
                   "protocol":config["protocol"],"augmented":False,"augmentation":"clean","restoration":"DCP","seed":config["seed"],
                   "clean_macro_f1":dscores["clean"]["macro_f1"],
                   "matched_single_macro_f1":float(np.mean([v["macro_f1"] for v in dscores.values() if v["condition"]["domain"]=="matched" and v["condition"]["kind"] in ("haze","cloud","glare") and v["condition"]["placement"]=="random"])),
                   "matched_combination_macro_f1":float(np.mean(dcombination)) if dcombination else None,
                   "shifted_combination_macro_f1":float(np.mean(dshifted)) if dshifted else None,
                   "worst_macro_f1":min(v["macro_f1"] for v in dscores.values()),
                   "gpu_wall_hours":config["training_wall_seconds"]/3600}
            if (run/"cost_dcp.json").exists():ditem.update(json.loads((run/"cost_dcp.json").read_text()))
            runs.append(ditem)
            comparison={name:{"macro_f1_difference":v["macro_f1"]-scores[name]["macro_f1"],
                              "accuracy_difference":v["accuracy"]-scores[name]["accuracy"]} for name,v in dscores.items()}
            write_json(run/"dcp_comparison.json",comparison)
    grouped=collections.defaultdict(list)
    for r in runs:
        grouped[(r["protocol"],r["scope"],r["model"],r["augmentation"],r.get("restoration"))].append(r)
    aggregates=[]
    for (protocol,scope,model,augmentation,restoration),members in grouped.items():
        item={"protocol":protocol,"scope":scope,"model":model,"augmented":augmentation!="clean",
              "augmentation":augmentation,"restoration":restoration,"seeds":[r["seed"] for r in members]}
        for metric in ("clean_macro_f1","matched_single_macro_f1","matched_combination_macro_f1","shifted_combination_macro_f1","worst_macro_f1"):
            values=[r[metric] for r in members if r.get(metric) is not None]
            item[metric+"_mean"]=float(np.mean(values)) if values else None
            item[metric+"_std"]=float(np.std(values,ddof=1)) if len(values)>1 else None
        aggregates.append(item)
    paired=[]
    for clean_run in runs:
        if clean_run["augmentation"]!="clean" or clean_run.get('restoration'):continue
        for mode in ("single","combined"):
            matches=[r for r in runs if r["augmentation"]==mode and all(r[k]==clean_run[k] for k in ("protocol","scope","model","seed"))]
            if not matches:continue
            aug_run=matches[0]
            matched_gain=aug_run["matched_single_macro_f1"]-clean_run["matched_single_macro_f1"]
            combination_gain=(aug_run["matched_combination_macro_f1"]-clean_run["matched_combination_macro_f1"]
                              if aug_run["matched_combination_macro_f1"] is not None and clean_run["matched_combination_macro_f1"] is not None else None)
            shifted_gain=(aug_run["shifted_combination_macro_f1"]-clean_run["shifted_combination_macro_f1"]
                          if aug_run["shifted_combination_macro_f1"] is not None and clean_run["shifted_combination_macro_f1"] is not None else None)
            degradation_gap=clean_run["clean_macro_f1"]-clean_run["matched_single_macro_f1"]
            paired.append({"protocol":clean_run["protocol"],"scope":clean_run["scope"],"model":clean_run["model"],
                           "augmentation":mode,"seed":clean_run["seed"],
                           "augmentation_clean_f1_change":aug_run["clean_macro_f1"]-clean_run["clean_macro_f1"],
                           "augmentation_matched_f1_change":matched_gain,
                           "augmentation_combination_f1_change":combination_gain,
                           "augmentation_shifted_combination_f1_change":shifted_gain,
                           "recovery_of_clean_model_average_drop":matched_gain/degradation_gap if degradation_gap>0 else None})
    dcp_comparisons=[]
    for base in runs:
        directory=ROOT/base["run"]
        dcp_path=directory/"evaluation_dcp/summary.json"
        if not dcp_path.exists():continue
        original=json.loads((directory/"evaluation/summary.json").read_text())
        restored=json.loads(dcp_path.read_text())
        changes={name:restored[name]["macro_f1"]-value["macro_f1"] for name,value in original.items() if name in restored}
        haze=[change for name,change in changes.items() if name.startswith("matched_haze_") and "+" not in name]
        dcp_comparisons.append({"run":base["run"],"clean_f1_change":changes.get("clean"),
                                "matched_haze_mean_f1_change":float(np.mean(haze)) if haze else None,
                                "condition_changes":changes})
    write_json(ROOT/"reports/results.json",{"runs":runs,"aggregates":aggregates,"augmentation_pairs":paired,"dcp_comparisons":dcp_comparisons})
    if aggregates:
        with (ROOT/'reports/results.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(aggregates[0]));writer.writeheader();writer.writerows(aggregates)
    lines=["# 实验结果（首轮 + 组合增强基线）", "", "所有数值均来自本地保存的预测；以下不是论文原始成绩。"
           "训练列区分干净、单退化增强与组合增强；组合增强是第二轮基线，不参与首轮架构排名。", "",
           "| 协议 | 类别 | 模型 | 训练 | 种子数 | 干净 Macro-F1 (%) | 单退化平均 Macro-F1 (%) | matched 组合平均 Macro-F1 (%) | 最差条件 Macro-F1 (%) |",
           "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for r in aggregates:
        def fmt(k):
            val=r[k+"_mean"]
            std=r[k+"_std"]
            return "—" if val is None else f"{100*val:.2f}"+(f" ± {100*std:.2f}" if std is not None else "（单种子）")
        training='DCP + 干净' if r.get('restoration') else {'clean':'干净','single':'单退化增强','combined':'组合增强'}[r['augmentation']]
        lines.append(f"| {r['protocol']} | {r['scope']} | {r['model']} | {training} | {len(r['seeds'])} | {fmt('clean_macro_f1')} | {fmt('matched_single_macro_f1')} | {fmt('matched_combination_macro_f1')} | {fmt('worst_macro_f1')} |")
    write_json(ROOT/'reports/paired_augmentation_changes.json',paired)
    lines.extend(["", "不确定性：重复图分组不等于按港口/船只划分；模拟眩光与雾均为近似模型。中心区域覆盖率不是船体遮挡率。"
                  "组合增强只在 matched 域、3/5 级训练；shifted 组合仍是分布偏移测试，不能据此声称真实组合泛化。", "",
                  "逐样本概率、参数与混淆矩阵位于各 runs 子目录；diagnostics.json 包含按原图组 bootstrap 的置信区间和高置信错误。",
                  "真实退化数据是否可独立验证，见 reports/fgsrcs_audit.json 和 docs/DATA_PROTOCOL.md。"])
    if paired:
        lines.extend(["", "## 相同架构与种子的增强对照", "", "下列变化是增强版本减干净训练版本，单位为 Macro-F1 百分点，不代表新模块收益。", "",
                      "| 类别 | 模型 | 训练 | 种子 | 干净图变化 | 单退化平均变化 | matched 组合平均变化 |", "|---|---|---|---:|---:|---:|---:|"])
        for r in paired:
            training={'single':'单退化增强','combined':'组合增强'}[r['augmentation']]
            combo='—' if r.get('augmentation_combination_f1_change') is None else f"{100*r['augmentation_combination_f1_change']:+.2f}"
            lines.append(f"| {r['scope']} | {r['model']} | {training} | {r['seed']} | {100*r['augmentation_clean_f1_change']:+.2f} | {100*r['augmentation_matched_f1_change']:+.2f} | {combo} |")
    if dcp_comparisons:
        lines.extend(["", "## DCP 预处理对照", "", "| 运行 | 干净图 F1 变化/百分点 | 五级 matched 雾平均 F1 变化/百分点 |", "|---|---:|---:|"])
        for r in dcp_comparisons:
            lines.append(f"| {r['run']} | {100*r['clean_f1_change']:+.2f} | {100*r['matched_haze_mean_f1_change']:+.2f} |")
    lines.extend(['','## 同设备成本（种子 17）','','端到端延迟包括读图、缩放、可选 DCP、归一化、传输及推理。FLOPs 为 PyTorch 支持算子的部分计数。','',
                  '| 协议/类别 | 模型/训练 | 参数量 M | 端到端中位 ms | P95 ms | 训练墙钟 GPU h |',
                  '|---|---|---:|---:|---:|---:|'])
    for r in runs:
        if r['seed']!=17 or 'end_to_end_ms_median' not in r:continue
        name=r['model']+('/DCP' if r.get('restoration') else '/combined' if r['augmentation']=='combined' else '/single' if r['augmentation']=='single' else '/clean')
        lines.append(f"| {r['protocol']}/{r['scope']} | {name} | {r['parameters']/1e6:.2f} | {r['end_to_end_ms_median']:.2f} | {r['end_to_end_ms_p95']:.2f} | {r['gpu_wall_hours']:.3f} |")
    (ROOT/"reports/RESULTS.md").write_text("\n".join(lines)+"\n")
    return {"runs":len(runs),"aggregates":len(aggregates)}
