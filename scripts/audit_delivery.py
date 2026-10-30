#!/usr/bin/env python
"""Verify available artifacts and report pending acceptance items without inventing results."""
import collections
import json
from pathlib import Path

from maritime_robust.common import ROOT, digest, directory_bytes, read_jsonl, write_json
from maritime_robust.degradations import conditions
from maritime_robust.engine import metrics


def audit():
    errors=[];pending=[];checked=[]
    for path in sorted((ROOT/"data/manifests").glob("fgsc23_research_*.jsonl")):
        rows=read_jsonl(path)
        for a,b in (("train","val"),("train","test"),("val","test")):
            if {r["source_group_id"] for r in rows if r["split"]==a}&{r["source_group_id"] for r in rows if r["split"]==b}:
                errors.append(str(path)+" leaks "+a+"/"+b)
        checked.append({"manifest":str(path.relative_to(ROOT)),"sha256":digest(path),"samples":len(rows)})
    expected={c["name"] for c in conditions()}
    completed=[]
    for path in sorted((ROOT/"runs").glob("fgsc23_*/*/complete.json")):
        config=json.loads(path.read_text());run=path.parent
        if digest(run/"best.pt")!=config["checkpoint_sha256"]:
            errors.append(str(run)+" checkpoint hash mismatch")
        if config.get("augmentation")=="combined":
            if digest(ROOT/"configs/combined_augmentation.json")!=config.get("combined_recipe_sha256"):
                errors.append(str(run)+" combined recipe hash mismatch")
            # Reporting code may change after a run; these modules define training/evaluation.
            for name in ("engine.py","degradations.py","common.py","cli.py","__init__.py"):
                relative=f"src/maritime_robust/{name}"
                if config["source_sha256"].get(relative)!=digest(ROOT/relative):
                    errors.append(str(run)+" training source mismatch "+relative)
        summary_path=run/"evaluation/summary.json"
        if not summary_path.exists():
            pending.append(str(run.relative_to(ROOT))+": evaluation pending")
            continue
        scores=json.loads(summary_path.read_text())
        if set(scores)!=expected:
            errors.append(str(run)+": missing/extra conditions")
        for name,score in scores.items():
            preds=read_jsonl(run/f"evaluation/{name}.jsonl")
            actual=metrics([r["label"] for r in preds],[r["prediction"] for r in preds],config["num_classes"])
            if abs(score["macro_f1"]-actual["macro_f1"])>1e-12:
                errors.append(str(run)+" score mismatch "+name)
            if any(abs(sum(r["probabilities"])-1)>1e-4 for r in preds):
                errors.append(str(run)+" invalid probabilities "+name)
        if config["seed"]==17 and config["protocol"]=="research" and not (run/"cost.json").exists():
            pending.append(str(run.relative_to(ROOT))+": cost profile pending")
        completed.append(str(run.relative_to(ROOT)))
    queue=ROOT/"reports/queue_status.json"
    q=json.loads(queue.read_text()) if queue.exists() else {"status":"not_started"}
    if q["status"]!="complete":pending.append("First-round queue status: "+q["status"])
    combined_queue=ROOT/"reports/combined_queue_status.json"
    cq=json.loads(combined_queue.read_text()) if combined_queue.exists() else {"status":"not_started"}
    if cq["status"]!="complete":pending.append("Combined baseline queue status: "+cq["status"])
    if not (ROOT/"reports/frozen_selection.json").exists():pending.append("Validation selection not frozen")
    if not (ROOT/"reports/fgsrcs_audit.json").exists():pending.append("FGSRCS admission incomplete")
    size=directory_bytes(ROOT)
    if size>50*10**9:errors.append("Project exceeds 50GB")
    budget=ROOT/"artifacts/budget.json"
    state=json.loads(budget.read_text()) if budget.exists() else {"jobs":[]}
    total=sum(j["elapsed_seconds"] for j in state["jobs"])/3600
    if total>72:errors.append("GPU budget exceeded")
    result={"status":"failed" if errors else "in_progress" if pending else "complete",
            "errors":errors,"pending":pending,"completed_evaluated_runs":completed,
            "combined_evaluated_runs":[run for run in completed if "_combo_" in run],
            "checked_manifests":checked,"disk_bytes":size,"charged_gpu_wall_hours":total,
            "queue":q,"combined_queue":cq,
            "external_limitations":["Domain-specific method reproduction depends on author implementation availability.",
                                    "Generative feasibility and real-data annotation status are separate from synthetic benchmark completion."]}
    write_json(ROOT/"reports/delivery_audit.json",result)
    return result


if __name__=="__main__":
    print(json.dumps(audit(),ensure_ascii=False,indent=2))
