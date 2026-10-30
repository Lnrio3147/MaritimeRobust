#!/usr/bin/env python
"""Serial, restartable first round. Freeze choices before opening test predictions."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from maritime_robust.common import ROOT, digest, write_json


def run(*args):
    subprocess.run([sys.executable,"-m","maritime_robust.cli",*map(str,args)],cwd=ROOT,check=True)


def run_dir(scope,model,aug,seed,protocol="research"):
    return ROOT/f"runs/fgsc23_{protocol}_{scope}/{model}_{'aug' if aug else 'clean'}_s{seed}"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["all","initial","confirm","test","report"],default="all")
    args=parser.parse_args()
    status={"stage":args.stage,"status":"running","started":time.time(),"pid":__import__('os').getpid()}
    write_json(ROOT/"reports/queue_status.json",status)
    config=json.loads((ROOT/"configs/protocol.json").read_text())
    try:
        if args.stage in ("all","initial"):
            for scope in config["scopes"]:
                for model in config["models"]:
                    for aug in (False,True):
                        job=["train","--manifest",ROOT/f"data/manifests/fgsc23_research_{scope}.jsonl","--model",model,"--seed",17]
                        if aug:job.append("--augmented")
                        run(*job)
                        run("evaluate","--run",run_dir(scope,model,aug,17),"--split","val","--validation-only")
            choices={}
            for scope in config["scopes"]:
                ranking=[]
                for model in config["models"]:
                    values=[]
                    for aug in (False,True):
                        scores=json.loads((run_dir(scope,model,aug,17)/"validation_robust/summary.json").read_text())
                        robust=[v["macro_f1"] for k,v in scores.items() if k!="clean"]
                        values.append(.5*(scores["clean"]["macro_f1"]+sum(robust)/len(robust)))
                    ranking.append({"model":model,"score":sum(values)/len(values)})
                ranking.sort(key=lambda r:(-r["score"],r["model"]))
                choices[scope]={"ranking":ranking,"selected_models":[r["model"] for r in ranking[:2]],"dcp_model":ranking[0]["model"]}
            frozen={"selection":choices,"protocol_sha256":digest(ROOT/"configs/protocol.json"),
                    "recipe_sha256":digest(ROOT/"configs/degradations.json"),"time":time.time(),
                    "basis":"seed17 validation only; equal clean/robust weight; average over clean and augmented variants"}
            frozen_path=ROOT/"reports/frozen_selection.json"
            if frozen_path.exists():
                previous=json.loads(frozen_path.read_text())
                if any(previous[k]!=frozen[k] for k in ("selection","protocol_sha256","recipe_sha256")):
                    raise ValueError("Existing frozen selection differs; do not reselect after test exposure")
            else:
                write_json(frozen_path,frozen)
        if args.stage in ("all","confirm"):
            choices=json.loads((ROOT/"reports/frozen_selection.json").read_text())["selection"]
            for scope in config["scopes"]:
                for model in choices[scope]["selected_models"]:
                    for seed in (29,43):
                        for aug in (False,True):
                            job=["train","--manifest",ROOT/f"data/manifests/fgsc23_research_{scope}.jsonl","--model",model,"--seed",seed]
                            if aug:job.append("--augmented")
                            run(*job)
            # Official label/split references, held training validation explicitly documented.
            for model in config["models"]:
                run("train","--manifest",ROOT/"data/manifests/fgsc23_official_all.jsonl","--model",model,"--seed",17)
        if args.stage in ("all","test"):
            frozen=json.loads((ROOT/"reports/frozen_selection.json").read_text())
            if frozen["protocol_sha256"]!=digest(ROOT/"configs/protocol.json") or frozen["recipe_sha256"]!=digest(ROOT/"configs/degradations.json"):
                raise ValueError("Frozen protocol changed")
            choices=frozen["selection"]
            for scope in config["scopes"]:
                for model in config["models"]:
                    seeds=config["seeds"] if model in choices[scope]["selected_models"] else [17]
                    for seed in seeds:
                        for aug in (False,True):
                            directory=run_dir(scope,model,aug,seed)
                            run("evaluate","--run",directory)
                            if seed==17:
                                run("profile","--run",directory)
                            if model==choices[scope]["dcp_model"] and not aug:
                                run("evaluate","--run",directory,"--restore")
                                if seed==17:run("profile","--run",directory,"--restore")
            for model in config["models"]:
                directory=run_dir("all",model,False,17,protocol="official")
                # Official split references need clean classification only, not a second robustness leaderboard.
                run("evaluate","--run",directory,"--split","val","--validation-only")
                run("evaluate","--run",directory)
        if args.stage in ("all","report"):
            run("report")
        status.update(status="complete",ended=time.time())
    except BaseException as e:
        status.update(status="failed",ended=time.time(),error=repr(e))
        raise
    finally:
        write_json(ROOT/"reports/queue_status.json",status)


if __name__=="__main__":
    main()
