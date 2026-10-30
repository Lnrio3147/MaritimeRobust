import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from .common import ROOT, Budget, digest, environment, read_jsonl, seed_all, stable_seed, write_json, write_jsonl
from .degradations import RECIPE, conditions, corrupt, corruption_seed, dcp

MODEL_WEIGHTS = {
    "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
    "convnext_tiny": models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
    "swin_t": models.Swin_T_Weights.IMAGENET1K_V1,
}

COMBINED_RECIPE_PATH = ROOT/"configs/combined_augmentation.json"


def resolve_augmentation(augmented=False, augmentation=None):
    """Map the legacy boolean and the explicit mode onto one canonical name."""
    if augmentation is None:
        return "single" if augmented else "clean"
    if augmentation not in ("clean", "single", "combined"):
        raise ValueError(f"Unknown augmentation mode: {augmentation}")
    if augmented and augmentation != "single":
        raise ValueError("--augmented only combines with --augmentation single")
    return augmentation


def combined_recipe(path=None):
    """Validated augmentation recipe; ``path`` overrides the frozen default.

    The override exists for controlled probes that change only the sampling
    distribution (e.g. specialist upper-bound runs); every run records which
    file it used and its hash.
    """
    recipe=json.loads(Path(path or COMBINED_RECIPE_PATH).read_text())
    mixture=recipe["degraded_mixture"]
    if set(mixture)!= {"single","combined"} or abs(sum(mixture.values())-1)>1e-9:
        raise ValueError("Degraded mixture must sum to one and name single/combined")
    if not 0<=recipe["clean_probability"]<=1:
        raise ValueError("Invalid clean probability")
    if any(kind not in ("haze","cloud","glare") for kind in recipe["single_kinds"]):
        raise ValueError("Invalid single-degradation kind")
    if any(kind not in ("haze+glare","haze+cloud") for kind in recipe["combined_kinds"]):
        raise ValueError("Invalid combined-degradation kind")
    if any(severity not in range(1,6) for severity in recipe["single_severities"]+recipe["combined_severities"]):
        raise ValueError("Severity must be 1..5")
    return recipe


def build_model(name, num_classes, pretrained=True):
    if name not in MODEL_WEIGHTS:
        raise ValueError(name)
    model = getattr(models, name)(weights=MODEL_WEIGHTS[name] if pretrained else None)
    if name == "resnet50":
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif name == "convnext_tiny":
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
    else:
        model.head = torch.nn.Linear(model.head.in_features, num_classes)
    return model


def metrics(y, pred, n):
    cm = confusion_matrix(y, pred, labels=list(range(n)))
    tp = cm.diagonal().astype(float)
    support = cm.sum(1)
    recall = np.divide(tp, support, out=np.zeros(n), where=support>0)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(n), where=cm.sum(0)>0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(n), where=precision+recall>0)
    return {"n":len(y), "accuracy":float(tp.sum()/max(1,len(y))),
            "macro_f1":float(f1.mean()), "balanced_accuracy":float(recall[support>0].mean()) if (support>0).any() else 0,
            "per_class_recall":recall.tolist(),"per_class_f1":f1.tolist(),
            "class_support":support.tolist(), "confusion_matrix":cm.tolist()}


class ShipDataset(Dataset):
    def __init__(self, rows, train=False, augmented=False, augmentation=None, seed=17, condition=None, restore=False, recipe_path=None):
        self.rows=rows
        self.train=train
        self.augmentation=resolve_augmentation(augmented,augmentation)
        self.augmented=self.augmentation!="clean"
        self.combined_recipe=combined_recipe(recipe_path) if self.augmentation=="combined" else None
        self.clean_probability=self.combined_recipe["clean_probability"] if self.combined_recipe else .5
        self.seed=seed
        self.epoch=0
        self.condition=condition or {"kind":"clean","severity":0,"domain":"matched","placement":"random"}
        self.restore=restore

    def __len__(self):
        return len(self.rows)

    def augmentation_condition(self, rng):
        if self.augmentation=="single":
            return {"kind":str(rng.choice(["haze","cloud","glare"])), "severity":int(rng.integers(1,6))}
        recipe=self.combined_recipe
        if rng.random()<recipe["degraded_mixture"]["single"]:
            return {"kind":str(rng.choice(recipe["single_kinds"])),
                    "severity":int(rng.choice(recipe["single_severities"]))}
        return {"kind":str(rng.choice(recipe["combined_kinds"])),
                "severity":int(rng.choice(recipe["combined_severities"]))}

    def image(self, index):
        row=self.rows[index]
        with Image.open(ROOT/row["path"]) as f:
            image=f.convert("RGB")
        # Preserve all content; corruption on image content before neutral letterbox padding.
        image=ImageOps.contain(image,(224,224),Image.Resampling.BICUBIC)
        mask=None
        if row.get('target_mask'):
            with Image.open(ROOT/row['target_mask']) as f:
                mask=f.convert('L').resize(image.size,Image.Resampling.NEAREST)
        if self.train:
            rng=np.random.default_rng(stable_seed(self.seed,self.epoch,row["sample_id"],"geometry"))
            if rng.random()<.5:
                image=ImageOps.mirror(image)
                if mask is not None:mask=ImageOps.mirror(mask)
            if rng.random()<.5:
                image=ImageOps.flip(image)
                if mask is not None:mask=ImageOps.flip(mask)
            rotation=90*int(rng.integers(4))
            image=image.rotate(rotation,expand=True)
            if mask is not None:mask=mask.rotate(rotation,expand=True)
        c=dict(self.condition)
        role="test" if row["split"]=="test" else "val"
        base=0
        if self.train:
            role="train"
            base=self.seed
            rng=np.random.default_rng(stable_seed(self.seed,self.epoch,row["sample_id"],"augmentation"))
            if self.augmented and rng.random()>=self.clean_probability:
                c.update(self.augmentation_condition(rng))
            else:
                c.update(kind="clean",severity=0)
            role+=f"_epoch_{self.epoch}"
        s=corruption_seed(row["sample_id"],role,c["kind"],c["severity"],c["domain"],base)
        image,meta=corrupt(image,seed=s,target_mask=None if mask is None else np.asarray(mask)>0,
                           **{k:c[k] for k in ("kind","severity","domain","placement")})
        if self.restore:
            image=dcp(image)
        image=ImageOps.pad(image,(224,224),method=Image.Resampling.BICUBIC,color=(124,116,104))
        return image,meta

    def __getitem__(self,index):
        image,meta=self.image(index)
        tensor=TF.to_tensor(image)
        tensor=TF.normalize(tensor,[.485,.456,.406],[.229,.224,.225])
        return tensor,self.rows[index]["label"],index,json.dumps(meta,separators=(",",":"))


def loader(dataset,batch_size,workers,shuffle=False,seed=17):
    generator=torch.Generator().manual_seed(seed)
    return DataLoader(dataset,batch_size=batch_size,num_workers=workers,shuffle=shuffle,
                      pin_memory=True,generator=generator,persistent_workers=False)


@torch.inference_mode()
def predict(model,dataset,n,batch_size=32,workers=4,budget=None):
    model.eval()
    records=[]
    torch.cuda.synchronize()
    started=time.perf_counter()
    for x,y,indices,metadata in loader(dataset,batch_size,workers):
        with torch.autocast("cuda",dtype=torch.float16):
            logits=model(x.cuda(non_blocking=True))
        probs=logits.float().softmax(1).cpu().numpy()
        for index,label,p,meta in zip(indices.tolist(),y.tolist(),probs,metadata):
            row=dataset.rows[index]
            records.append({"sample_id":row["sample_id"],"source_group_id":row["source_group_id"],
                            "source_scene_id":row.get("source_scene_id"),"label":label,
                            "prediction":int(p.argmax()),"probabilities":p.tolist(),
                            "degradation":json.loads(meta),"original_width":row["width"],"original_height":row["height"]})
        if budget:
            budget.check()
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    result=metrics([r["label"] for r in records],[r["prediction"] for r in records],n)
    result["wall_seconds"]=elapsed
    return result,records


def train(manifest,model_name,augmented=False,augmentation=None,seed=17,epochs=None,batch_size=None,workers=None,limit=None,phase=None):
    config=json.loads((ROOT/"configs/protocol.json").read_text())
    if json.loads((ROOT/"configs/degradations.json").read_text()) != RECIPE:
        raise ValueError("Recipe configuration and implementation disagree")
    mode=resolve_augmentation(augmented,augmentation)
    combo=combined_recipe() if mode=="combined" else None
    epochs=epochs or config["epochs"]
    batch_size=batch_size or config["batch_size"]
    workers=config["workers"] if workers is None else workers
    rows=read_jsonl(manifest)
    if not rows:
        raise ValueError("Empty manifest")
    # Research runs must prove group-disjoint before touching a model.
    for a,b in (("train","val"),("train","test"),("val","test")):
        overlap={r["source_group_id"] for r in rows if r["split"]==a}&{r["source_group_id"] for r in rows if r["split"]==b}
        if overlap and rows[0]["protocol"]=="research":
            raise ValueError(f"Leaking source groups: {a}/{b}")
    n=max(r["label"] for r in rows)+1
    labels={r["label"]:r["label_name"] for r in rows}
    train_rows=[r for r in rows if r["split"]=="train"]
    val_rows=[r for r in rows if r["split"]=="val"]
    if limit:
        train_rows=train_rows[:limit]
        val_rows=val_rows[:max(2,limit//4)]
    if not train_rows or not val_rows:
        raise ValueError("Training and validation both required")
    suffix={"clean":"clean","single":"aug","combined":"combo"}[mode]
    tag=f"{Path(manifest).stem}/{model_name}_{suffix}_s{seed}"
    if limit:
        tag="smoke/"+tag
    out=ROOT/"runs"/tag
    out.mkdir(parents=True,exist_ok=True)
    if (out/"complete.json").exists():
        completed=json.loads((out/"complete.json").read_text())
        completed_mode=completed.get("augmentation") or ("single" if completed.get("augmented") else "clean")
        if completed["manifest_sha256"]!=digest(manifest) or completed["epochs"]!=epochs or completed["config_sha256"]!=digest(ROOT/"configs/protocol.json") or completed_mode!=mode:
            raise ValueError("Completed run has different settings; choose a separate run directory")
        if mode=="combined" and completed.get("combined_recipe_sha256")!=digest(COMBINED_RECIPE_PATH):
            raise ValueError("Combined augmentation recipe changed after training")
        return completed
    seed_all(seed)
    torch.set_num_threads(4)
    torch.hub.set_dir(str(ROOT/"artifacts/weights"))
    settings={"manifest":str(Path(manifest).resolve()),"manifest_sha256":digest(manifest),
              "model":model_name,"augmented":mode!="clean","augmentation":mode,"seed":seed,"epochs":epochs,
              "batch_size":batch_size,"workers":workers,"classes":labels,"num_classes":n,
              "train_n":len(train_rows),"val_n":len(val_rows),"scope":rows[0]["scope"],
              "protocol":rows[0]["protocol"],"pretrained_url":MODEL_WEIGHTS[model_name].url,
              "config_sha256":digest(ROOT/"configs/protocol.json"),"recipe_sha256":digest(ROOT/"configs/degradations.json"),
              "environment":environment(),"smoke_only":bool(limit)}
    if combo:
        settings.update(combined_recipe=combo,combined_recipe_sha256=digest(COMBINED_RECIPE_PATH))
    settings["source_sha256"]={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/"src").rglob("*.py"))}
    write_json(out/"config.json",settings)
    phase=phase or ("augmentation" if mode!="clean" else "baseline")
    with Budget(phase) as budget:
        model=build_model(model_name,n).cuda()
        optimizer_config=config["optimizer"][model_name]
        kwargs={k:v for k,v in optimizer_config.items() if k!="name"}
        optimizer=(torch.optim.SGD if optimizer_config["name"]=="sgd" else torch.optim.AdamW)(model.parameters(),**kwargs)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
        scaler=torch.amp.GradScaler("cuda")
        ds=ShipDataset(train_rows,train=True,augmentation=mode,seed=seed)
        history=[]
        best=-1
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(epochs):
            start=time.perf_counter()
            ds.epoch=epoch
            model.train()
            loss_sum=0
            correct=0
            for x,y,_,_ in loader(ds,batch_size,workers,shuffle=True,seed=seed+epoch):
                x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.float16):
                    output=model(x)
                    loss=F.cross_entropy(output,y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                scaler.step(optimizer)
                scaler.update()
                loss_sum+=loss.item()*len(y)
                correct+=(output.argmax(1)==y).sum().item()
                budget.check()
            result,_=predict(model,ShipDataset(val_rows),n,batch_size,workers,budget)
            record={"epoch":epoch+1,"train_loss":loss_sum/len(ds),"train_accuracy":correct/len(ds),
                    "val_macro_f1":result["macro_f1"],"val_accuracy":result["accuracy"],
                    "lr":optimizer.param_groups[0]["lr"],"seconds":time.perf_counter()-start}
            history.append(record)
            write_json(out/"history.json",history)
            if result["macro_f1"]>best:
                best=result["macro_f1"]
                checkpoint={"model":model.state_dict(),"settings":settings,"epoch":epoch+1,"validation":result}
                tmp=out/"best.tmp.pt"
                torch.save(checkpoint,tmp)
                tmp.replace(out/"best.pt")
            scheduler.step()
            print(json.dumps({"run":tag,**record}),flush=True)
        elapsed=time.monotonic()-budget.started
        settings.update(best_val_macro_f1=best,training_wall_seconds=elapsed,
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                        parameter_count=sum(p.numel() for p in model.parameters()),
                        checkpoint_sha256=digest(out/"best.pt"),
                        pretrained_sha256=digest(ROOT/"artifacts/weights/checkpoints"/Path(MODEL_WEIGHTS[model_name].url).name))
        write_json(out/"complete.json",settings)
    return settings


def load_run(run):
    run=Path(run)
    checkpoint=torch.load(run/"best.pt",map_location="cpu",weights_only=False)
    settings=checkpoint["settings"]
    manifest=settings["manifest"]
    if digest(manifest)!=settings["manifest_sha256"]:
        raise ValueError("Manifest changed after training")
    if digest(ROOT/"configs/degradations.json")!=settings["recipe_sha256"]:
        raise ValueError("Degradation recipe changed after training")
    if settings.get("augmentation")=="combined" and digest(COMBINED_RECIPE_PATH)!=settings.get("combined_recipe_sha256"):
        raise ValueError("Combined augmentation recipe changed after training")
    model=build_model(settings["model"],settings["num_classes"],pretrained=False)
    model.load_state_dict(checkpoint["model"],strict=True)
    model=model.cuda().eval()
    return model,settings,read_jsonl(manifest)


def evaluate(run,split="test",restore=False,validation_only=False):
    torch.set_num_threads(4)
    run=Path(run)
    output=run/("validation_robust" if split=="val" else "evaluation_dcp" if restore else "evaluation")
    output.mkdir(parents=True,exist_ok=True)
    with Budget("evaluation") as budget:
        model,settings,rows=load_run(run)
        rows=[r for r in rows if r["split"]==split]
        if not rows:
            raise ValueError("Empty evaluation split")
        conds=conditions(include_shifted=not validation_only)
        if validation_only:
            conds=[c for c in conds if c["kind"]=="clean" or c["kind"] in ("haze","cloud","glare") and c["placement"]=="random"]
        summaries={}
        for c in conds:
            path=output/(c["name"]+".json")
            if path.exists():
                summaries[c["name"]]=json.loads(path.read_text())
                continue
            ds=ShipDataset(rows,condition=c,restore=restore)
            score,preds=predict(model,ds,settings["num_classes"],workers=4,budget=budget)
            score.update(condition=c,checkpoint_sha256=digest(run/"best.pt"),manifest_sha256=settings["manifest_sha256"],
                         recipe_sha256=settings["recipe_sha256"],restoration="DCP" if restore else None)
            write_jsonl(output/(c["name"]+".jsonl"),preds)
            write_json(path,score)
            summaries[c["name"]]=score
            print(json.dumps({"run":str(run.relative_to(ROOT)),"condition":c["name"],"macro_f1":score["macro_f1"],"accuracy":score["accuracy"]}),flush=True)
        write_json(output/"summary.json",summaries)
    return summaries


def profile(run,restore=False,repeats=100):
    torch.set_num_threads(4)
    run=Path(run)
    with Budget("evaluation") as budget:
        model,settings,rows=load_run(run)
        ds=ShipDataset([r for r in rows if r["split"]=="val"],restore=restore)
        x=ds[0][0].unsqueeze(0).cuda()
        timings=[]
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
            for _ in range(20):
                model(x)
            torch.cuda.synchronize()
            for i in range(repeats):
                start=time.perf_counter()
                # Disk decode, resize, corruption/restoration, normalize and transfer included.
                image=ds[i%len(ds)][0].unsqueeze(0).cuda()
                model(image)
                torch.cuda.synchronize()
                timings.append((time.perf_counter()-start)*1000)
            a=model(x).float().softmax(1).cpu()
        # Meaningful checkpoint reload equivalence check.
        del model
        torch.cuda.empty_cache()
        model,_,_=load_run(run)
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
            b=model(x).float().softmax(1).cpu()
        error=float((a-b).abs().max())
        if error>1e-5:
            raise AssertionError(f"Checkpoint prediction changed: {error}")
        # PyTorch profiler FLOPs count is explicitly partial (unsupported ops omitted).
        with torch.inference_mode(), torch.profiler.profile(with_flops=True) as prof:
            model(x)
        flops=sum(e.flops for e in prof.key_averages())
        result={"parameters":sum(p.numel() for p in model.parameters()),
                "profiled_flops_partial":flops,"flops_note":"PyTorch supported operators only; not total architecture FLOPs",
                "batch_size":1,"precision":"float16 autocast","warmup":20,"repeats":repeats,
                "end_to_end_ms_median":float(np.median(timings)),"end_to_end_ms_p95":float(np.percentile(timings,95)),
                "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),"reload_max_probability_difference":error,
                "restoration":"DCP" if restore else None,"environment":environment()}
        write_json(run/("cost_dcp.json" if restore else "cost.json"),result)
        budget.check(force=True)
    return result
