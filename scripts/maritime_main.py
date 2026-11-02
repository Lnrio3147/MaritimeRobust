"""Prospective five-arm maritime comparison with independent data roles.

    .venv/bin/python scripts/maritime_main.py prepare
    .venv/bin/python scripts/maritime_main.py all
    .venv/bin/python scripts/maritime_main.py all --smoke

All training precedes evaluation; failed jobs stop the queue. Legacy runs are
never loaded. Budget enforces the existing exclusive GPU lock and resource caps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import Dataset, TensorDataset
from torchvision.transforms import AugMix, functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maritime_robust.common import (Budget, digest, environment, read_jsonl,
                                    seed_all, stable_seed, write_json, write_jsonl)
from maritime_robust.engine import build_model, loader, metrics
from maritime_robust.main_protocol import (apply_components, assert_isolated,
    clean_image, error_decomposition, evaluation_conditions, partition_rows)
from maritime_robust.cifar_fields import sample_degradation
from maritime_robust.cifar_field_estimator import (FieldEstimator, apply_blur_torch,
    apply_field_torch, gaussian_kernels, supervised_loss)

MEAN, STD = [.485, .456, .406], [.229, .224, .225]
CONFIG = ROOT / "configs/maritime_main_v2.json"


def effective_config(smoke=False, path=None):
    config = json.loads((path or CONFIG).read_text())
    if smoke:
        config["version"] += "_smoke"
        config["output"] += "_smoke"
        config.update(epochs=1, batch_size=4, workers=0, seeds=[17])
        config["estimator"].update(steps=2, batch_size=2)
        config["calibration"]["draws"] = 4
    return config


def root_for(config):
    return ROOT / config["output"]


def protocol_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def prepare(config):
    out = root_for(config)
    source = ROOT / config["manifest"]
    partition = partition_rows(read_jsonl(source), config)
    if config["version"].endswith("_smoke"):
        sizes = {"fit": 16, "selection": 8, "calibration": 4, "target": 4, "evaluation": 8}
        partition = {k: v[:sizes[k]] for k, v in partition.items()}
    source_files = [Path(__file__).resolve(), ROOT / "configs/degradations.json",
                    *sorted((ROOT / "src/maritime_robust").rglob("*.py"))]
    lock = {"protocol_sha256": protocol_hash(config), "manifest_sha256": digest(source),
            "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_files},
            "counts": {k: len(v) for k, v in partition.items()}}
    if (out / "LOCK.json").exists():
        if json.loads((out / "LOCK.json").read_text()) != lock:
            raise ValueError("Prepared protocol/code/data changed: use a new version/output directory")
        if json.loads((out / "roles.json").read_text()) != partition:
            raise ValueError("Stored role manifest changed")
        return partition
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "protocol.json", config)
    write_json(out / "roles.json", partition)
    write_json(out / "LOCK.json", lock)
    write_json(ROOT / "reports" / f"{config['version']}_protocol.json", {
        **lock, "configuration": config,
        "roles": {k: [{f: r.get(f) for f in ("sample_id", "source_group_id", "sha256", "pixel_sha256")}
                       for r in v] for k, v in partition.items()},
        "source_label_stratification": "source labels used to allocate roles only; target input files omit labels",
        "excluded_test_images": 236, "evaluation_is_blind": False})
    return partition


def tensors(rows, size):
    return torch.stack([TF.to_tensor(clean_image(ROOT / r["path"], size)) for r in rows])


def prior_batch(rng, batch, size, device):
    values = [sample_degradation(rng, 32, 32) for _ in range(batch)]
    ks, ns, ss, bs = zip(*values)
    k = torch.from_numpy(np.stack(ks))[:, None].to(device)
    n = torch.from_numpy(np.stack(ns)).permute(0, 3, 1, 2).to(device)
    k = F.interpolate(k, size=(size, size), mode="bilinear", align_corners=False)
    n = F.interpolate(n, size=(size, size), mode="bilinear", align_corners=False)
    return k, n, torch.tensor(ss, device=device), torch.from_numpy(np.stack(bs)).to(device)


def prior_view(clean, rng):
    k, n, sigma, blur = prior_batch(rng, len(clean), clean.shape[-1], clean.device)
    value = apply_field_torch(apply_blur_torch(clean, blur), k, n)
    value = (value + torch.randn_like(value) * sigma[:, None, None, None]).clamp(0, 1)
    return value, (k, n, sigma, blur)


class FieldBank:
    def __init__(self, path, device="cuda"):
        with np.load(path) as data:
            self.k = torch.as_tensor(data["K"].astype(np.float32), device=device)
            self.n = torch.as_tensor(data["N"].astype(np.float32), device=device)
            self.sigma = torch.as_tensor(data["sigma"].astype(np.float32), device=device)
            self.kernels = gaussian_kernels(torch.as_tensor(data["blur"], device=device))

    @torch.no_grad()
    def apply(self, clean, generator, repeats=1, fields=2):
        value = clean.clone()
        b, c, h, w = value.shape
        for _ in range(fields):
            picks = torch.randint(len(self.k), (b,), device=value.device, generator=generator)
            k = F.interpolate(self.k[picks], size=(h, w), mode="bilinear", align_corners=False)
            n = F.interpolate(self.n[picks], size=(h, w), mode="bilinear", align_corners=False)
            kernels = self.kernels[picks].repeat_interleave(c, 0)
            for _ in range(repeats):
                padded = F.pad(value.reshape(1, b * c, h, w), (5, 5, 5, 5), mode="replicate")
                value = F.conv2d(padded, kernels, groups=b * c).reshape(b, c, h, w).clamp(0, 1)
                value = (k * value + n).clamp(0, 1)
            noise = torch.randn(value.shape, device=value.device, generator=generator)
            value = (value + noise * self.sigma[picks, None, None, None]).clamp(0, 1)
        return value


def build_fields(config, roles):
    out = root_for(config) / "fields"
    if (out / "complete.json").exists():
        saved = json.loads((out / "complete.json").read_text())
        for name, key in (("bank.npz", "bank_sha256"), ("estimator.pt", "estimator_sha256"),
                          ("unlabeled_target_images.jsonl", "target_manifest_sha256")):
            if digest(out / name) != saved[key] or saved["protocol_sha256"] != protocol_hash(config):
                raise ValueError("Prepared NCF artifacts changed")
        return
    out.mkdir(parents=True, exist_ok=True)
    settings = config["estimator"]
    size = config["image_size"]
    with Budget("augmentation") as budget:
        seed_all(settings["seed"])
        torch.set_num_threads(4)
        source = tensors(roles["fit"], size).cuda()
        estimator = FieldEstimator(width=settings["width"], stride=settings["stride"]).cuda()
        optimizer = torch.optim.AdamW(estimator.parameters(), lr=settings["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings["steps"])
        rng = np.random.default_rng(settings["seed"])
        started = time.perf_counter()
        for step in range(settings["steps"]):
            indices = rng.integers(0, len(source), settings["batch_size"])
            clean = source[indices]
            degraded, (k, n, sigma, blur) = prior_view(clean, rng)
            loss, detail = supervised_loss(estimator, degraded, clean, k, n, sigma, b_true=blur)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite estimator loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(estimator.parameters(), 5)
            optimizer.step()
            scheduler.step()
            budget.check()
            if step % 100 == 0 or step + 1 == settings["steps"]:
                print(json.dumps({"stage": "estimator", "step": step + 1,
                                  "loss": loss.item(), **detail}), flush=True)
        torch.save(estimator.state_dict(), out / "estimator.pt")
        estimator_seconds = time.perf_counter() - started
        del source
        estimator.eval()
        ks, ns, ss, bs, target_records = [], [], [], [], []
        image_dir = out / "target_images"
        image_dir.mkdir(exist_ok=True)
        # Only these corrupted images and anonymous IDs enter the estimator.
        with torch.no_grad():
            for row in roles["target"]:
                base = clean_image(ROOT / row["path"], size)
                for family in config["weather_families"]:
                    for severity in config["severities"]:
                        image = apply_components(base.copy(), [family], severity,
                            row["sample_id"], "target", config)
                        token = hashlib.sha256(f"{row['sample_id']}:{family}:{severity}".encode()).hexdigest()
                        path = image_dir / f"{token}.png"
                        image.save(path)
                        x = TF.to_tensor(image)[None].cuda()
                        k, n, sigma = estimator.coarse(x)
                        ks.append(k[0].cpu().numpy())
                        ns.append(n[0].cpu().numpy())
                        ss.append(float(sigma[0]))
                        bs.append(estimator.blur_params(x)[0].cpu().numpy())
                        target_records.append({"image_id": token, "path": str(path.relative_to(ROOT)),
                                               "source_id": row["sample_id"],
                                               "source_group_id": row["source_group_id"],
                                               "sha256": digest(path)})
                        budget.check()
        write_jsonl(out / "unlabeled_target_images.jsonl", target_records)
        np.savez_compressed(out / "bank.npz", K=np.stack(ks), N=np.stack(ns),
                            sigma=np.asarray(ss, np.float32), blur=np.stack(bs))
        bank = FieldBank(out / "bank.npz")
        calibration = tensors(roles["calibration"], size).cuda()
        cal = config["calibration"]
        seed_all(cal["seed"])
        rng = np.random.default_rng(cal["seed"])
        selected = calibration[rng.integers(0, len(calibration), cal["draws"])]
        reference = selected
        # No named family and no clean target counterpart is used for calibration.
        for _ in range(cal["fields_composed"]):
            reference, _ = prior_view(reference, rng)
        psnr = lambda a, b: float((-10 * ((a - b).square().mean((1, 2, 3))).clamp_min(1e-10).log10()).mean())
        target_strength = psnr(reference, selected)
        candidates = []
        for repeat in cal["candidate_repeats"]:
            generator = torch.Generator(device="cuda").manual_seed(cal["seed"])
            estimates = bank.apply(selected, generator, repeat, cal["fields_composed"])
            candidates.append({"repeats": repeat, "source_synthetic_psnr": psnr(estimates, selected)})
            budget.check()
        repeat = min(candidates, key=lambda r: (abs(r["source_synthetic_psnr"] - target_strength),
                                               r["repeats"]))["repeats"]
        write_json(out / "complete.json", {
            "protocol_sha256": protocol_hash(config), "estimator_seconds": estimator_seconds,
            "total_preparation_seconds": time.monotonic() - budget.started,
            "target_original_images": len(roles["target"]), "target_fields": len(ks),
            "repeats": repeat, "source_prior_reference_psnr": target_strength,
            "calibration_candidates": candidates, "bank_sha256": digest(out / "bank.npz"),
            "estimator_sha256": digest(out / "estimator.pt"),
            "target_manifest_sha256": digest(out / "unlabeled_target_images.jsonl")})
        print(json.dumps({"stage": "fields_complete", "repeats": repeat,
                          "target_original_images": len(roles["target"])}), flush=True)


class TrainImages(Dataset):
    def __init__(self, rows, config, arm, seed):
        self.rows, self.config, self.arm, self.seed = rows, config, arm, seed
        self.images = [clean_image(ROOT / r["path"], config["image_size"]) for r in rows]
        self.epoch = 0
        parameters = {k: v for k, v in config["augmix"].items()
                      if k not in ("jsd_weight", "cross_entropy")}
        self.augmix = AugMix(**parameters)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = self.images[index].copy()
        rng = np.random.default_rng(stable_seed(self.seed, self.epoch, row["sample_id"], "geometry"))
        if rng.random() < .5:
            image = ImageOps.mirror(image)
        if rng.random() < .5:
            image = ImageOps.flip(image)
        image = image.rotate(90 * int(rng.integers(4)))
        rng = np.random.default_rng(stable_seed(self.seed, self.epoch, row["sample_id"], "augment"))
        active = rng.random() < self.config["degraded_probability"]
        if active and self.arm in ("strong_combo", "pair_coverage"):
            recipe = self.config[self.arm]
            if rng.random() < recipe["single_fraction_among_degraded"]:
                parts = [str(rng.choice(self.config["weather_families"]))]
                severity = int(rng.choice(self.config["severities"]))
            else:
                parts = self.config["training_pairs"][int(rng.integers(len(self.config["training_pairs"])))]
                severity = int(rng.choice(recipe["pair_severities"]))
            image = apply_components(image, parts, severity, row["sample_id"],
                                     f"fit_epoch_{self.epoch}", self.config, self.seed)
        tensor = TF.to_tensor(image)
        result = {"x": tensor if self.arm == "ncf" else TF.normalize(tensor, MEAN, STD),
                  "label": row["label"], "active": active}
        if self.arm == "augmix":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(stable_seed(self.seed, self.epoch, row["sample_id"], "augmix"))
                views = [self.augmix(TF.pil_to_tensor(image)) if active else TF.pil_to_tensor(image)
                         for _ in range(2)]
            result.update({f"mix{i}": TF.normalize(v.float() / 255, MEAN, STD)
                           for i, v in enumerate(views)})
        return result


@torch.no_grad()
def score(model, x, y, config):
    predictions = []
    for batch, _ in loader(TensorDataset(x, y), config["batch_size"], 0):
        with torch.autocast("cuda", dtype=torch.float16):
            predictions.append(model(batch.cuda()).float().argmax(1).cpu())
    pred = torch.cat(predictions).numpy()
    return metrics(y.numpy(), pred, 8), pred


def train(config, roles, arm, seed):
    out = root_for(config) / f"{arm}_s{seed}"
    if (out / "complete.json").exists():
        saved = json.loads((out / "complete.json").read_text())
        if (saved["protocol_sha256"] != protocol_hash(config)
                or saved["checkpoint_sha256"] != digest(out / "best.pt")):
            raise ValueError("Completed classifier artifacts changed")
        return
    out.mkdir(exist_ok=True)
    settings = {"protocol_sha256": protocol_hash(config), "arm": arm, "seed": seed,
                "model": config["model"], "epochs": config["epochs"], "environment": environment(),
                "selection": config["selection"], "role_manifest_sha256": digest(root_for(config) / "roles.json")}
    write_json(out / "config.json", settings)
    with Budget("augmentation") as budget:
        seed_all(seed)
        torch.set_num_threads(4)
        torch.hub.set_dir(str(ROOT / "artifacts/weights"))
        model = build_model(config["model"], 8).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"])
        scaler = torch.amp.GradScaler("cuda")
        ds = TrainImages(roles["fit"], config, arm, seed)
        # Only the independent selection set is materialized during training.
        selection_x = TF.normalize(tensors(roles["selection"], config["image_size"]), MEAN, STD)
        selection_y = torch.tensor([r["label"] for r in roles["selection"]])
        bank = FieldBank(root_for(config) / "fields/bank.npz") if arm == "ncf" else None
        field_info = json.loads((root_for(config) / "fields/complete.json").read_text()) if bank else None
        if field_info:
            settings.update(field_bank_sha256=field_info["bank_sha256"], field_repeats=field_info["repeats"])
            write_json(out / "config.json", settings)
        best, history = -1., []
        for epoch in range(config["epochs"]):
            ds.epoch = epoch
            model.train()
            loss_sum = 0.
            for bi, batch in enumerate(loader(ds, config["batch_size"], config["workers"], True, seed + epoch)):
                x, y = batch["x"].cuda(), batch["label"].cuda()
                if bank:
                    active = batch["active"].cuda()
                    if active.any():
                        generator = torch.Generator(device="cuda").manual_seed(stable_seed(seed, epoch, bi, "field"))
                        x[active] = bank.apply(x[active], generator, field_info["repeats"])
                    x = TF.normalize(x, MEAN, STD)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                    if arm == "augmix":
                        p = [logits.float().softmax(1)]
                        p += [model(batch[f"mix{i}"].cuda()).float().softmax(1) for i in range(2)]
                        mean = sum(p) / 3
                        jsd = sum((v * (v.clamp_min(1e-8).log() - mean.clamp_min(1e-8).log())).sum(1).mean()
                                  for v in p) / 3
                        loss = loss + config["augmix"]["jsd_weight"] * jsd
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite classifier loss: {arm} {seed}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item() * len(y)
                budget.check()
            model.eval()
            validation, _ = score(model, selection_x, selection_y, config)
            record = {"epoch": epoch + 1, "loss": loss_sum / len(ds),
                      "selection_macro_f1": validation["macro_f1"],
                      "selection_accuracy": validation["accuracy"]}
            history.append(record)
            write_json(out / "history.json", history)
            if validation["macro_f1"] > best:
                best = validation["macro_f1"]
                torch.save({"model": model.state_dict(), "settings": settings,
                            "epoch": epoch + 1, "selection": validation}, out / "best.tmp.pt")
                (out / "best.tmp.pt").replace(out / "best.pt")
            scheduler.step()
            print(json.dumps({"stage": "train", "arm": arm, "seed": seed, **record}), flush=True)
        write_json(out / "complete.json", {**settings, "best_selection_macro_f1": best,
            "training_wall_seconds": time.monotonic() - budget.started,
            "checkpoint_sha256": digest(out / "best.pt")})


def evaluate(config, roles, arm, seed):
    run = root_for(config) / f"{arm}_s{seed}"
    out = ROOT / "reports" / config["version"] / run.name
    if (out / "complete.json").exists():
        return
    if not (run / "complete.json").exists():
        raise ValueError("Cannot evaluate an incomplete training run")
    with Budget("evaluation") as budget:
        torch.set_num_threads(4)
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        model = build_model(config["model"], 8, pretrained=False)
        model.load_state_dict(checkpoint["model"])
        model = model.cuda().eval()
        rows = roles["evaluation"]
        clean = [clean_image(ROOT / r["path"], config["image_size"]) for r in rows]
        y = torch.tensor([r["label"] for r in rows])
        predictions, results = {}, {}
        for condition in evaluation_conditions(config):
            images = [apply_components(image.copy(), condition["components"], condition["severity"],
                                       row["sample_id"], "evaluation", config)
                      for image, row in zip(clean, rows)]
            x = TF.normalize(torch.stack([TF.to_tensor(image) for image in images]), MEAN, STD)
            result, pred = score(model, x, y, config)
            predictions[condition["key"]] = pred
            results[condition["key"]] = result
            budget.check()
        interactions = {}
        for condition in evaluation_conditions(config):
            if len(condition["components"]) != 2:
                continue
            a, b = condition["components"]
            severity = condition["severity"]
            flags = [predictions[key] == y.numpy() for key in
                     (condition["key"], f"{a}_{severity}", f"{b}_{severity}")]
            decomposition = error_decomposition(*flags)
            e0 = 1 - results["clean"]["accuracy"]
            decomposition["additive_excess"] = (
                (1 - results[condition["key"]]["accuracy"])
                - (1 - results[f"{a}_{severity}"]["accuracy"])
                - (1 - results[f"{b}_{severity}"]["accuracy"]) + e0)
            interactions[condition["key"]] = decomposition
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "predictions.npz", labels=y.numpy(),
                            sample_ids=np.array([r["sample_id"] for r in rows]), **predictions)
        write_json(out / "complete.json", {"run": run.name, "protocol_sha256": protocol_hash(config),
            "checkpoint_sha256": digest(run / "best.pt"), "n": len(rows),
            "conditions": results, "interactions": interactions,
            "evaluation_wall_seconds": time.monotonic() - budget.started,
            "predictions_sha256": digest(out / "predictions.npz")})


def report(config):
    output = ROOT / "reports" / f"{config['version'].upper()}.md"
    lines = ["# 海事主表：独立角色协议 v2", "",
        "本页由 maritime_main.py 从新运行生成，旧结果不并入。"
        "评测使用以前已检查过的 157 图 development val，不能称为盲测。", "",
        "训练／选型／源校准／无标签目标池／评测在已识别的 source_group、原图 SHA 与像素 SHA 上隔离。"
        "完整流程不使用目标图的干净参考标定；PSNR 参考来自独立源图上的连续合成先验。", "",
        "| 方法 | seed | Clean % | Macro-F1 % | 复合 % | 最差复合 % | 并集残差 pp | 未见配对 % | 池外族复合 % | 训练 h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    expected = {c["key"] for c in evaluation_conditions(config)}
    pending = []
    for arm in config["arms"]:
        for seed in config["seeds"]:
            path = ROOT / "reports" / config["version"] / f"{arm}_s{seed}/complete.json"
            training = root_for(config) / f"{arm}_s{seed}/complete.json"
            if not path.exists():
                state = "训练已完成，待评测" if training.exists() else "训练未完成"
                pending.append(f"- `{arm}_s{seed}`：{state}。")
                continue
            data = json.loads(path.read_text())
            if (data["protocol_sha256"] != protocol_hash(config)
                    or set(data["conditions"]) != expected):
                raise ValueError(f"Protocol/incomplete grid mismatch: {path}")
            c = data["conditions"]
            pairs = [v["key"] for v in evaluation_conditions(config) if v["group"] == "weather_pair"]
            held = [f"{'+'.join(config['held_pair'])}_{s}" for s in config["severities"]]
            novel = [v["key"] for v in evaluation_conditions(config) if v["group"] == "held_family_pair"]
            avg = lambda keys: np.mean([c[k]["accuracy"] for k in keys]) * 100
            union = np.mean([data["interactions"][k]["union_excess"] for k in pairs]) * 100
            hours = json.loads(training.read_text())["training_wall_seconds"] / 3600
            lines.append(f"| {arm} | {seed} | {100*c['clean']['accuracy']:.2f} | "
                         f"{100*c['clean']['macro_f1']:.2f} | {avg(pairs):.2f} | "
                         f"{100*min(c[k]['accuracy'] for k in pairs):.2f} | {union:+.2f} | "
                         f"{avg(held):.2f} | {avg(novel):.2f} | {hours:.3f} |")
    lines += ["", "信息条件：standard/augmix 不读取目标池；strong_combo 是固定天气配方；"
              "pair_coverage 明确使用已知天气配方；ncf 从无标签目标图估计场。"
              "strong_combo 混合单项／配对，pair_coverage 的退化分支只采两对，不能视为单变量消融。", "",
              "主复合指标含三对天气×五级，共 15 条件；cloud+glare 留出于显式训练配方。"
              "defocus 不进入目标池或显式配方，但连续先验含通用模糊参数，池外族不代表参数机制完全未见。", "",
              "训练成本仅为分类训练；NCF 公共估计器／场库／标定成本在 fields/complete.json 单独报告。"
              "同 epoch / 数据 / 优化器不代表相同 FLOPs；AugMix 使用三视图及 JSD。", "",
              "## 未完成运行", "", *(pending or ["全部计划运行已完成。"]), ""]
    write_json(ROOT / "reports" / f"{config['version']}_status.json", {
        "protocol_sha256": protocol_hash(config), "pending": pending,
        "output": str(root_for(config).relative_to(ROOT))})
    output.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fields", "train", "evaluate", "report", "all"))
    parser.add_argument("--arm", choices=("standard", "augmix", "strong_combo", "pair_coverage", "ncf"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", type=Path, default=None,
                        help="alternate frozen config; defaults to configs/maritime_main_v2.json")
    args = parser.parse_args()
    config = effective_config(args.smoke, args.config)
    roles = prepare(config)
    if args.stage == "prepare":
        report(config)
        print(json.dumps({"prepared": config["version"], "counts": {k: len(v) for k, v in roles.items()}}))
    elif args.stage == "fields":
        build_fields(config, roles)
    elif args.stage in ("train", "evaluate"):
        if not args.arm:
            parser.error("--arm is required")
        (train if args.stage == "train" else evaluate)(config, roles, args.arm, args.seed)
        report(config)
    elif args.stage == "report":
        report(config)
    else:
        queue_path = ROOT / "reports" / f"{config['version']}_queue.json"
        queue_base = {"pid": os.getpid(), "protocol_sha256": protocol_hash(config),
                      "started_at": time.time()}
        write_json(queue_path, {**queue_base, "status": "running", "stage": "fields"})
        try:
            build_fields(config, roles)
            # Freeze every classifier checkpoint before inspecting evaluation results.
            for seed in config["seeds"]:
                for arm in config["arms"]:
                    write_json(queue_path, {**queue_base, "status": "running", "stage": "train", "arm": arm, "seed": seed})
                    train(config, roles, arm, seed)
                    report(config)
            for seed in config["seeds"]:
                for arm in config["arms"]:
                    write_json(queue_path, {**queue_base, "status": "running", "stage": "evaluate", "arm": arm, "seed": seed})
                    evaluate(config, roles, arm, seed)
                    report(config)
        except BaseException as exc:
            write_json(queue_path, {**queue_base, "status": "failed", "error": repr(exc)})
            raise
        write_json(queue_path, {**queue_base, "status": "complete", "ended_at": time.time()})


if __name__ == "__main__":
    main()
