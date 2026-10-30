"""Archive admission, duplicate audit, explicit label maps and grouped splits."""
import collections
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

from .common import ROOT, digest, stable_seed, write_json, write_jsonl

LABELS = ["non-ship", "air carrier", "destroyer", "landing craft", "frigate",
          "amphibious transport dock", "cruiser", "Tarawa-class amphibious assault ship",
          "amphibious assault ship", "command ship", "submarine", "medical ship",
          "combat boat", "auxiliary ship", "container ship", "car carrier", "hovercraft",
          "bulk carrier", "oil tanker", "fishing boat", "passenger ship", "liquefied gas ship", "barge"]
CIVILIAN = {14, 15, 17, 18, 19, 20, 21, 22}


def safe_extract(archive, destination, max_bytes=30*10**9):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            members = z.infolist()
            if sum(m.file_size for m in members) > max_bytes:
                raise ValueError("Archive exceeds extraction limit")
            for m in members:
                if not (destination / m.filename).resolve().is_relative_to(destination):
                    raise ValueError("Unsafe archive path")
                if (m.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Symlink in archive")
            z.extractall(destination)
    else:
        import py7zr
        if not py7zr.is_7zfile(archive):
            raise ValueError("Unsupported archive format")
        with py7zr.SevenZipFile(archive) as z:
            members = z.list()
            if sum(m.uncompressed or 0 for m in members) > max_bytes:
                raise ValueError("Archive exceeds extraction limit")
            for name in z.getnames():
                if not (destination / name).resolve().is_relative_to(destination):
                    raise ValueError("Unsafe archive path")
            z.extractall(destination)


class UnionFind:
    def __init__(self, n):
        self.parents = list(range(n))

    def find(self, i):
        while self.parents[i] != i:
            self.parents[i] = self.parents[self.parents[i]]
            i = self.parents[i]
        return i

    def union(self, i, j):
        a, b = self.find(i), self.find(j)
        self.parents[max(a, b)] = min(a, b)


def audit_fgsc23(root):
    root = Path(root).resolve()
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
    if not paths:
        raise ValueError("No FGSC-23 image files")
    rows, tiny, phashes, failures = [], [], [], []
    for path in paths:
        parts = path.relative_to(root).parts
        split_i = next((i for i,p in enumerate(parts) if p.lower() in {"train", "test"}), None)
        if split_i is None or split_i + 1 >= len(parts):
            failures.append({"path": str(path), "reason": "missing train/test/category hierarchy"})
            continue
        try:
            label = int(parts[split_i+1])
            if not 0 <= label < len(LABELS):
                raise ValueError("unknown label")
            with Image.open(path) as f:
                image = f.convert("RGB")
                pixels = np.asarray(image)
                pxhash = hashlib.sha256(str(image.size).encode()+pixels.tobytes()).hexdigest()
                thumb = np.asarray(image.resize((32,32)), dtype=np.float32)/255
                coeff = dctn(np.asarray(image.convert("L").resize((32,32))), norm="ortho")[:8,:8]
                ph = coeff.ravel() > np.median(coeff.ravel()[1:])
                ph[0] = False
                phashes.append(ph)
                tiny.append(thumb)
                rel = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
                rows.append({"sample_id": "fgsc23_"+digest(path)[:16]+"_"+stable_seed(str(path.relative_to(root))).__str__(),
                             "path": rel, "original_split": parts[split_i].lower(), "original_label": label,
                             "label_name": LABELS[label], "civilian": label in CIVILIAN,
                             "source_dataset": "FGSC-23", "source_url": "https://github.com/xiong577/ship-datasets",
                             "source_scene_id": None, "port_id": None, "vessel_id": None,
                             "sha256": digest(path), "pixel_sha256": pxhash,
                             "width": image.width, "height": image.height,
                             "target_mask": None, "degradation_recipe": None})
        except (ValueError, OSError) as e:
            failures.append({"path": str(path), "reason": str(e)})
    if failures:
        write_json(ROOT/"reports/fgsc23_read_errors.json", failures)
        raise ValueError(f"{len(failures)} admission errors; inspect report")
    uf = UnionFind(len(rows))
    seen = {}
    for i,r in enumerate(rows):
        key=r["pixel_sha256"]
        if key in seen:
            uf.union(i,seen[key])
        else:
            seen[key]=i
    phashes = np.asarray(phashes)
    near = []
    # Vectorized O(N^2) on the small benchmark, bounded one-row memory.
    for i in range(len(rows)):
        distances = np.count_nonzero(phashes[i+1:] != phashes[i], axis=1)
        for j in (np.flatnonzero(distances <= 4) + i + 1):
            j=int(j)
            if rows[i]["pixel_sha256"] == rows[j]["pixel_sha256"]:
                continue
            mae=float(np.abs(tiny[i]-tiny[j]).mean())
            confirmed=mae <= .035
            near.append({"a":rows[i]["sample_id"], "b":rows[j]["sample_id"],
                         "hamming":int(distances[j-i-1]),"thumbnail_mae":mae,"grouped":confirmed})
            if confirmed:
                uf.union(i,j)
    groups = collections.defaultdict(list)
    for i,r in enumerate(rows):
        groups[uf.find(i)].append(i)
    for inds in groups.values():
        gid="duplicate_"+min(rows[i]["sha256"] for i in inds)[:20]
        labels={rows[i]["original_label"] for i in inds}
        for i in inds:
            rows[i]["source_group_id"]=gid
            rows[i]["source_group_basis"]="duplicate_component; original scene unavailable"
            rows[i]["label_conflict"]=len(labels)>1
    crossings=[inds for inds in groups.values() if len({rows[i]["original_split"] for i in inds})>1]
    write_jsonl(ROOT/"data/manifests/fgsc23_inventory.jsonl",rows)
    write_json(ROOT/"reports/fgsc23_near_duplicates.json",near)
    manifests={}
    for protocol in ("official", "research"):
        for scope in ("civilian", "all"):
            selected=[dict(r) for r in rows if scope=="all" or r["civilian"]]
            if protocol=="research":
                # Collapse identical pixels and exclude contradictory-label components.
                unique={}
                for r in selected:
                    if r["label_conflict"]:
                        continue
                    key=r["pixel_sha256"]
                    if key not in unique or r["original_split"]=="test":
                        unique[key]=r
                selected=list(unique.values())
            test_groups={r["source_group_id"] for r in selected if r["original_split"]=="test"}
            # Research protocol preserves official test samples, quarantines linked train groups.
            for r in selected:
                r["split"]="test" if (r["original_split"]=="test" or protocol=="research" and r["source_group_id"] in test_groups) else "train"
            label_map={old:i for i,old in enumerate(sorted({r["original_label"] for r in selected}))}
            val_groups=set()
            for label in label_map:
                train_groups=sorted({r["source_group_id"] for r in selected if r["split"]=="train" and r["original_label"]==label},
                                    key=lambda g:stable_seed(20261027,g))
                count=max(1,round(len(train_groups)*.2)) if len(train_groups)>1 else 0
                val_groups.update(train_groups[:count])
            for r in selected:
                if r["split"]=="train" and r["source_group_id"] in val_groups:
                    r["split"]="val"
                r.update(label=label_map[r["original_label"]],protocol=protocol,scope=scope)
            intersections={}
            for a,b in (("train","val"),("train","test"),("val","test")):
                ag={r["source_group_id"] for r in selected if r["split"]==a}
                bg={r["source_group_id"] for r in selected if r["split"]==b}
                intersections[a+"/"+b]=len(ag&bg)
            if protocol=="research" and any(intersections.values()):
                raise AssertionError("source group leakage")
            name=f"fgsc23_{protocol}_{scope}"
            manifest=ROOT/f"data/manifests/{name}.jsonl"
            write_jsonl(manifest,selected)
            manifests[name]={"sha256":digest(manifest),"counts":dict(collections.Counter(r["split"] for r in selected)),
                             "class_counts":{split:dict(collections.Counter(r["label_name"] for r in selected if r["split"]==split)) for split in ("train","val","test")},
                             "label_map":label_map,"group_intersections":intersections}
    report={"dataset":"FGSC-23","actual_image_count":len(rows),"unique_pixels":len(seen),
            "exact_duplicate_extras":len(rows)-len(seen),"near_candidate_pairs":len(near),
            "cross_official_split_components":len(crossings),
            "cross_official_examples":[[rows[i]["sample_id"] for i in inds] for inds in crossings[:30]],
            "label_conflict_components":sum(len({rows[i]["original_label"] for i in inds})>1 for inds in groups.values()),
            "manifests":manifests,
            "limitations":["Original scene/port/vessel identities absent; duplicate-free is not scene-independent.",
                            "Near-duplicate screening is conservative and cannot certify no augmented/rotated overlap.",
                            "No ground-truth ship mask, therefore no true ship-coverage measurements.",
                            "Official protocol includes train-derived validation; full-train published numbers are not identical training protocols."],
            "civilian_ids":sorted(CIVILIAN),"ambiguous_excluded_ids":[11,13,16]}
    write_json(ROOT/"reports/fgsc23_audit.json",report)
    return report
