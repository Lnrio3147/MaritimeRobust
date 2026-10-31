"""Produce a unified, machine-readable snapshot of the project state.

Weights, crops and raw images are intentionally kept out of git.  This script
records what git cannot: hashes of every experiment config, data split,
checkpoint and evaluation output, so the archived results stay checkable.

    python scripts/snapshot.py                 # fast: existence + size
    python scripts/snapshot.py --verify-weights # re-hash every checkpoint (~15 GB)

Output: reports/snapshot.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CODE_DIRS = ("src", "scripts", "configs")
DATA_GLOBS = ("data/manifests/*.jsonl", "data/manifests/folds/*.jsonl")
REPORT_GLOBS = ("reports/*.md", "reports/*.json", "reports/s2/*.json", "reports/s2b/*.json")
EXTERNAL_DIRS = ("external", "modules")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def inventory(globs) -> dict[str, dict]:
    out = {}
    for pattern in globs:
        for path in sorted(ROOT.glob(pattern)):
            if not path.is_file():
                continue
            out[rel(path)] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return out


def git_state() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
        "tracked_files": len(run("ls-files").splitlines()),
    }


def derive_tag(run_dir: Path) -> dict:
    """Recover method/init/epochs from the run directory name."""
    name = run_dir.name
    parts = name.split("_")
    method, init, epochs = "none", "imagenet1k", None
    if "s" in parts:
        # ... _ {method} _ [e40 _] [i22k _] s{seed}
        head = []
        for p in parts[2:]:
            if p.startswith("s") and p[1:].isdigit():
                break
            head.append(p)
        if head:
            method = head[0]
        if "e40" in parts:
            epochs = 40
        if "i22k" in parts:
            init = "timm22k"
    return {"method": method, "init": init, "epochs_from_name": epochs, "seed_from_name": int(parts[-1][1:]) if parts[-1][1:].isdigit() else None}


RUN_DIRS = ("runs", "runs_external", "runs_cifar")


def run_inventory(verify_weights: bool) -> list[dict]:
    runs = []
    completes = []
    for d in RUN_DIRS:
        completes.extend(sorted((ROOT / d).glob("**/complete.json")))
    for complete in sorted(completes):
        run_dir = complete.parent
        try:
            rec = json.loads(complete.read_text())
        except json.JSONDecodeError:
            runs.append({"run": rel(run_dir), "error": "unreadable complete.json"})
            continue
        ckpt = run_dir / "best.pt"
        if not ckpt.exists():
            ckpt = run_dir / "model.pt"
        entry = {
            "run": rel(run_dir),
            **derive_tag(run_dir),
            "model": rec.get("model"),
            "declared_method": rec.get("method"),
            "seed": rec.get("seed"),
            "epochs": rec.get("epochs"),
            "train_n": rec.get("train_n"),
            "val_n": rec.get("val_n"),
            "best_val_macro_f1": rec.get("best_val_macro_f1"),
            "manifest_sha256": rec.get("manifest_sha256"),
            "config_sha256": rec.get("config_sha256"),
            "recipe_sha256": rec.get("recipe_sha256"),
            "combined_recipe_sha256": rec.get("combined_recipe_sha256"),
            "source_sha256": rec.get("source_sha256"),
            "pretrained_url": rec.get("pretrained_url"),
            "pretrained_sha256": rec.get("pretrained_sha256"),
            "parameter_count": rec.get("parameter_count"),
            "training_wall_seconds": rec.get("training_wall_seconds"),
            "peak_cuda_allocated_bytes": rec.get("peak_cuda_allocated_bytes"),
            "smoke_only": rec.get("smoke_only"),
            "recorded_checkpoint_sha256": rec.get("checkpoint_sha256"),
        }
        artifacts = {}
        for pat in ("*.json", "*.pt", "*.pth"):
            for p in sorted(run_dir.glob(pat)):
                artifacts[p.name] = {"bytes": p.stat().st_size}
        if ckpt.exists():
            artifacts[ckpt.name]["exists"] = True
            if verify_weights:
                digest = sha256(ckpt)
                artifacts[ckpt.name]["sha256"] = digest
                entry["checkpoint_sha256_matches"] = (
                    digest == rec.get("checkpoint_sha256") if rec.get("checkpoint_sha256") else None
                )
        else:
            entry["checkpoint_missing"] = True
        entry["artifacts"] = artifacts
        runs.append(entry)
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-weights", action="store_true", help="re-hash every checkpoint")
    ap.add_argument("--out", default="reports/snapshot.json")
    args = ap.parse_args()

    runs = run_inventory(args.verify_weights)
    snapshot = {
        "schema": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(ROOT),
        "git": git_state(),
        "code": inventory([f"{d}/**/*.py" for d in CODE_DIRS] + [f"{d}/**/*.json" for d in CODE_DIRS] + ["pyproject.toml"]),
        "data": inventory(DATA_GLOBS),
        "reports": inventory(REPORT_GLOBS),
        "external": inventory([f"{d}/**/*.py" for d in EXTERNAL_DIRS]) if any((ROOT / d).exists() for d in EXTERNAL_DIRS) else {},
        "weights_verified": args.verify_weights,
        "run_count": len(runs),
        "runs": runs,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=1, ensure_ascii=False) + "\n")
    missing = [r["run"] for r in runs if r.get("checkpoint_missing")]
    mismatched = [r["run"] for r in runs if r.get("checkpoint_sha256_matches") is False]
    print(f"wrote {rel(out)}: {len(runs)} runs, {len(snapshot['code'])} code, "
          f"{len(snapshot['data'])} data, {len(snapshot['reports'])} report files")
    print(f"missing checkpoints: {len(missing)}" + (f" -> {missing[:5]}" if missing else ""))
    if args.verify_weights:
        print(f"sha mismatches: {len(mismatched)}" + (f" -> {mismatched[:5]}" if mismatched else ""))


if __name__ == "__main__":
    main()
