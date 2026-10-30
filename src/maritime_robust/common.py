import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def environment():
    import torch
    import torchvision
    return {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def directory_bytes(path):
    total = 0
    for base, _, files in os.walk(path):
        for name in files:
            p = Path(base) / name
            if not p.is_symlink():
                try:
                    total += p.stat().st_size
                except FileNotFoundError:
                    pass
    return total


class Budget:
    """One exclusive GPU job, persistent elapsed-time ledger, interruption accounting."""
    def __init__(self, phase, reserve_seconds=60):
        self.phase = phase
        self.reserve_seconds = reserve_seconds

    def __enter__(self):
        import fcntl
        self.f = (ROOT / "artifacts/gpu.lock").open("a+")
        fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.path = ROOT / "artifacts/budget.json"
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {"jobs": []}
        # Dead jobs are charged until their last heartbeat, at most one batch unaccounted.
        for job in self.state["jobs"]:
            if job["status"] == "running":
                job["status"] = "interrupted"
        self.job = {"phase": self.phase, "pid": os.getpid(), "start": time.time(),
                    "elapsed_seconds": 0, "status": "running"}
        self.state["jobs"].append(self.job)
        self.started = time.monotonic()
        self.last_write = 0
        self.check(force=True)
        return self

    def check(self, force=False):
        self.job["elapsed_seconds"] = time.monotonic() - self.started
        phase_limits = {"preflight": 8, "baseline": 24, "augmentation": 24,
                        "evaluation": 16, "cifar": 30}
        total = sum(j["elapsed_seconds"] for j in self.state["jobs"])
        phase = sum(j["elapsed_seconds"] for j in self.state["jobs"] if j["phase"] == self.phase)
        if force or time.monotonic() - self.last_write > 15:
            write_json(self.path, self.state)
            self.last_write = time.monotonic()
            if directory_bytes(ROOT) > 50 * 10**9:
                raise RuntimeError("50 GB project disk budget exhausted")
        if total + self.reserve_seconds >= 72 * 3600 or phase + self.reserve_seconds >= phase_limits[self.phase] * 3600:
            raise RuntimeError(f"GPU budget exhausted: total={total / 3600:.3f} h, phase={phase / 3600:.3f} h")

    def __exit__(self, typ, value, tb):
        self.job.update(elapsed_seconds=time.monotonic() - self.started,
                        status="complete" if typ is None else "failed", end=time.time())
        write_json(self.path, self.state)
        self.f.close()
