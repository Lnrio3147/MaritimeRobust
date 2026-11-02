"""Phase 2 of docs/MODULE_NCF.md: estimate the deployment field distribution.

The estimator of phase 1 maps one image to one field. What the training arm needs
is a *distribution* of fields, estimated from unlabeled deployment images — no
labels, no clean pairs, no family names. This script runs the estimator over a
deployment pool and stores the fields as a bank, which is then the only source
of training degradations for ``--aug ncf``.

The pool is built by corrupting held-out test images with the *named* families
only to create the deployment images; the estimator never sees a name. Tier B
(§5) builds the pool with the held-out families excluded, which is the honest
retest of the unknown-family question.

    python scripts/cifar_field_bank.py --out artifacts/fields_a.npz
    python scripts/cifar_field_bank.py --exclude fog,frost,snow --out artifacts/fields_b.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from maritime_robust.cifar_field_estimator import FieldEstimator  # noqa: E402
from maritime_robust.external import cifar_c  # noqa: E402

def deployment_conditions(severities: list[int], exclude: tuple = ()) -> list[tuple]:
    """Single-family conditions that seed the pool, as (family, severity).

    Deliberately **single** families only. The compound conditions the coverage
    arms are tested on are then never in the pool: NCF has to reach them by
    composing two estimated fields. That is the claim, and it is the ablation
    ``--field-compound`` off.

    ``exclude`` drops families entirely, which is the unknown-family test: a
    family the estimator was never shown, evaluated on every condition that
    contains it.
    """
    fams = [n for n in cifar_c.NAMES if n not in exclude]
    return [(f, s) for f in fams for s in severities]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-condition", type=int, default=16)
    ap.add_argument("--severities", default="1,2,3,4,5")
    ap.add_argument("--exclude", default="", help="comma-separated families to hold out")
    ap.add_argument("--ckpt", default="artifacts/field_estimator.pt")
    ap.add_argument("--out", default="artifacts/fields_a.npz")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FieldEstimator().to(device)
    model.load_state_dict(torch.load(PROJECT / args.ckpt, map_location=device))
    model.eval()

    import torchvision
    ds = torchvision.datasets.CIFAR10(PROJECT / "data/cifar", train=False, download=False)
    rng = np.random.default_rng(17)
    idx = rng.choice(len(ds), size=args.per_condition, replace=False)
    images = np.asarray(ds.data)[idx]

    sevs = [int(s) for s in args.severities.split(",")]
    exclude = tuple(f for f in args.exclude.split(",") if f)
    conds = deployment_conditions(sevs, exclude)
    Ks, Ns, Ss, Bs, meta = [], [], [], [], []
    with torch.no_grad():
        for ci, (fam, sev) in enumerate(conds):
            batch = np.stack([cifar_c.apply(im, fam, sev) for im in images])
            x = torch.from_numpy(batch).permute(0, 3, 1, 2).float().div_(255.).to(device)
            k, n, s = model.coarse(x)
            blur = model.blur_params(x).cpu().numpy()
            for i in range(len(batch)):
                Ks.append(k[i, 0].cpu().numpy().astype(np.float16))
                Ns.append(n[i].cpu().numpy().astype(np.float16))
                Ss.append(float(s[i]))
                Bs.append(blur[i].astype(np.float32))
                meta.append({"family": fam, "severity": sev})
            if ci % 20 == 0:
                print(f"{ci}/{len(conds)} conditions", flush=True)
    np.savez_compressed(PROJECT / args.out,
                        K=np.stack(Ks)[:, None], N=np.stack(Ns),
                        sigma=np.array(Ss, np.float32), blur=np.stack(Bs))
    (PROJECT / args.out).with_suffix(".json").write_text(json.dumps({"meta": meta}, indent=0))
    print(f"{len(Ks)} fields from {len(conds)} conditions -> {args.out}")


if __name__ == "__main__":
    main()
