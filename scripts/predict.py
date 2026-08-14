"""Score a preprocessed split with the trained fold ensemble and write submission.csv.

Averages the per-fold checkpoints written by `train.py`. Studies whose series all failed
to decode fall back to the per-target training prior rather than to 0.5 -- under macro
AUC a constant is rank-neutral, but the prior at least orders them sensibly against each
other if several studies fail.

    python scripts/predict.py --run run1 --split test --out submission.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import N_SLOTS, KneeNet  # noqa: E402
from report_labeler import TARGETS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("RSNA_CACHE", ROOT / "work" / "cache"))
WORK = ROOT / "work"


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; refusing to run inference on CPU.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    return torch.device("cuda")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--cache-size", type=int, default=224)
    ap.add_argument("--slices", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--folds", default="",
                    help="comma-separated fold indices to ensemble, e.g. 0,1. "
                         "Default is every fold*.pt present -- which is WRONG while "
                         "training is still running, because a fold's checkpoint is "
                         "written on every validation improvement, so an in-progress "
                         "fold already has a half-trained file on disk.")
    args = ap.parse_args()

    device = require_cuda()
    rundir = WORK / "runs" / args.run
    if args.folds:
        want = [int(f) for f in args.folds.split(",")]
        ckpts = [rundir / f"fold{f}.pt" for f in want]
        missing = [c for c in ckpts if not c.exists()]
        if missing:
            raise SystemExit(f"missing checkpoints: {[c.name for c in missing]}")
    else:
        ckpts = sorted(rundir.glob("fold*.pt"))
    if not ckpts:
        raise SystemExit(f"no fold checkpoints under {rundir}")
    print(f"ensembling {len(ckpts)} folds: {[c.name for c in ckpts]}")
    for c in ckpts:
        m = torch.load(c, map_location="cpu", weights_only=False)
        print(f"  {c.name}: val macro AUC {m.get('auc', float('nan')):.4f}")

    index = pd.read_csv(CACHE / f"{args.split}_index.csv")
    slot_cols = [f"slot{i}" for i in range(N_SLOTS)]
    masks = index[slot_cols].to_numpy().astype(np.float32)

    shape = (len(index), N_SLOTS, args.slices, args.cache_size, args.cache_size)
    mm = np.memmap(CACHE / f"{args.split}_{args.cache_size}.u8", dtype=np.uint8, mode="r", shape=shape)

    meta = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    size = meta["args"]["size"]
    backbone = meta["args"]["backbone"]

    acc = np.zeros((len(index), len(TARGETS)), np.float64)
    for ck in ckpts:
        state = torch.load(ck, map_location="cpu", weights_only=False)
        model = KneeNet(backbone, n_slices=args.slices, n_targets=len(TARGETS), pretrained=False).to(device)
        model.load_state_dict(state["model"])
        model.eval()

        preds = []
        with torch.no_grad():
            for i in range(0, len(index), args.batch):
                x = torch.from_numpy(np.asarray(mm[i : i + args.batch], np.float32) / 255.0)
                if size != x.shape[-1]:
                    b, s = x.shape[:2]
                    x = torch.nn.functional.interpolate(
                        x.flatten(0, 1), size=(size, size), mode="bilinear", align_corners=False
                    ).view(b, s, -1, size, size)
                m = torch.from_numpy(masks[i : i + args.batch])
                preds.append(torch.sigmoid(model(x.to(device), m.to(device))).cpu().numpy())
        acc += np.concatenate(preds)
        del model
        torch.cuda.empty_cache()
        print(f"  {ck.name} done")

    pred = acc / len(ckpts)

    # A study with no decodable series carries no image evidence; the training prior is
    # a better ranking than an arbitrary constant.
    dead = masks.sum(axis=1) == 0
    if dead.any():
        prior = pd.read_csv(WORK / "labels.csv")[TARGETS].mean().to_numpy()
        pred[dead] = prior
        print(f"{dead.sum()} studies had no decodable series; filled with training prior")

    sub = pd.DataFrame(pred, columns=TARGETS)
    sub.insert(0, "StudyInstanceUID", index["StudyInstanceUID"].to_numpy())
    sub.to_csv(ROOT / args.out, index=False)
    print(f"\nwrote {ROOT / args.out}  ({len(sub)} rows)")
    print(sub.head().to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
