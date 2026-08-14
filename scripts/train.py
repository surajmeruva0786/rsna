"""Train the study-level knee-MRI classifier on the preprocessed cache.

Runs cross-validated folds, writes per-fold weights and an out-of-fold prediction
matrix, and reports macro AUC -- the competition metric -- both against the weak
report-derived labels and, separately, against the 58 gold-annotated studies.

CUDA is required and is asserted, not silently fallen back from: this is tuned for a
4 GB Quadro P1000 and a CPU run would take days rather than hours.

    python scripts/train.py --folds 5 --epochs 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from model import N_SLOTS, KneeNet  # noqa: E402
from report_labeler import TARGETS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("RSNA_CACHE", ROOT / "work" / "cache"))
WORK = ROOT / "work"

# Horizontal flip is deliberately absent from the augmentation set. Six of the twelve
# targets are laterality-specific (Medial/Lateral Meniscus, Medial/Lateral OA, MCL),
# and mirroring a knee swaps medial for lateral -- it would silently corrupt the label
# of every flipped sample.


class StudyDataset(Dataset):
    def __init__(self, mm_path: Path, shape: tuple, rows: np.ndarray, masks: np.ndarray,
                 labels: np.ndarray | None, size: int, train: bool,
                 weights: np.ndarray | None = None):
        self.mm_path, self.shape = mm_path, shape
        self.rows, self.masks, self.labels = rows, masks, labels
        self.size, self.train = size, train
        self.weights = weights
        self._mm = None

    def __len__(self) -> int:
        return len(self.rows)

    def _memmap(self) -> np.memmap:
        # Opened lazily so each DataLoader worker gets its own handle; a memmap cannot
        # be pickled across the fork/spawn boundary.
        if self._mm is None:
            self._mm = np.memmap(self.mm_path, dtype=np.uint8, mode="r", shape=self.shape)
        return self._mm

    def __getitem__(self, i: int):
        vol = np.asarray(self._memmap()[self.rows[i]], dtype=np.float32) / 255.0
        x = torch.from_numpy(vol)  # (6, slices, H, W)
        mask = torch.from_numpy(self.masks[i].astype(np.float32))

        if self.size != x.shape[-1]:
            x = torch.nn.functional.interpolate(
                x, size=(self.size, self.size), mode="bilinear", align_corners=False
            )

        if self.train:
            x = self._augment(x)

        out = [x, mask]
        if self.labels is not None:
            out.append(torch.from_numpy(self.labels[i].astype(np.float32)))
            w = 1.0 if self.weights is None else float(self.weights[i])
            out.append(torch.tensor(w, dtype=torch.float32))
        return tuple(out)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Geometric and intensity jitter applied identically to all slices of a slot."""
        if torch.rand(1).item() < 0.8:
            # Small affine: knees are consistently positioned, so large warps would
            # move the model off-distribution rather than regularise it.
            angle = (torch.rand(1).item() * 2 - 1) * 12.0 * np.pi / 180
            scale = 1.0 + (torch.rand(1).item() * 2 - 1) * 0.12
            tx = (torch.rand(1).item() * 2 - 1) * 0.08
            ty = (torch.rand(1).item() * 2 - 1) * 0.08
            cos, sin = np.cos(angle) / scale, np.sin(angle) / scale
            theta = torch.tensor([[[cos, -sin, tx], [sin, cos, ty]]], dtype=torch.float32)
            theta = theta.expand(x.shape[0], 2, 3)
            grid = torch.nn.functional.affine_grid(theta, list(x.shape), align_corners=False)
            x = torch.nn.functional.grid_sample(x, grid, align_corners=False, padding_mode="zeros")

        if torch.rand(1).item() < 0.5:
            x = torch.clamp(x * (0.8 + torch.rand(1).item() * 0.4) + (torch.rand(1).item() - 0.5) * 0.1, 0, 1)

        # Drop a whole slot occasionally so the attention head cannot become dependent
        # on any single plane always being present -- test studies vary in coverage.
        if torch.rand(1).item() < 0.2:
            x[torch.randint(0, x.shape[0], (1,)).item()] = 0
        return x


def macro_auc(y: np.ndarray, p: np.ndarray) -> tuple[float, list[float]]:
    aucs = []
    for j in range(y.shape[1]):
        yj = y[:, j]
        aucs.append(roc_auc_score(yj, p[:, j]) if len(np.unique(yj)) > 1 else float("nan"))
    return float(np.nanmean(aucs)), aucs


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. This pipeline is built for the local Quadro P1000; "
            "refusing to fall back to CPU, which would take days. Check the driver and "
            "that torch was installed with CUDA support."
        )
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {name}  {total:.1f} GiB  sm_{cap[0]}{cap[1]}  torch {torch.__version__}")
    return dev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--cache-size", type=int, default=224, help="size the cache was built at")
    ap.add_argument("--slices", type=int, default=16)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only-fold", type=int, default=-1)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", help="fp16 autocast (slow on Pascal)")
    ap.add_argument("--gold-weight", type=float, default=5.0)
    ap.add_argument("--subset", type=int, default=0,
                    help="debug: train on only N studies, to smoke-test the full loop cheaply")
    ap.add_argument("--out", default="run1")
    args = ap.parse_args()

    device = require_cuda()
    torch.backends.cudnn.benchmark = True
    outdir = WORK / "runs" / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(CACHE / "train_index.csv")
    labels_df = pd.read_csv(WORK / "labels.csv").set_index("StudyInstanceUID")
    labels_df = labels_df.loc[index["StudyInstanceUID"]].reset_index()

    slot_cols = [f"slot{i}" for i in range(N_SLOTS)]
    masks = index[slot_cols].to_numpy()
    y_soft = labels_df[TARGETS].to_numpy(np.float32)
    is_gold = labels_df["is_gold"].to_numpy()
    y_hard = (y_soft >= 0.5).astype(int)

    usable = masks.sum(axis=1) > 0
    print(f"studies: {len(index)} total, {usable.sum()} with >=1 usable series, {is_gold.sum()} gold")
    keep = np.where(usable)[0]
    if args.subset:
        # Keep the gold studies in the subset so the gold-subset metric path is exercised.
        gold_rows = keep[is_gold[keep] == 1]
        rest = keep[is_gold[keep] == 0][: max(args.subset - len(gold_rows), 0)]
        keep = np.sort(np.concatenate([gold_rows, rest]))
        print(f"DEBUG subset: {len(keep)} studies")

    shape = (len(index), N_SLOTS, args.slices, args.cache_size, args.cache_size)
    mm_path = CACHE / f"train_{args.cache_size}.u8"

    # Stratify on the number of positive findings: it keeps rare multi-finding studies
    # from clustering in one fold, which a plain KFold over 12 sparse labels would allow.
    strat = np.clip(y_hard[keep].sum(axis=1), 0, 5)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    oof = np.zeros((len(index), len(TARGETS)), np.float32)
    oof_seen = np.zeros(len(index), bool)
    history = []

    for fold, (tr_i, va_i) in enumerate(skf.split(keep, strat)):
        if args.only_fold >= 0 and fold != args.only_fold:
            continue
        tr, va = keep[tr_i], keep[va_i]
        print(f"\n===== fold {fold}: {len(tr)} train / {len(va)} val =====", flush=True)

        # The 58 gold studies are the only labels not filtered through the report
        # labeller, so they are worth several weak studies each.
        w_tr = np.where(is_gold[tr] == 1, args.gold_weight, 1.0).astype(np.float32)
        ds_tr = StudyDataset(mm_path, shape, tr, masks[tr], y_soft[tr], args.size, True, w_tr)
        ds_va = StudyDataset(mm_path, shape, va, masks[va], y_soft[va], args.size, False)
        dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                           pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
        dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                           pin_memory=True, persistent_workers=args.workers > 0)

        model = KneeNet(args.backbone, n_slices=args.slices, n_targets=len(TARGETS)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        steps = max(1, len(dl_tr) // args.accum) * args.epochs
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.25)
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
        # Reduction is manual so gold studies can be up-weighted per sample.
        crit = nn.BCEWithLogitsLoss(reduction="none")

        best_auc, best_path = -1.0, outdir / f"fold{fold}.pt"
        for epoch in range(args.epochs):
            model.train()
            t0, tot, nb = time.time(), 0.0, 0
            opt.zero_grad(set_to_none=True)
            for bi, (x, m, yb, wb) in enumerate(dl_tr):
                x, m, yb = x.to(device, non_blocking=True), m.to(device), yb.to(device)
                wb = wb.to(device)
                with torch.amp.autocast("cuda", enabled=args.amp):
                    per_sample = crit(model(x, m), yb).mean(dim=1)
                    loss = (per_sample * wb).sum() / wb.sum() / args.accum
                scaler.scale(loss).backward()
                if (bi + 1) % args.accum == 0:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    if sched.last_epoch < steps - 1:
                        sched.step()
                tot += loss.item() * args.accum
                nb += 1

            model.eval()
            preds = []
            with torch.no_grad():
                # Starred: the dataset yields a sample weight after the label whenever
                # labels are present, and the validation set has labels too.
                for x, m, *_ in dl_va:
                    with torch.amp.autocast("cuda", enabled=args.amp):
                        preds.append(torch.sigmoid(model(x.to(device), m.to(device))).float().cpu().numpy())
            p = np.concatenate(preds)
            auc, _ = macro_auc(y_hard[va], p)
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"  epoch {epoch+1}/{args.epochs}  loss {tot/max(nb,1):.4f}  "
                  f"val macroAUC {auc:.4f}  {(time.time()-t0)/60:.1f}m  peak {peak:.2f} GiB", flush=True)
            history.append(dict(fold=fold, epoch=epoch + 1, loss=tot / max(nb, 1), auc=auc))

            if auc > best_auc:
                best_auc = auc
                torch.save({"model": model.state_dict(), "args": vars(args), "auc": auc}, best_path)
                oof[va] = p
                oof_seen[va] = True

        print(f"  fold {fold} best macroAUC {best_auc:.4f}  -> {best_path.name}", flush=True)
        del model, opt, dl_tr, dl_va
        torch.cuda.empty_cache()

    seen = np.where(oof_seen)[0]
    if len(seen):
        auc, per = macro_auc(y_hard[seen], oof[seen])
        print(f"\n==== OOF macro AUC (weak labels, {len(seen)} studies): {auc:.4f} ====")
        print(pd.Series(per, index=TARGETS).to_string(float_format="{:.4f}".format))

        g = seen[is_gold[seen] == 1]
        if len(g) > 10:
            gauc, _ = macro_auc(y_hard[g], oof[g])
            print(f"\nOOF macro AUC on the {len(g)} gold-labelled studies: {gauc:.4f}")

        np.save(outdir / "oof.npy", oof)
        pd.DataFrame(history).to_csv(outdir / "history.csv", index=False)
        json.dump({"oof_macro_auc": auc, "per_target": dict(zip(TARGETS, per))},
                  open(outdir / "metrics.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
