"""Kaggle submission inference: DICOM -> submission.csv, self-contained and offline.

This is the artefact that actually scores. Submissions to this competition are
notebooks, not files: the `submission.csv` produced locally is scored against three
public example studies and means nothing, while this script runs against the ~1,300
hidden test studies inside Kaggle's container.

Constraints it is written against:
  * no internet, so timm backbones are built with pretrained=False and every weight
    comes from an attached Kaggle Dataset;
  * <=9 h runtime, so DICOM decode runs in DataLoader workers and overlaps the GPU;
  * test studies must all appear in the output even if every series fails to decode.

Attach the trained folds as a Kaggle Dataset and point WEIGHTS_DIR at it.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

COMP = Path("/kaggle/input/rsna-knee-abnormality-detection")
WEIGHTS_DIR = Path("/kaggle/input/rsna-knee-folds")
OUT = Path("/kaggle/working/submission.csv")

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
PLANES = {"Axial": 0, "Coronal": 1, "Sagittal": 2}
N_SLOTS, N_SLICES, SIZE = 6, 16, 224

# Fallback ranking for studies with no decodable series: the training-set prevalence of
# each target. Rank-neutral among themselves but sensibly ordered against each other.
PRIOR = np.array([0.174, 0.101, 0.298, 0.161, 0.174, 0.132,
                  0.227, 0.446, 0.213, 0.258, 0.261, 0.139], np.float32)


# ---------------------------------------------------------------------------
# Model (mirrors scripts/model.py; inlined so the notebook needs no repo)
# ---------------------------------------------------------------------------


class SlotAttentionPool(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh())
        self.gate = nn.Sequential(nn.Linear(dim, hidden), nn.Sigmoid())
        self.score = nn.Linear(hidden, 1)

    def forward(self, x, mask):
        w = self.score(self.attn(x) * self.gate(x)).squeeze(-1)
        w = w.masked_fill(mask < 0.5, float("-inf"))
        empty = mask.sum(dim=1, keepdim=True) < 0.5
        w = torch.where(empty.expand_as(w), torch.zeros_like(w), w)
        return torch.einsum("bs,bsd->bd", F.softmax(w, dim=1), x)


class KneeNet(nn.Module):
    def __init__(self, backbone="efficientnet_b0", n_slices=N_SLICES, n_targets=12, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, in_chans=n_slices, num_classes=0)
        dim = self.backbone.num_features
        self.slot_embed = nn.Parameter(torch.zeros(N_SLOTS, dim))
        self.pool = SlotAttentionPool(dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_targets))

    def forward(self, x, mask):
        b, s = x.shape[:2]
        feat = self.backbone(x.flatten(0, 1)).view(b, s, -1)
        feat = feat + self.slot_embed.unsqueeze(0)
        return self.head(self.pool(feat, mask))


# ---------------------------------------------------------------------------
# DICOM decode (mirrors scripts/preprocess.py)
# ---------------------------------------------------------------------------


def _sort_key(ds) -> float:
    try:
        iop = [float(v) for v in ds.ImageOrientationPatient]
        ipp = [float(v) for v in ds.ImagePositionPatient]
        return float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
    except Exception:
        pass
    for attr in ("SliceLocation", "InstanceNumber"):
        try:
            return float(getattr(ds, attr))
        except Exception:
            continue
    return 0.0


def _to_uint8(arr, invert):
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, (0.5, 99.5))
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            return np.zeros(arr.shape, np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    if invert:
        arr = 1.0 - arr
    return (arr * 255.0).astype(np.uint8)


def load_series(d: Path):
    files = sorted(d.glob("*.dcm"))
    if not files:
        return None
    heads = []
    for f in files:
        try:
            heads.append((_sort_key(pydicom.dcmread(f, stop_before_pixels=True)), f))
        except Exception:
            continue
    if not heads:
        return None
    heads.sort(key=lambda t: t[0])
    ordered = [f for _, f in heads]

    picked = []
    for j in np.unique(np.linspace(0, len(ordered) - 1, N_SLICES).round().astype(int)):
        try:
            ds = pydicom.dcmread(ordered[j])
            px = ds.pixel_array
            if px.ndim == 3:
                px = px[px.shape[0] // 2]
            img = _to_uint8(px, str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1")
            picked.append(cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
        except Exception:
            continue
    if not picked:
        return None
    vol = np.zeros((N_SLICES, SIZE, SIZE), np.uint8)
    for i in range(N_SLICES):
        vol[i] = picked[min(int(round(i * (len(picked) - 1) / max(N_SLICES - 1, 1))), len(picked) - 1)]
    return vol


class TestStudies(Dataset):
    """One item per study: decodes its <=6 slots on a DataLoader worker."""

    def __init__(self, studies, slots, root: Path):
        self.studies, self.slots, self.root = studies, slots, root

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        study = self.studies[i]
        x = np.zeros((N_SLOTS, N_SLICES, SIZE, SIZE), np.uint8)
        mask = np.zeros(N_SLOTS, np.float32)
        for slot, series_uid in self.slots.get(study, {}).items():
            vol = load_series(self.root / study / series_uid)
            if vol is not None:
                x[slot] = vol
                mask[slot] = 1.0
        return torch.from_numpy(x).float().div_(255.0), torch.from_numpy(mask), i


def build_slots(series_csv: Path, root: Path):
    ser = pd.read_csv(series_csv)
    slots, best = {}, {}
    for r in ser.itertuples(index=False):
        plane = PLANES.get(str(r.Anatomical_Plane))
        if plane is None:
            continue
        slot = plane + (N_SLOTS // 2 if int(r.Fluid_Sensitive) == 1 else 0)
        try:
            n = sum(1 for _ in (root / r.StudyInstanceUID / r.SeriesInstanceUID).glob("*.dcm"))
        except OSError:
            n = 0
        if n == 0:
            continue
        key = (r.StudyInstanceUID, slot)
        if n > best.get(key, 0):
            best[key] = n
            slots.setdefault(r.StudyInstanceUID, {})[slot] = r.SeriesInstanceUID
    return slots


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    studies = pd.read_csv(COMP / "test.csv")["StudyInstanceUID"].tolist()
    root = COMP / "test_series"
    slots = build_slots(COMP / "test_series.csv", root)
    print(f"{len(studies)} studies, {sum(len(v) for v in slots.values())} usable series")

    ckpts = sorted(WEIGHTS_DIR.glob("fold*.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {WEIGHTS_DIR}")
    meta = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    backbone = meta["args"]["backbone"]
    size = meta["args"]["size"]
    print(f"{len(ckpts)} folds, backbone={backbone}, size={size}")

    models = []
    for ck in ckpts:
        m = KneeNet(backbone, n_targets=len(TARGETS)).to(device).eval()
        m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"])
        models.append(m)

    dl = DataLoader(
        TestStudies(studies, slots, root),
        batch_size=2,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 2),
        pin_memory=True,
    )

    pred = np.zeros((len(studies), len(TARGETS)), np.float32)
    seen = np.zeros(len(studies), bool)
    with torch.no_grad():
        for x, mask, idx in dl:
            if size != x.shape[-1]:
                b, s = x.shape[:2]
                x = F.interpolate(x.flatten(0, 1), size=(size, size),
                                  mode="bilinear", align_corners=False).view(b, s, -1, size, size)
            x, mask = x.to(device), mask.to(device)
            acc = sum(torch.sigmoid(m(x, mask)) for m in models) / len(models)
            pred[idx.numpy()] = acc.cpu().numpy()
            seen[idx.numpy()] = mask.sum(dim=1).cpu().numpy() > 0

    if (~seen).any():
        print(f"{(~seen).sum()} studies had no decodable series; using training prior")
        pred[~seen] = PRIOR

    sub = pd.DataFrame(pred, columns=TARGETS)
    sub.insert(0, "StudyInstanceUID", studies)
    sub.to_csv(OUT, index=False)
    print(f"wrote {OUT}  {sub.shape}")
    print(sub.head().to_string(index=False))


if __name__ == "__main__":
    main()
