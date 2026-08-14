"""Reduce the 570 GB DICOM tree to a compact uint8 volume cache for training.

Training cannot stream from the raw tree: 819,640 DICOMs across 24,371 series, most of
them compressed, would make every epoch an I/O-bound decode marathon. This pass decodes
once and writes a fixed-shape memmap that an epoch can random-access in a few GB.

Layout
------
Each study is reduced to six *slots* -- the cross product of the three anatomical
planes with the fluid-sensitive flag from `*_series.csv`:

    0 Axial     non-fluid      3 Axial     fluid-sensitive
    1 Coronal   non-fluid      4 Coronal   fluid-sensitive
    2 Sagittal  non-fluid      5 Sagittal  fluid-sensitive

`Fluid_Sensitive` and `Fat_Suppression` are perfectly collinear across the whole
dataset, so one flag carries both. Where a study has several series in one slot the
longest (most slices) wins. Slots with no series are zero-filled and masked out.

Within a series, slices are ordered along the true through-plane axis (the component of
ImagePositionPatient normal to ImageOrientationPatient, falling back to SliceLocation
then InstanceNumber) and then sampled at `--slices` evenly spaced positions, so a
40-slice and a 200-slice acquisition both yield the same tensor covering the same
anatomy.

Outputs (under work/cache/):
    {split}_{size}.u8    memmap, uint8, (n_studies, 6, slices, size, size)
    {split}_index.csv    StudyInstanceUID order plus a present/absent flag per slot
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "rsna-knee-abnormality-detection"

# The DICOM tree lives on a 7200rpm SATA HDD, where the cache's random per-study reads
# would be seek-bound. RSNA_CACHE points it at the NVMe volume instead; training reads
# are ~5 MB per study per step and that difference sets the epoch time.
CACHE = Path(os.environ.get("RSNA_CACHE", ROOT / "work" / "cache"))

PLANES = {"Axial": 0, "Coronal": 1, "Sagittal": 2}
N_SLOTS = 6

# Globals bound once per worker process; the memmap is reopened per process rather
# than pickled across the pool boundary.
_G: dict = {}


# --------------------------------------------------------------------------------
# DICOM -> normalised uint8
# --------------------------------------------------------------------------------


def _sort_key(ds: pydicom.Dataset) -> float:
    """Position of a slice along the through-plane axis.

    InstanceNumber alone is unreliable -- interleaved acquisitions and multi-echo
    series renumber arbitrarily -- so the geometric position is preferred where the
    orientation tags survived the anonymisation.
    """
    try:
        iop = [float(v) for v in ds.ImageOrientationPatient]
        ipp = [float(v) for v in ds.ImagePositionPatient]
        normal = np.cross(iop[:3], iop[3:])
        return float(np.dot(ipp, normal))
    except Exception:
        pass
    for attr in ("SliceLocation", "InstanceNumber"):
        try:
            return float(getattr(ds, attr))
        except Exception:
            continue
    return 0.0


def _to_uint8(arr: np.ndarray, invert: bool) -> np.ndarray:
    """Per-slice robust contrast normalisation.

    Intensities are not comparable across scanners, sequences or vendors in MRI -- there
    is no Hounsfield-like absolute scale -- so each slice is stretched between its own
    0.5/99.5 percentiles. Clipping the tails keeps a single bright vessel or artefact
    from compressing the tissue contrast into a few levels.
    """
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


def load_series(series_dir: Path, n_slices: int, size: int) -> np.ndarray | None:
    """Decode one series into a (n_slices, size, size) uint8 volume."""
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        return None

    # Header-only pass to order the stack. Skipping pixel data here costs one extra
    # open per file but avoids decoding slices that the sampling step will discard --
    # a clear win when a 200-slice series contributes only `n_slices` of them.
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

    idx = np.unique(np.linspace(0, len(ordered) - 1, n_slices).round().astype(int))
    vol = np.zeros((n_slices, size, size), np.uint8)
    picked = []
    for j in idx:
        try:
            ds = pydicom.dcmread(ordered[j])
            px = ds.pixel_array
            if px.ndim == 3:  # rare multiframe object
                px = px[px.shape[0] // 2]
            invert = str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1"
            img = _to_uint8(px, invert)
            picked.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))
        except Exception:
            continue
    if not picked:
        return None

    # Fewer usable slices than requested: repeat the sampled stack to fill, keeping the
    # tensor shape fixed rather than padding with black frames the model would learn on.
    for i in range(n_slices):
        vol[i] = picked[min(int(round(i * (len(picked) - 1) / max(n_slices - 1, 1))), len(picked) - 1)]
    return vol


# --------------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------------


def _init(mm_path: str, shape: tuple, series_root: str, n_slices: int, size: int) -> None:
    _G["mm"] = np.memmap(mm_path, dtype=np.uint8, mode="r+", shape=shape)
    _G["root"] = Path(series_root)
    _G["n_slices"] = n_slices
    _G["size"] = size


def _work(job: tuple) -> tuple:
    row, study, slots = job
    mask = [0] * N_SLOTS
    for slot, series_uid in slots.items():
        vol = load_series(_G["root"] / study / series_uid, _G["n_slices"], _G["size"])
        if vol is not None:
            _G["mm"][row, slot] = vol
            mask[slot] = 1
    return row, study, mask


# --------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------


def build_jobs(series_csv: Path, studies: list[str]) -> list[tuple]:
    """Pick at most one series per (plane, fluid-sensitivity) slot, longest wins."""
    ser = pd.read_csv(series_csv)
    # Restrict before touching the disk: the per-series file count below is a directory
    # listing on a 7200rpm HDD, so scanning series outside the requested studies would
    # dominate the runtime of any partial run.
    ser = ser[ser["StudyInstanceUID"].isin(set(studies))]
    by_study: dict[str, dict[int, str]] = {}
    best_len: dict[tuple[str, int], int] = {}

    root = series_csv.parent / (series_csv.stem.replace("_series", "") + "_series")
    for r in ser.itertuples(index=False):
        plane = PLANES.get(str(r.Anatomical_Plane))
        if plane is None:
            continue
        slot = plane + (N_SLOTS // 2 if int(r.Fluid_Sensitive) == 1 else 0)
        d = root / r.StudyInstanceUID / r.SeriesInstanceUID
        try:
            n = sum(1 for _ in d.glob("*.dcm"))
        except OSError:
            n = 0
        if n == 0:
            continue
        key = (r.StudyInstanceUID, slot)
        if n > best_len.get(key, 0):
            best_len[key] = n
            by_study.setdefault(r.StudyInstanceUID, {})[slot] = r.SeriesInstanceUID

    return [(i, s, by_study.get(s, {})) for i, s in enumerate(studies)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--slices", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N studies")
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    studies = pd.read_csv(DATA / f"{args.split}.csv")["StudyInstanceUID"].tolist()
    if args.limit:
        studies = studies[: args.limit]

    shape = (len(studies), N_SLOTS, args.slices, args.size, args.size)
    mm_path = CACHE / f"{args.split}_{args.size}.u8"
    nbytes = int(np.prod(shape))
    print(f"{args.split}: {len(studies)} studies -> {mm_path.name}  shape={shape}  {nbytes/2**30:.1f} GiB")

    # Preallocate so workers can write disjoint rows into a shared file.
    np.memmap(mm_path, dtype=np.uint8, mode="w+", shape=shape).flush()

    jobs = build_jobs(DATA / f"{args.split}_series.csv", studies)
    print(f"indexed series in {sum(len(j[2]) for j in jobs)} slots across {len(jobs)} studies")

    rows, t0 = [], time.time()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init,
        initargs=(str(mm_path), shape, str(DATA / f"{args.split}_series"), args.slices, args.size),
    ) as ex:
        for k, (row, study, mask) in enumerate(ex.map(_work, jobs, chunksize=4), 1):
            rows.append({"row": row, "StudyInstanceUID": study, **{f"slot{i}": m for i, m in enumerate(mask)}})
            if k % 100 == 0 or k == len(jobs):
                el = time.time() - t0
                eta = el / k * (len(jobs) - k)
                print(f"  {k}/{len(jobs)}  {el/60:.1f}m elapsed  ETA {eta/60:.1f}m", flush=True)

    idx = pd.DataFrame(rows).sort_values("row").reset_index(drop=True)
    idx.to_csv(CACHE / f"{args.split}_index.csv", index=False)

    slot_cols = [f"slot{i}" for i in range(N_SLOTS)]
    print(f"\nwrote {CACHE / f'{args.split}_index.csv'}")
    print("slot coverage:\n", idx[slot_cols].sum().to_string())
    print(f"studies with zero usable series: {(idx[slot_cols].sum(axis=1) == 0).sum()}")
    print(f"total {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
