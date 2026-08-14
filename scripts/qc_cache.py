"""Visual QC for the preprocessed cache.

Silent preprocessing bugs -- inverted MONOCHROME1 series, a percentile stretch that
crushes contrast, slices ordered by a tag that turned out to be meaningless -- do not
raise exceptions. They just quietly train a worse model. This dumps a montage so the
cache can be looked at before hours are spent training on it.

    python scripts/qc_cache.py --split train --study 0 --out work/qc.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("RSNA_CACHE", ROOT / "work" / "cache"))
N_SLOTS = 6
SLOT_NAMES = ["Ax", "Cor", "Sag", "Ax-FS", "Cor-FS", "Sag-FS"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--study", type=int, default=0, help="row index in the cache")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--slices", type=int, default=16)
    ap.add_argument("--every", type=int, default=4, help="show every Nth slice")
    ap.add_argument("--out", default="work/qc.png")
    args = ap.parse_args()

    index = pd.read_csv(CACHE / f"{args.split}_index.csv")
    shape = (len(index), N_SLOTS, args.slices, args.size, args.size)
    mm = np.memmap(CACHE / f"{args.split}_{args.size}.u8", dtype=np.uint8, mode="r", shape=shape)

    row = args.study
    vol = np.asarray(mm[row])
    cols = list(range(0, args.slices, args.every))
    tile = 160

    canvas = np.zeros((N_SLOTS * tile, len(cols) * tile), np.uint8)
    for s in range(N_SLOTS):
        for c, j in enumerate(cols):
            img = cv2.resize(vol[s, j], (tile, tile), interpolation=cv2.INTER_AREA)
            canvas[s * tile : (s + 1) * tile, c * tile : (c + 1) * tile] = img

    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    for s in range(N_SLOTS):
        present = int(index.iloc[row][f"slot{s}"])
        cv2.putText(canvas, f"{SLOT_NAMES[s]}{'' if present else ' (absent)'}",
                    (4, s * tile + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 0) if present else (0, 0, 255), 1, cv2.LINE_AA)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)

    print(f"study row {row}: {index.iloc[row]['StudyInstanceUID']}")
    print(f"slots present: {[SLOT_NAMES[s] for s in range(N_SLOTS) if index.iloc[row][f'slot{s}']]}")
    print(f"intensity: mean {vol[vol > 0].mean():.1f}  nonzero {100*(vol>0).mean():.1f}%")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
