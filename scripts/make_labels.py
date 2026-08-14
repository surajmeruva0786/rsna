"""Build the training label table: gold where it exists, report-derived elsewhere.

Writes `work/labels.csv` with one row per training study, the twelve targets as soft
values in [0, 1], and a `is_gold` flag so training can weight the 58 hand-labelled
studies above the 4,349 weakly-labelled ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from report_labeler import TARGETS, label_report  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "rsna-knee-abnormality-detection"
WORK = ROOT / "work"


def main() -> int:
    WORK.mkdir(exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")

    weak = pd.DataFrame([label_report(r) for r in train["Report"].fillna("")], columns=TARGETS)
    is_gold = train[TARGETS].notna().all(axis=1).to_numpy()

    out = pd.DataFrame({"StudyInstanceUID": train["StudyInstanceUID"], "is_gold": is_gold.astype(int)})
    for t in TARGETS:
        # Gold labels override the labeller wherever a radiologist annotation exists.
        out[t] = np.where(is_gold, train[t].fillna(0.0), weak[t])

    out.to_csv(WORK / "labels.csv", index=False)

    print(f"wrote {WORK / 'labels.csv'}  ({len(out)} studies, {int(is_gold.sum())} gold)\n")
    summary = pd.DataFrame(
        {
            "mean_score": out[TARGETS].mean(),
            "asserted(>=0.9)": (out[TARGETS] >= 0.9).sum(),
            "soft(0.3-0.9)": ((out[TARGETS] >= 0.3) & (out[TARGETS] < 0.9)).sum(),
            "negated(<0.05)": (out[TARGETS] < 0.05).sum(),
        }
    )
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
