"""Score `report_labeler` against the 58 gold-labelled training studies.

These 58 rows are the only ground truth available for the report->label step, so they
are the only check on whether the weak labels driving the image model are any good.
Reports AUC, and precision/recall at a 0.5 threshold, per target.

Usage:
    python scripts/validate_labeler.py [--errors ACL]   # dump misses for one target
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from report_labeler import TARGETS, label_report  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "rsna-knee-abnormality-detection"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors", help="dump misclassified reports for this target")
    ap.add_argument("--limit", type=int, default=6)
    args = ap.parse_args()

    train = pd.read_csv(DATA / "train.csv")
    gold = train[train[TARGETS].notna().all(axis=1)].reset_index(drop=True)
    print(f"gold-labelled studies: {len(gold)}\n")

    preds = pd.DataFrame([label_report(r) for r in gold["Report"]], columns=TARGETS)

    rows = []
    for t in TARGETS:
        y = gold[t].to_numpy()
        p = preds[t].to_numpy()
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
        hard = (p >= 0.5).astype(int)
        tp = int(((hard == 1) & (y == 1)).sum())
        fp = int(((hard == 1) & (y == 0)).sum())
        fn = int(((hard == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec and rec and prec + rec else float("nan")
        rows.append(
            dict(target=t, pos=int(y.sum()), auc=auc, precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn)
        )

    res = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(res.to_string(index=False))
    print(f"\nmacro AUC (report labeller vs gold): {res['auc'].mean():.4f}")
    print(f"macro F1  @0.5:                      {res['f1'].mean():.4f}")

    if args.errors:
        t = args.errors
        y = gold[t].to_numpy()
        p = preds[t].to_numpy()
        hard = (p >= 0.5).astype(int)
        for kind, mask in (("FALSE NEGATIVE", (hard == 0) & (y == 1)), ("FALSE POSITIVE", (hard == 1) & (y == 0))):
            idx = np.where(mask)[0][: args.limit]
            for i in idx:
                print(f"\n===== {t} {kind} (score={p[i]:.2f}) =====")
                print(gold["Report"].iloc[i][:1200])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
