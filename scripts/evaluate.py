"""Evaluate the OOF predictions, with emphasis on the 58 gold-labelled studies.

Two numbers come out of training and they mean different things:

* OOF macro AUC against the **weak** labels measures how well the image model reproduces
  the report labeller. It is inflated as an estimate of competition performance, because
  the labeller is itself only ~0.756 against gold -- imitating it perfectly would still
  inherit all of its mistakes.
* OOF macro AUC against the **58 gold** studies is the honest estimate, because those
  labels were produced the way the test labels were: by reading the images.

The gold set is tiny, so a point estimate alone is misleading. This bootstraps a
confidence interval over studies to show how much of the difference between any two
numbers is signal and how much is 58 samples.

    python scripts/evaluate.py --run run1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from report_labeler import TARGETS, label_report  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "rsna-knee-abnormality-detection"
CACHE = Path(os.environ.get("RSNA_CACHE", ROOT / "work" / "cache"))
WORK = ROOT / "work"


def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    aucs = [
        roc_auc_score(y[:, j], p[:, j]) if len(np.unique(y[:, j])) > 1 else np.nan
        for j in range(y.shape[1])
    ]
    return float(np.nanmean(aucs))


def bootstrap_ci(y: np.ndarray, p: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile CI over resampled *studies* (not targets)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        # A resample can leave a target single-class; macro_auc already nan-skips those.
        with np.errstate(invalid="ignore"):
            try:
                vals.append(macro_auc(y[idx], p[idx]))
            except ValueError:
                continue
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run1")
    args = ap.parse_args()

    oof = np.load(WORK / "runs" / args.run / "oof.npy")
    index = pd.read_csv(CACHE / "train_index.csv")
    labels = pd.read_csv(WORK / "labels.csv").set_index("StudyInstanceUID")
    labels = labels.loc[index["StudyInstanceUID"]].reset_index()

    is_gold = labels["is_gold"].to_numpy() == 1
    y_hard = (labels[TARGETS].to_numpy() >= 0.5).astype(int)

    print(f"OOF weak-label macro AUC ({len(oof)} studies): {macro_auc(y_hard, oof):.4f}\n")

    # Gold evaluation. Compare three things on the same 58 studies: the image model, the
    # report labeller that taught it, and chance.
    g = np.where(is_gold)[0]
    train = pd.read_csv(DATA / "train.csv").set_index("StudyInstanceUID")
    gold_uids = index["StudyInstanceUID"].to_numpy()[g]
    y_gold = train.loc[gold_uids, TARGETS].to_numpy().astype(int)
    p_model = oof[g]
    p_labeller = np.array([[label_report(train.loc[u, "Report"])[t] for t in TARGETS] for u in gold_uids])

    m_auc = macro_auc(y_gold, p_model)
    l_auc = macro_auc(y_gold, p_labeller)
    lo, hi = bootstrap_ci(y_gold, p_model)

    print(f"=== Gold subset ({len(g)} studies) ===")
    print(f"image model (OOF) : {m_auc:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"report labeller   : {l_auc:.4f}   (training-time only; no reports at test time)")
    print(f"chance            : 0.5000\n")

    rows = []
    for j, t in enumerate(TARGETS):
        yj = y_gold[:, j]
        single = len(np.unique(yj)) < 2
        rows.append(
            dict(
                target=t,
                pos=int(yj.sum()),
                model=np.nan if single else roc_auc_score(yj, p_model[:, j]),
                labeller=np.nan if single else roc_auc_score(yj, p_labeller[:, j]),
            )
        )
    df = pd.DataFrame(rows)
    df["delta"] = df["model"] - df["labeller"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
