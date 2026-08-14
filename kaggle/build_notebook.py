"""Wrap kaggle_inference.py into submission.ipynb.

Kaggle scores notebooks, not scripts, but a notebook is a poor place to keep logic under
review. The logic therefore lives in `kaggle_inference.py` and this script mechanically
wraps it, so the reviewed file and the submitted file cannot drift apart.

    python kaggle/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

HEADER = """\
# RSNA Knee Abnormality Detection - inference

Study-level multi-label model over knee MRI. Each study is encoded as up to six
plane x fluid-sensitivity slots, each slot fed to the backbone as a 16-channel image,
then pooled across slots by masked gated attention.

Setup:
1. Attach the competition dataset.
2. Attach the trained folds as a Dataset and set `WEIGHTS_DIR` to its path.
3. Internet must be **off**; nothing here downloads anything.

Writes `/kaggle/working/submission.csv`.
"""


def main() -> int:
    src = (HERE / "kaggle_inference.py").read_text(encoding="utf-8")
    # Strip the __main__ guard: the notebook calls main() from its own cell.
    body = src.split('if __name__ == "__main__":')[0].rstrip() + "\n"

    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": HEADER.splitlines(keepends=True)},
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": body.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["main()\n"],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out = HERE / "submission.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out}  ({len(body.splitlines())} source lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
