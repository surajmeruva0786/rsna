# Submitting

Kaggle scores a **notebook**, not a file. The `submission.csv` in the repo root is a
format check against the three public example studies and carries no leaderboard
meaning — it cannot be uploaded. `submission.ipynb` is the artefact that scores.

## 1. Upload the weights as a Dataset

Internet is disabled during scoring, so every weight has to arrive as an attached
Dataset — including the backbone, which is why the model is built with
`pretrained=False` and loaded entirely from the checkpoints.

```bash
python scripts/train.py --folds 5 --epochs 10 --batch 2 --accum 8 --out run1
mkdir -p work/kaggle_dataset && cp work/runs/run1/fold*.pt work/kaggle_dataset/
```

`work/kaggle_dataset/` is **85 MB** (five folds × 17.7 MB). Upload it as a new Kaggle
Dataset — name it `rsna-knee-folds` to match the default, or set `WEIGHTS_DIR` in the
notebook to whatever path it lands at.

## 2. Build and upload the notebook

```bash
python kaggle/build_notebook.py     # kaggle_inference.py -> submission.ipynb
```

The logic lives in `kaggle_inference.py` and the notebook is generated from it, so the
file under review and the file that runs cannot drift apart.

In the Kaggle editor:

1. Attach the competition dataset.
2. Attach the weights Dataset from step 1.
3. **Turn internet off.**
4. Set accelerator to GPU.
5. Run all, then Submit.

## 3. Verify before submitting

The notebook prints a running ETA. Expect roughly:

| | |
| --- | --- |
| Throughput | ~4 s/study (5-fold ensemble, DICOM decode included) |
| ~1,300 test studies | **~1.4 h** |
| Kaggle limit | 9 h |

Most of that time is DICOM decode, not GPU. Decoding runs in DataLoader workers so it
overlaps inference.

## Why this is tested, not just compiled

A notebook that has only ever been compiled is an untested notebook, and Kaggle gives no
feedback beyond a failed run. `kaggle_inference.py` therefore takes its paths from
`RSNA_COMP` / `RSNA_WEIGHTS` / `RSNA_OUT`, so the exact code path that will run on Kaggle
can be executed locally:

```bash
RSNA_COMP=F:/rsna/rsna-knee-abnormality-detection \
RSNA_WEIGHTS=F:/rsna/work/runs/run1 \
RSNA_OUT=F:/rsna/work/kaggle_submission.csv \
python kaggle/kaggle_inference.py
```

This run reproduces `scripts/predict.py` to **4.6e-08** — float noise. That equivalence is
worth more than it looks: `predict.py` reads the preprocessed cache, while this path
decodes raw DICOM, orders slices, samples and normalises from scratch. Agreement to float
precision means the inference-time preprocessing is identical to what the model was
trained on. A silent mismatch there — a different slice order, a different normalisation —
is the classic way a model that validates well scores near chance on the leaderboard.

## Robustness

- Studies whose series all fail to decode are filled with the training prior rather than
  a constant. Under macro AUC a constant is rank-neutral, but the prior at least orders
  such studies sensibly against one another.
- Every study in `test.csv` appears in the output, decoded or not.
- Series selection mirrors training exactly: longest series wins each of the six
  plane × fluid-sensitivity slots.
