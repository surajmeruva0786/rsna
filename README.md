# RSNA Knee Abnormality Detection

Multi-label classification of knee MRI studies across 12 abnormality findings.

| | |
| --- | --- |
| Task | Multi-label binary classification, 12 targets per study |
| Train | 4,407 studies / 24,371 series |
| Test | 3 studies / 15 series (public split) |
| Modality | MRI, DICOM |
| Data on disk | 570 GB extracted, 819,640 files |

Competition brief, data description, and rules are in [`overview.txt`](overview.txt),
[`data.txt`](data.txt), and [`rules.txt`](rules.txt).

## Repository layout

```
.
├── DATASET.md              # dataset state, layout, verification results
├── LABELS.md               # report -> label weak supervision, and its validation
├── PREPROCESSING.md        # DICOM -> uint8 cache: slots, slice order, normalisation
├── MODEL.md                # architecture, augmentation constraints, results
├── extract_report.json     # machine-readable audit output
├── scripts/
│   ├── extract.py          # resumable, idempotent audit + extraction
│   ├── crc_spotcheck.py    # CRC-32 integrity sampling
│   ├── report_labeler.py   # multilingual report -> 12 targets
│   ├── validate_labeler.py # scores the labeller against the 58 gold studies
│   ├── make_labels.py      # writes work/labels.csv (gold where present, weak elsewhere)
│   ├── preprocess.py       # DICOM -> fixed-shape uint8 memmap cache
│   ├── qc_cache.py         # visual QC montage of the cache
│   ├── model.py            # slot encoder + masked attention pooling
│   ├── train.py            # cross-validated training, CUDA required
│   └── predict.py          # fold ensemble -> submission.csv
├── kaggle/
│   ├── kaggle_inference.py # self-contained offline inference
│   └── build_notebook.py   # wraps the above into submission.ipynb
├── overview.txt            # competition overview
├── data.txt                # data description
└── rules.txt               # competition rules
```

The data itself is **not** tracked — see [`.gitignore`](.gitignore). The extracted
tree lives beside this repo at `rsna-knee-abnormality-detection/` and is ignored.

## Prediction targets

`ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`,
`PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`

Only **58 of the 4,407** training studies carry these labels. The other 4,349 carry only
a free-text `Report`, in at least nine languages (English, Spanish, Turkish, Greek,
German, Dutch, French, Portuguese, Italian). Turning those reports into labels is the
prerequisite for using the training set at all — see [`LABELS.md`](LABELS.md). Reports
exist only at training time; the test set is DICOMs and study IDs.

> **Parsing note:** the reports contain embedded newlines. `train.csv` must be read
> with a real CSV parser — a line count returns 58,555 against a true 4,407 rows, and
> any line-oriented processing will silently corrupt the data.

## Data status

**Extraction is complete and verified** (2026-08-11). All 819,640 archive entries are
present at their exact recorded sizes, accounting for all 569,763,893,144 bytes, with
a clean CRC-32 sample over 405 files. Full detail, method, and the one outstanding
disk-space anomaly are in [`DATASET.md`](DATASET.md).

## Working with the data

Verify the extracted tree at any time (read-only):

```bash
python scripts/extract.py --audit-only
```

Extract or repair — the same command, idempotent; it extracts only what is missing or
size-mismatched, so an interrupted run resumes rather than restarting:

```bash
python scripts/extract.py
```

Sample content integrity against the archive's stored checksums:

```bash
python scripts/crc_spotcheck.py
```

`extract.py` exits `0` when the destination matches the archive and `1` when work
remains, so it can gate a pipeline.

## Building the labels

```bash
python scripts/validate_labeler.py   # score the labeller against the 58 gold studies
python scripts/make_labels.py        # write work/labels.csv for all 4,407
```

Current labeller agreement with gold: **0.756 macro AUC**. Method, language coverage and
the reason that number cannot reach 1.0 are in [`LABELS.md`](LABELS.md).

## Building the image cache

Training never reads the DICOM tree. One pass reduces it to a fixed-shape `uint8` memmap
(~21 GiB, ~27× smaller), keyed by six plane × fluid-sensitivity slots per study:

```bash
export RSNA_CACHE=C:/rsna_cache          # keep the cache off the HDD
python scripts/preprocess.py --split train --workers 6      # ~2.3 h
python scripts/preprocess.py --split test  --workers 6
python scripts/qc_cache.py --study 0 --out work/qc0.png     # look at it
```

Slot layout, slice ordering and per-slice normalisation are in
[`PREPROCESSING.md`](PREPROCESSING.md).

## Training and submission

```bash
python scripts/train.py --folds 5 --only-fold 0 --epochs 10 --batch 3 --accum 5 --out run1
python scripts/predict.py --run run1 --split test --out submission.csv
python kaggle/build_notebook.py     # -> kaggle/submission.ipynb
```

CUDA is required and asserted — no silent CPU fallback. Architecture, the reason
horizontal flip is excluded, and results are in [`MODEL.md`](MODEL.md).

> **Submission note:** this is a code competition. Kaggle scores a *notebook* run against
> ~1,300 hidden test studies; the local `submission.csv` only validates the format against
> the three public example studies. `kaggle/submission.ipynb` is the artefact that scores.

## Getting the data

The archive is not redistributable through this repo, and the local copy has been
deleted now that extraction is verified. To obtain it again:

```bash
kaggle competitions download -c rsna-knee-abnormality-detection
python scripts/extract.py --zip rsna-knee-abnormality-detection.zip \
                          --dest rsna-knee-abnormality-detection/
```

Note the archive has no top-level folder — its entries root directly at
`train_series/`, `test_series/`, and the five CSVs — so it must be extracted **into**
a named directory rather than into the current one.
