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
├── STATUS.md               # what is done, what is running, open caveats
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
│   ├── predict.py          # fold ensemble -> submission.csv
│   ├── evaluate.py         # OOF metrics incl. bootstrapped gold-subset CI
│   └── run_after_preprocess.sh  # chains preprocessing -> training
├── kaggle/
│   ├── README.md           # how to actually submit
│   ├── kaggle_inference.py # self-contained offline inference
│   ├── build_notebook.py   # wraps the above into submission.ipynb
│   └── submission.ipynb    # generated; the artefact Kaggle scores
├── submission.csv          # format check only (3 public example studies)
└── .mcp.json.example       # Kaggle MCP config (copy to .mcp.json; never commit a token)
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
python scripts/train.py --folds 5 --epochs 10 --batch 2 --accum 8 --out run1   # ~9 h
python scripts/evaluate.py --run run1                       # OOF + gold-subset CI
python scripts/predict.py --run run1 --split test --folds 0,1,2,3,4 --out submission.csv
python kaggle/build_notebook.py                             # -> kaggle/submission.ipynb
```

CUDA is required and asserted — no silent CPU fallback. Architecture, the reason
horizontal flip is excluded, and results are in [`MODEL.md`](MODEL.md).

Two operational notes that cost real time to learn:

- **Launch long runs detached** (`Start-Process` on Windows). Shell-parented background
  jobs get killed with their parent; one training run died three minutes in.
- **Pass `--folds` explicitly to `predict.py`.** Checkpoints are written on every
  validation improvement, so a still-training fold already has a `fold{N}.pt` on disk and
  the default glob would silently average a half-trained model into the ensemble.

> **Submission note:** this is a code competition. Kaggle scores a *notebook* run against
> ~1,300 hidden test studies; the local `submission.csv` only validates the format against
> the three public example studies. `kaggle/submission.ipynb` is the artefact that scores —
> see [`kaggle/README.md`](kaggle/README.md).

## Results

| | macro AUC |
| --- | --- |
| OOF, weak labels (4,407 studies) | 0.7748 |
| **OOF, 58 gold studies** | **0.6910** — 95% CI [0.6329, 0.7470] |
| Report labeller, same 58 | 0.7558 *(training-time only)* |
| Chance | 0.5000 |

Folds: 0.7753 / 0.7829 / 0.7658 / 0.7899 / 0.7776 (spread 0.024).

The gold figure is the honest estimate — those labels were made by reading images, as the
test labels were. The weak-label figure measures agreement with the report labeller, which
is itself only 0.7558 against gold. Full breakdown, including which targets the model beats
its own teacher on and why, is in [`MODEL.md`](MODEL.md); current state and remaining work
in [`STATUS.md`](STATUS.md).

**0.6910 is decisively above chance and is not a winning score**, and nothing has been
submitted yet, so there is no leaderboard position — only this offline estimate. The
binding constraint is resolution: the model beats its own text teacher on diffuse findings
and loses on small localised ones, which is what 16 slices at 224×224 on a 4 GiB card
predicts.

## Pipeline at a glance

| Stage | Input | Output | Cost |
| --- | --- | --- | --- |
| Extract + verify | 265 GB zip | 819,640 files | (pre-existing) |
| Report → weak labels | `train.csv` | `work/labels.csv` | seconds |
| DICOM → cache | 570 GB tree | 19.8 GiB memmap | 121 min |
| Train 5 folds | cache + labels | `fold{0..4}.pt` (85 MB) | ~9 h |
| Predict | cache + folds | `submission.csv` | seconds |
| Kaggle notebook | DICOM + folds | scored submission | ~1.4 h on Kaggle |

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
