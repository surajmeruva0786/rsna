# Dataset — RSNA Knee Abnormality Detection

Status of the local competition data on this machine. **None of the data itself is
tracked in git** (see [`.gitignore`](.gitignore)) — this file records where it lives
and what state it is in.

## Source archive

| Property | Value |
| --- | --- |
| File | `rsna-knee-abnormality-detection.zip` (repo root, untracked) |
| Compressed size | 265,018,885,676 bytes (~246.8 GiB / 265 GB) |
| Uncompressed size | 569,763,893,144 bytes (~530.6 GiB / 570 GB) |
| Entries | 819,640 files (0 directory entries) |
| Format | ZIP64 |
| Downloaded | 2026-08-06 |

The archive has **no top-level folder** — entries are rooted at `train_series/…`,
`test_series/…`, and the five CSVs. It was extracted into the
`rsna-knee-abnormality-detection/` subdirectory.

### Entry breakdown

| Prefix | Entries |
| --- | --- |
| `train_series/` | 819,078 |
| `test_series/` | 557 |
| `train.csv`, `train_series.csv`, `test.csv`, `test_series.csv`, `sample_submission.csv` | 5 |

## Extraction target

```
F:\rsna\rsna-knee-abnormality-detection\
├── train.csv                 # 4,407 studies × 12 label columns + free-text Report
├── train_series.csv          # 24,371 series (Fluid_Sensitive, Fat_Suppression, Anatomical_Plane)
├── test.csv                  # StudyInstanceUID only
├── test_series.csv           # 15 series
├── sample_submission.csv     # 12 probability columns, all 0.5
├── train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
└── test_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
```

### Label columns (12 binary targets)

`ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`,
`PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`

`train.csv` also carries a free-text `Report` field (Spanish radiology reports).
Note that reports contain embedded newlines, so `train.csv` must be parsed with a
proper CSV reader — a naive line count returns 58,555 rather than the true 4,407 rows.

## Extraction status — pre-verification (2026-08-11)

The extraction begun on 2026-08-06 stopped partway on 2026-08-07 and was never
resumed. Directory-level counts matched expectations:

| Check | Extracted | Expected |
| --- | --- | --- |
| `train_series/` study dirs | 4,407 | 4,407 |
| `test_series/` study dirs | 3 | 3 |
| CSVs present | 5 / 5 | 5 |

Matching *study* counts does not prove completeness — a study directory exists as
soon as its first DICOM lands, so an interrupted run leaves full-looking directories
with missing slices inside. A file-level audit against the ZIP central directory is
therefore required, and is in progress; results are recorded below once it finishes.

## Verification method

`scripts/verify_extraction.py` walks the ZIP64 central directory and, for each of the
819,640 entries, stats the corresponding path on disk and compares the byte size
against the archive's recorded uncompressed size. An entry counts as extracted only
if it exists **and** its size matches exactly — this catches the truncated final file
an interrupted extraction leaves behind. The script writes:

- `verify_report.json` — summary counts
- `todo_entries.txt` — the exact entries needing (re-)extraction, for a resume pass

Extraction is treated as complete only when both `missing_count` and
`size_mismatch_count` are zero.

## Disk budget

The drive held 762 GB free with the partial extraction in place. A full extraction
needs ~570 GB total; deleting the 265 GB archive afterwards recovers that space.
The archive is only safe to delete once verification reports zero missing and zero
mismatched entries.

## Reproducing the download

The data is not redistributable through this repo. Fetch it from Kaggle:

```bash
kaggle competitions download -c rsna-knee-abnormality-detection
unzip rsna-knee-abnormality-detection.zip -d rsna-knee-abnormality-detection/
```
