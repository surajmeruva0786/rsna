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

## Extraction status: COMPLETE (verified 2026-08-11)

**The extraction is complete. No files were missing; nothing needed re-extracting.**

A file-level audit of all 819,640 archive entries against the extracted tree:

| Metric | Value |
| --- | --- |
| Entries in archive | 819,640 |
| Entries present on disk at exact size | **819,640** |
| Entries outstanding | **0** |
| Bytes accounted for | 569,763,893,144 / 569,763,893,144 (100%) |
| Audit duration | 1,302 s (~22 min) |

Recorded verbatim in [`extract_report.json`](extract_report.json).

### Why the audit was necessary

Directory-level counts alone (4,407 train studies, 3 test studies, 5/5 CSVs) looked
correct, but proved nothing: a study directory is created as soon as its *first*
DICOM lands, so an extraction interrupted midway leaves 4,407 complete-looking
directories with missing slices inside. The mtimes reinforced the suspicion — the
run started 2026-08-06 and the tree was last touched 2026-08-07, consistent with an
interrupted job. Only a per-entry comparison against the archive could distinguish
"finished" from "stopped near the end", and it returned finished.

### Integrity spot-check

Size equality proves nothing about *content*. `scripts/crc_spotcheck.py` samples
entries at random and compares each file's on-disk CRC-32 against the checksum stored
in the ZIP central directory:

| Metric | Value |
| --- | --- |
| Files checked | 405 (all 5 CSVs + 400 random DICOMs, seeded sample) |
| Bytes read | 0.27 GiB |
| CRC mismatches | **0** |

A full CRC pass would mean reading 570 GB and decompressing 265 GB; a seeded random
sample catches systematic corruption at a negligible fraction of that cost.

## Verification and extraction

`scripts/extract.py` handles both, in one pass over the archive:

1. **Audit** — walk the ZIP64 central directory and, for each of the 819,640 entries,
   stat the corresponding path on disk. An entry counts as already extracted only if
   it exists **and** its size matches the archive's recorded uncompressed size
   exactly. Size equality rather than mere existence is the test, because an
   interrupted extraction leaves a truncated final file that an existence check would
   accept.
2. **Extract** — write only the outstanding entries.

Audit and extraction are fused deliberately: extraction has to do the identical stat
walk anyway, and on this drive a walk over 819,640 files costs ~15 minutes, so
running a standalone verifier first would pay that cost twice for no new information.
Use `--audit-only` when a read-only check is what's wanted.

The script is **idempotent and resumable**. Re-running after an interruption costs
one stat walk plus only the remaining bytes, never a full re-extraction. Each file is
written to a `.part` sibling and atomically renamed into place only once fully
written, so an interruption of the script itself cannot leave a half-written file for
a later run to mistake for a good one.

Outputs `extract_report.json` (summary counts) and `todo_entries.txt` (outstanding
entries). Extraction is complete only when `entries_todo` is zero.

## Disk budget

| | |
| --- | --- |
| Free space (with archive present) | 762 GB |
| Full extraction requires | 570 GB |
| Worst-case headroom remaining | ~192 GB |

The full tree and the archive **do** coexist on the drive, so the archive never needed
to be removed to make room. With the audit reporting zero outstanding entries and a
clean CRC sample, the archive is now redundant and safe to delete; doing so recovers
265 GB. See "Archive disposition" below.

## Archive disposition

`rsna-knee-abnormality-detection.zip` was **deleted on 2026-08-11**, after the audit
reported zero outstanding entries and the CRC sample came back clean. The extracted
tree is now the only copy of the data on this machine.

### Open issue: 265 GB was not reclaimed

Deleting the archive did **not** return its space to the volume. Free space on `F:`
read 761.83 GiB both before and after the delete, and the file is genuinely unlinked
(absent from the directory, absent from the Recycle Bin, which holds only 4.6 MB).

Ruled out so far:

- **Recycle Bin** — 4.6 MB total; Git Bash `rm` unlinks rather than recycling.
- **A second copy elsewhere** — no RSNA archive under `F:\rsna` or `F:\Fdownloads`
  (the large zips there are unrelated `01_Clinical_Trial-*` files).
- **A process holding an open handle** — the one candidate exited with no change in
  free space.

Volume accounting supports this. `F:` is 1,863 GiB with 1,101.2 GiB used:

| Location | Size |
| --- | --- |
| `rsna/` extracted tree | ~531 GiB |
| `Fdownloads/` (unrelated `01_Clinical_Trial-*` archives) | ~130 GiB |
| `biohub/` | 82 GiB |
| `stressID/` + `stressIDdataset/` | 10.2 GiB |
| `System Volume Information` | unreadable (access denied) |
| **Measurable total** | **~753 GiB** |
| **Unaccounted** | **~348 GiB** |

The ~348 GiB gap comfortably contains the archive's 246.8 GiB, leaving ~100 GiB for
`System Volume Information` — consistent with the deleted file's blocks still being
allocated but no longer reachable through the filesystem.

The leading explanation is therefore a **Volume Shadow Copy / System Protection
snapshot** on `F:` still referencing those blocks, which keeps them allocated until
the snapshot is released. Confirming this needs an **elevated** shell:

```powershell
vssadmin list shadowstorage
vssadmin list shadows
```

If shadow storage on `F:` accounts for the missing 265 GB, reclaiming it means
deleting the relevant snapshots (or reducing `shadowstorage` maxsize) — a decision to
make deliberately, since it discards restore points. **This does not affect the
dataset**, which is fully extracted and verified; it only concerns free space.

## Reproducing the download

The data is not redistributable through this repo. Fetch it from Kaggle:

```bash
kaggle competitions download -c rsna-knee-abnormality-detection
unzip rsna-knee-abnormality-detection.zip -d rsna-knee-abnormality-detection/
```
