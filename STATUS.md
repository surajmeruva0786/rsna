# Pipeline status

Last updated: 2026-08-14, during fold-0 training.

## Where things stand

| Stage | State | Artefact |
| --- | --- | --- |
| Data extraction + verification | done (pre-existing) | 819,640 files, CRC-sampled |
| Report → weak labels | **done** | `work/labels.csv`, 0.756 macro AUC vs gold |
| DICOM → uint8 cache | **done** | 21.2 GiB, 4,407 studies, 121.2 min |
| Test split cache | **done** | 3 studies |
| Fold-0 training | **running** | `work/runs/run1/fold0.pt` |
| Remaining folds | not started | budget decision pending epoch timing |
| `submission.csv` | not yet produced | needs a trained fold |
| Kaggle notebook | **built**, untested end-to-end | `kaggle/submission.ipynb` |

## Answering the original question

**No training had been run and no submission file existed** when this session started.
The repository contained data-acquisition work only — `extract.py`, `crc_spotcheck.py`,
and the extraction audit documents across seven commits. No model, no training code, no
predictions.

## What was built

Four stages, each committed and documented separately:

1. **[`LABELS.md`](LABELS.md)** — the competition's real bottleneck. Only 58 of 4,407
   training studies carry per-condition labels; the rest carry free-text reports in nine
   languages. A negation-aware multilingual labeller turns those into training targets at
   0.756 macro AUC against the 58 gold studies.
2. **[`PREPROCESSING.md`](PREPROCESSING.md)** — 570 GB of DICOM reduced to a 21.2 GiB
   fixed-shape memmap on the NVMe volume, six plane × fluid-sensitivity slots per study,
   geometrically ordered slices, per-slice percentile normalisation. Visually QC'd.
3. **[`MODEL.md`](MODEL.md)** — slot encoder feeding 16 slices as channels, masked gated
   attention pooling to study level, 12-way BCE. 4.36 M parameters, 1.07 GiB peak.
4. **`kaggle/`** — the artefact that actually scores: an offline notebook generated from
   a reviewable Python source.

## Open items and honest caveats

**The leaderboard claim cannot be made.** This is a code competition: Kaggle scores a
notebook run against ~1,300 hidden studies, so no locally produced CSV can be submitted.
Separately, training runs on a **Quadro P1000, 4 GiB Pascal**, shared with two other jobs
— roughly 1/40th the throughput available to competitive teams. The pipeline is complete
and sound; a top placement is not something this hardware can be promised to deliver.

**The labels are a noisy proxy.** Gold labels are image-derived and sometimes contradict
their own report — one gold study reads `Effusion=0` against a report stating
`Moderate joint effusion, distended suprapatellar bursa`. The image model therefore trains
against a teacher that is itself ~0.756 against the annotation process that will score it.
The OOF figure to trust is the gold-subset one, not the weak-label one.

**Untested end-to-end:** the Kaggle notebook compiles and its logic mirrors the local
path, but it has not yet been run against real Kaggle paths. That needs a trained
checkpoint uploaded as a Dataset.

## A bug worth remembering

The automation chain waited for an `EXIT=` sentinel appended by the shell that launched
preprocessing. That shell was killed; the sentinel never arrived; the chain waited
indefinitely while preprocessing had in fact completed cleanly two hours earlier. Nothing
was lost — the cache was intact and verified byte-for-byte — but ~2 h of GPU time was.

Lesson: a completion signal should come from the process whose completion it reports.
Fixed in `scripts/run_after_preprocess.sh`.
