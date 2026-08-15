# Pipeline status

Last updated: 2026-08-14, during fold-0 training.

## Where things stand

| Stage | State | Artefact |
| --- | --- | --- |
| Data extraction + verification | done (pre-existing) | 819,640 files, CRC-sampled |
| Report → weak labels | **done** | `work/labels.csv`, 0.756 macro AUC vs gold |
| DICOM → uint8 cache | **done** | 21.2 GiB, 4,407 studies, 121.2 min |
| Test split cache | **done** | 3 studies |
| Training, 5 folds × 10 epochs | **done** (~9 h) | `work/runs/run1/fold{0..4}.pt` |
| OOF evaluation | **done** | `scripts/evaluate.py` |
| `submission.csv` | **produced** from all 5 folds, format validated | `submission.csv` |
| Kaggle notebook | **built and executed end-to-end** | `kaggle/submission.ipynb` |

The notebook's code path was run locally against the real dataset and reproduces
`scripts/predict.py` to 4.6e-08 — float noise — despite decoding raw DICOM rather than
reading the cache. That equivalence is the check that matters: it proves inference-time
preprocessing matches training, which is where a model that validates well otherwise
quietly scores near chance. Measured ~4 s/study → ~1.4 h for ~1,300 studies, against a 9 h
limit.

**Remaining to actually submit:** upload `work/kaggle_dataset/` (85 MB, five folds) as a
Kaggle Dataset, attach it to the notebook, internet off, run all. See
[`kaggle/README.md`](kaggle/README.md).

## Results

Fold validation macro AUC: 0.7753 / 0.7829 / 0.7658 / 0.7899 / 0.7776 — mean 0.7783,
spread 0.024. OOF over all 4,407 studies: **0.7748** against the weak labels.

On the 58 gold studies — the honest estimate, since those labels were made by reading
images the way the test labels were:

| | macro AUC |
| --- | --- |
| **Image model (OOF)** | **0.6910**, 95% CI [0.6329, 0.7470] |
| Report labeller | 0.7558 (training-time only) |
| Chance | 0.5000 |

Well above chance, well below a winning score. The model beats its own text teacher on
diffuse findings (Effusion +0.078, Contusion +0.041) and loses on small localised ones
(Medial Meniscus −0.221, Fracture −0.169) — the predictable signature of 16 sampled
slices at 224×224 on a 4 GiB GPU. Full breakdown in [`MODEL.md`](MODEL.md).

Measured: ~10 min/epoch, ~1.7 h/fold, 1.97 GiB peak at batch 2 (2.51 GiB at batch 3, too
close to the ~3.3 GiB free alongside the two other GPU jobs).

All folds trace the same curve: ~0.62–0.66 after one epoch, ~0.75 by epoch 4–5, peak at
epoch 6–7, then mild overfitting. **Mean 0.7785, spread 0.024** across four completed
folds — consistent enough that the result is a property of the setup, not of one split.

The image model is clearly reading the pixels: chance is 0.5 and every fold cleared 0.62
within a single epoch.

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

**The pipeline is complete; the leaderboard claim is not supportable.** Measured
performance is **0.6910 macro AUC** on the gold studies, 95% CI [0.6329, 0.7470]. That is
decisively better than chance and is not a winning score. Training ran on a **Quadro
P1000, 4 GiB Pascal** shared with two other jobs — roughly 1/40th the throughput
competitive teams bring — and the per-target breakdown shows that constraint biting
exactly where expected, on the small localised findings.

**The labels are a noisy proxy.** Gold labels are image-derived and sometimes contradict
their own report — one gold study reads `Effusion=0` against a report stating
`Moderate joint effusion, distended suprapatellar bursa`. The image model trains against a
teacher that is itself 0.7558 against the annotation process that will score it.

**Nothing has been submitted.** The notebook is built and tested but the weights Dataset
has not been uploaded and no Kaggle run has been made, so there is no leaderboard
position — only the offline estimate above.

## If the work continues

In descending order of expected return:

1. **Resolution and slice count.** The four worst targets — Medial Meniscus, Fracture,
   MCL, ACL — are all small-lesion tasks, and the teacher scores 0.74–0.89 on them, so the
   information exists and the model simply cannot resolve it. This is the binding
   constraint and it is hardware, not design.
2. **A second backbone in the ensemble**, and longer training with stronger
   regularisation. Cheap, modest gains.
3. **Better weak labels.** Helps less than it looks: every target is capped by the
   teacher, but the targets the model fails are ones the teacher already handles well.

## Bugs worth remembering

**Completion signals must come from the process whose completion they report.** The
automation chain waited for an `EXIT=` sentinel appended by the shell that launched
preprocessing. That shell was killed; the sentinel never arrived; the chain waited
indefinitely while preprocessing had in fact completed cleanly two hours earlier. The
cache was intact and verified byte-for-byte, but ~2 h of GPU idle time was lost. Fixed in
`scripts/run_after_preprocess.sh`, which now waits on a line `preprocess.py` prints
itself.

**Long runs must not be children of a transient shell.** Shell-parented background jobs
were killed at turn boundaries — one training process died ~3 minutes in, before its
first epoch. Training now launches via `Start-Process` as a detached Windows process
(PID recorded in `C:\rsna_cache\train.pid`) and survives independently.

**A checkpoint on disk is not a finished model.** `train.py` writes a fold's checkpoint on
every validation improvement, so a fold still training already has a `fold{N}.pt` file.
`predict.py` globbed `fold*.pt`, so running it mid-training would have averaged a
half-trained model into the ensemble — silently, with no error and no obvious symptom in
the output. `--folds` now selects explicit indices and the ensemble prints each member's
recorded validation AUC before predicting.

**Bugs that hide behind an epoch boundary are expensive.** The validation loop unpacked
three values while the dataset had begun yielding four — adding the per-sample gold
weight widened the tuple, and the validation set has labels too. The crash landed at the
*end* of epoch 1, so a one-line error cost a full epoch to surface. `--subset N` now runs
the entire loop, checkpointing and both metric paths included, in under a minute.
