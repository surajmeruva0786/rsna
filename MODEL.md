# Model and training

## What the model has to do

Predict twelve independent probabilities per **study**, scored as macro-averaged AUC.
A study is not an image — it is 3–14 MRI series (median 5.5) in different planes and
contrasts, each showing a different subset of the findings. Sagittal fluid-sensitive
carries the menisci and cruciates; axial carries the patellofemoral joint and Baker's
cyst; coronal carries the collaterals and the tibiofemoral compartments.

So the architecture question is: how do you turn a variable-size bag of 3D volumes into
twelve numbers, on a **4 GB Pascal GPU** shared with two other training jobs?

## Architecture

```
study ─┬─ slot 0 (Ax)      ─┐
       ├─ slot 1 (Cor)      │  16-channel image → backbone → 1280-d
       ├─ slot 2 (Sag)      ├─ + learned slot embedding
       ├─ slot 3 (Ax-FS)    │
       ├─ slot 4 (Cor-FS)   │
       └─ slot 5 (Sag-FS)  ─┘
                            │
                   masked gated attention pool
                            │
                    dropout → linear → 12 logits
```

`efficientnet_b0`, `in_chans=16`, **4.36 M parameters**.

### Slices as channels, not as a batch

Each slot enters the backbone as one **16-channel image** rather than 16 separate
forward passes. This is the decisive choice for this hardware: it cuts backbone
evaluations per study from ~96 to ≤6, roughly a 16× reduction in compute and activation
memory, while the first convolution still mixes across the slice axis — which is exactly
where the through-plane continuity of a tear lives. A meniscal tear that appears on three
consecutive sagittal slices is visible to the stem; it would not be to a per-slice model
without a separate temporal aggregator.

Measured: **1.07 GiB peak** for a full forward+backward at batch 2, against 3.3 GiB free.

### Masked gated attention pooling

Mean pooling would dilute the one informative plane with up to five uninformative ones.
Max pooling is unstable under weak labels. Gated attention lets the network learn which
plane to trust.

The mask matters as much as the attention. Studies have different slot coverage — axial
non-fluid exists for only a small minority — and an unmasked pool would let absent slots
contribute a zero vector, systematically dragging predictions toward zero for studies
with fewer series. Absent slots are set to `-inf` before the softmax so they contribute
nothing at all. A study with *no* usable slots would softmax over all `-inf` and produce
`NaN`, so that case is caught and falls back to uniform weights.

A learned per-slot embedding is added before pooling. Without it the attention head
cannot distinguish a sagittal feature vector from an axial one, and "which plane should
I trust for this finding" is precisely the question it needs to answer.

## Augmentation: no horizontal flip

This is a correctness constraint, not a tuning choice.

Six of the twelve targets are laterality-specific — Medial/Lateral Meniscus, Medial/
Lateral OA, MCL. **Mirroring a knee swaps medial for lateral.** A horizontal flip would
silently invert the label of every flipped sample for half the target set. It is the
single most tempting and most damaging augmentation available here, and it is excluded.

What is used instead:

- small affine (±12°, ±12% scale, ±8% shift), applied identically across all 16 slices
  of a slot — knees are consistently positioned, so large warps move the model
  off-distribution rather than regularise it;
- brightness/contrast jitter;
- random **slot dropout** (p=0.2) — test studies vary in coverage, so the attention head
  must not become dependent on any one plane always being present.

## Labels and loss

Targets are the soft scores from [`LABELS.md`](LABELS.md), not hard bits — BCE against a
0.45 soft-tier score expresses "probably, weakly" better than rounding to 0 or 1 does.

The 58 gold studies are up-weighted (`--gold-weight`, default 5). They are the only
labels not filtered through the report labeller, and the only ones produced the same way
the test labels were.

Folds are stratified on **positive-finding count**, so rare multi-finding studies do not
cluster into one fold — a plain `KFold` over 12 sparse labels allows exactly that, and it
would make fold AUCs incomparable.

## Hardware constraints

| | |
| --- | --- |
| GPU | Quadro P1000, 4 GiB, sm_61 (Pascal) |
| Free VRAM | ~3.3 GiB (two other training jobs running) |
| Batch | 3 studies × ≤6 slots, gradient accumulation 5 (effective 15) |
| AMP | off by default — Pascal runs fp16 at 1/64 FP32 rate, so autocast costs speed to save memory that is not short |

CUDA is **asserted**, not silently fallen back from. `train.py` and `predict.py` exit with
an explanatory error if CUDA is unavailable, because a CPU run would take days and
silently produce the same artefacts.

## Running it

```bash
export RSNA_CACHE=C:/rsna_cache
python scripts/train.py --folds 5 --epochs 10 --batch 2 --accum 8 --out run1
python scripts/predict.py --run run1 --split test --out submission.csv
```

Folds run sequentially and each writes its checkpoint on completion, so fold 0 alone
(~2 h) is already enough to produce a submission and a crash at fold 3 still leaves three
usable models.

### Measured cost

| | |
| --- | --- |
| Peak VRAM | 2.51 GiB at batch 3; batch 2 used instead for headroom |
| Epoch | ~10 min (3,525 train / 882 val) |
| Fold | ~2 h at 10 epochs |
| Full 5-fold | ~10 h |

### Launch it detached

Long runs **must not** be children of a transient shell. Use a detached process:

```powershell
Start-Process python -ArgumentList "scripts/train.py",... -WindowStyle Hidden -PassThru
```

Two runs were lost to this: a shell-parented training process was killed ~3 minutes in,
and the preprocessing wrapper was killed at the moment it finished, which cost the
completion signal a dependent job was waiting on.

### Debugging cheaply

`--subset N` trains on N studies, always including the 58 gold ones so the gold-metric
path is covered. The full loop — checkpointing, OOF assembly, both metrics — runs in
under a minute.

This exists because of a real cost. The validation loop unpacked three values while the
dataset had started yielding four (adding the per-sample gold weight widened the tuple,
and the validation set has labels too). The crash landed at the *end* of epoch 1, so a
one-line bug cost a full epoch to surface. Bugs that hide behind an epoch boundary are
worth a flag that removes the epoch boundary.

## Results

### Fold 0 (complete)

| epoch | loss | val macro AUC |
| --- | --- | --- |
| 1 | 0.5513 | 0.6564 |
| 2 | 0.4840 | 0.6973 |
| 3 | 0.4678 | 0.7505 |
| 4 | 0.4512 | 0.7666 |
| 5 | 0.4333 | 0.7690 |
| 6 | 0.4161 | **0.7753** |
| 7 | 0.3943 | 0.7751 |
| 8 | 0.3762 | 0.7639 |
| 9 | 0.3590 | 0.7667 |
| 10 | 0.3529 | 0.7702 |

Training loss falls monotonically while validation AUC peaks at epoch 6 and drifts down
after — textbook mild overfitting, and confirmation that 10 epochs is roughly the right
budget rather than a number that needed guessing. The saved checkpoint is epoch 6's.

Folds 1–4 are still running.

### Reading these numbers honestly

**0.7753 is agreement with the report labeller on held-out studies, not accuracy.** The
labeller is itself only ~0.756 against gold, so a model scoring 1.0 here would have
learned to reproduce the labeller's mistakes perfectly. The figure confirms the image
model is extracting real signal from pixels — chance is 0.5, and it reached 0.65 in one
epoch — but it is not the competition metric.

The gold-subset OOF figure, computed over the 58 hand-annotated studies once all folds
finish, is the meaningful one, because those labels were produced the same way the test
labels were.

## Submission

Kaggle scores **notebooks**, not files. A local `submission.csv` is a format check against
three public example studies and carries no leaderboard meaning. The artefact that scores
is `kaggle/submission.ipynb`, generated from `kaggle/kaggle_inference.py`:

```bash
python kaggle/build_notebook.py
```

It runs offline (`pretrained=False`, all weights from an attached Kaggle Dataset),
decodes test DICOMs in DataLoader workers to overlap I/O with the GPU, and falls back to
the training prior for any study whose series all fail to decode.
