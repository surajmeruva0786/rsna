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
python scripts/train.py --folds 5 --only-fold 0 --epochs 10 --batch 3 --accum 5 --out run1
python scripts/predict.py --run run1 --split test --out submission.csv
```

Fold 0 is trained first to measure real epoch time on the shared GPU before committing to
a full 5-fold budget; its checkpoint alone is already enough to produce a submission.

## Results

_Pending — fold 0 is training. This section will record OOF macro AUC against the weak
labels, the separate figure against the 58 gold studies, and per-target AUCs._

The gold-subset number is the one to trust. The weak-label OOF figure measures agreement
with the report labeller, which is itself only ~0.756 against gold — a model that scored
1.0 there would have learned to imitate the labeller's mistakes.

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
