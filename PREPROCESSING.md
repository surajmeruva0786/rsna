# Preprocessing: 570 GB of DICOM → a 21 GiB cache

## Why a cache at all

The raw tree is 819,640 DICOM files across 24,371 series, sitting on a **7200rpm SATA
HDD** (`ST2000NM000A`). A cold read costs ~32 ms per file — seek-dominated, not
bandwidth-dominated. Streaming that per epoch would mean every epoch pays the full decode
and seek cost again, and the GPU would idle through all of it.

So: decode once into a fixed-shape `uint8` memmap, then never touch the DICOM tree again
during training.

```
work/cache/train_224.u8    (4407, 6, 16, 224, 224) uint8   ≈ 21.2 GiB
work/cache/train_index.csv StudyInstanceUID order + per-slot present flag
```

The cache location is `RSNA_CACHE`. It is deliberately **not** on `F:` — it lives on the
NVMe volume, because training does ~5 MB of random reads per study per step and that
choice sets the epoch time. `F:` holds the DICOMs; `C:` holds the cache.

## The six slots

A knee study is a bag of series, not one image. Each study reduces to six slots — the
three anatomical planes crossed with the fluid-sensitive flag:

| slot | plane | fluid-sensitive | coverage (train) |
| --- | --- | --- | --- |
| 0 | Axial | no | rare (~1,179 series exist dataset-wide) |
| 1 | Coronal | no | common |
| 2 | Sagittal | no | common |
| 3 | Axial | yes | common |
| 4 | Coronal | yes | common |
| 5 | Sagittal | yes | common |

`Fluid_Sensitive` and `Fat_Suppression` are **perfectly collinear** across all 24,371
series — every fluid-sensitive series is fat-suppressed and vice versa — so one flag
carries both and the cross product is 6, not 12.

Where a study has several series in one slot, the one with the most slices wins. Slots
with no series are zero-filled and flagged absent in the index, and the model masks them
out rather than pooling over a zero vector.

## Slice ordering

Slices are ordered by **geometric position**, not `InstanceNumber`:

```python
normal = cross(ImageOrientationPatient[:3], ImageOrientationPatient[3:])
key    = dot(ImagePositionPatient, normal)
```

`InstanceNumber` is unreliable here — interleaved acquisitions and multi-echo series
renumber arbitrarily, and a scrambled stack would destroy the through-plane continuity
that the 16-channel input exists to exploit. `SliceLocation` then `InstanceNumber` are
fallbacks for series whose orientation tags did not survive anonymisation.

This costs a header-only pass over every file in a series before any pixels are decoded.
That pass is the bulk of the runtime, and it is worth it.

## Sampling and normalisation

Series run 20–45 slices (median 30) with a tail to a few hundred. Sixteen evenly spaced
positions are taken, so a 40-slice and a 200-slice acquisition produce the same tensor
covering the same anatomy. Series yielding fewer than 16 usable slices are stretched by
repetition rather than padded with black frames, which the model would otherwise learn on.

Intensity is stretched **per slice** between its own 0.5/99.5 percentiles. MRI has no
absolute scale — no Hounsfield equivalent — and this dataset spans a wide mix of
scanners, field strengths and vendors, so cross-series comparability has to be
manufactured. Clipping the tails stops a single bright vessel or artefact from compressing
all the tissue contrast into a handful of levels. `MONOCHROME1` series are inverted.

## Visual QC

Size checks and exception-free runs prove nothing about content. An inverted photometric
interpretation, a percentile stretch that crushes contrast, or a slice order sorted by a
meaningless tag all complete silently and simply train a worse model.

```bash
python scripts/qc_cache.py --split train --study 0 --out work/qc0.png
```

Dumps a 6-slot × 4-slice montage with slot labels. Checked before committing to the full
run: planes match their labels, fat-suppressed slots show the expected bright fluid /
dark fat, sagittal slices advance lateral→medial and coronal anterior→posterior, and
absent slots are correctly flagged.

## Cost

| | |
| --- | --- |
| Benchmark | 60 studies / 1.9 min at 6 workers |
| Full train split | **121.2 min**, 4,407 studies (predicted ~140 min) |
| Test split | 0.3 min, 3 studies |
| Output | 21,228,060,672 bytes — exactly the predicted size |
| Compression vs. raw | ~27× |

Six workers rather than twelve: the machine is running two other training jobs.

### Realised slot coverage

| slot | plane | fluid-sens. | studies with this slot |
| --- | --- | --- | --- |
| 0 | Axial | no | 857 |
| 1 | Coronal | no | 3,406 |
| 2 | Sagittal | no | 4,266 |
| 3 | Axial | yes | **4,407** (every study) |
| 4 | Coronal | yes | 4,248 |
| 5 | Sagittal | yes | 4,150 |

21,334 slots filled across 4,407 studies — **mean 4.84 slots/study**, and **zero studies
with no usable series**. Coverage is uneven enough to justify the masked pooling: axial
non-fluid exists for only 19% of studies, so a model that assumed six slots would be
training on mostly-empty tensors for that channel.

### One bug worth recording

The first version of `build_jobs` counted files in **every** series directory in
`train_series.csv` regardless of `--limit`, so a 48-study debug run still did 24,371
directory listings on the HDD and appeared to hang. Filtering the series table to the
requested studies before touching the disk fixed it. On a seek-bound volume, *deciding*
what to read can cost more than reading it.
