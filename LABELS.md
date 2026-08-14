# Weak labels from radiology reports

## The problem

`train.csv` has 4,407 studies. **58** carry the twelve per-condition labels. The other
**4,349** carry only a free-text radiology report.

```
>>> train[TARGETS].notna().all(axis=1).sum()
58
```

Training an image model on 58 studies across 12 sparse targets is hopeless — the rarest
target (MCL) has 9 positives among them. So the reports have to become labels, or 98.7%
of the training set is dead weight. That is the single highest-leverage step in this
competition, and it happens before any pixel is read.

Note the asymmetry: reports exist **only at training time**. `test.csv` is study IDs and
DICOMs, nothing else. The report labeller is a training-time teacher, never part of
inference.

## Language mix

Reports are multilingual and the mix is not documented anywhere in the competition
material. A first keyword pass classified 901 of 4,407 as "unknown"; inspecting the
labeller's misses showed those were largely **Greek (321)** and **Dutch (137)**, neither
of which was in the initial vocabulary. Adding them was worth ~0.04 macro AUC on its own.

| Language | Approx. reports | Sample finding phrase |
| --- | --- | --- |
| English | ~1,855 | `complete tear of the anterior cruciate ligament` |
| Spanish | ~679 | `rotura del cuerno posterior del menisco medial` |
| Turkish | ~547 | `ön çapraz bağda tam kat yırtık` |
| Greek | ~321 | `ρήξη στο οπίσθιο κέρας του μηνίσκου` |
| German | ~259 | `Innenmeniskusriss`, `Knochenmarködem` |
| Dutch | ~137 | `femoropatellair kraakbeenlijden` |
| French / Portuguese / Italian | ~166 | `lésion du ligament croisé antérieur` |

## How it works

`scripts/report_labeler.py`, ~350 lines of accent-folded regex. No model — with 58
labelled examples there is nothing to train a text classifier on, and rules stay
auditable against those 58.

**1. Normalise.** Lowercase, NFKD accent-strip, plus an explicit fold for characters
Unicode does not decompose: Turkish dotless `ı`, Greek final sigma `ς`, German `ß`. This
means one pattern matches `menisco`/`menisco`, `odem`/`ödem`, `ρηξη`/`ρήξη`. Greek
reports that spell mu as the micro sign `µ` are handled by NFKD for free.

**2. Segment.** Split on strong boundaries only — `.` `;` `\n` `>` `--` `•`.

Notably **`:` does not split, and neither do `and`, `with` or commas.** An earlier
version split on all of them and it was the single worst bug in the labeller: structured
reports write

```
MEDIAL COMPARTMENT:
Medial meniscus: Extensive complete tearing of the body, posterior horn, and posterior root
```

and splitting on `:` strands the anatomy in one fragment and the pathology in the next,
so nothing fires. Same for `Longitudinal vertical tear at the body and posterior horn of
the medial meniscus` — split on `and`, and the tear loses its anatomy. Fixing
segmentation alone took Medial Meniscus from 0.694 to 0.840 AUC.

**3. Match by proximity, not co-occurrence.** Ligament and meniscus targets require an
anatomy mention *and* a tear mention within 120 characters. Anatomy alone must not fire —
`ACL is intact` mentions the ACL and means the opposite.

**4. Negate over a tight span.** Negation is searched in a window from 45 chars before to
30 chars after the matched pair, not segment-wide. This is what lets

```
tear of the medial meniscus and the lateral meniscus is intact
```

assert Medial Meniscus and negate Lateral Meniscus from one sentence. The window is
two-sided rather than left-looking because Turkish, German and Spanish routinely
post-pose negation: `yırtık izlenmedi`, `Meniskusriss nicht nachweisbar`, `menisco
íntegro`.

**5. Score softly.** Output is a probability, not a bit:

| Evidence | Score |
| --- | --- |
| Asserted | 1.00 |
| Hedged (`suspect`, `V.a.`, `probable`) | 0.65 |
| Soft — correlate, not assertion | 0.45 |
| Never mentioned | 0.06 |
| Explicitly negated | 0.02 |

"Never mentioned" is deliberately *not* 0. A report that is silent on Baker's cyst is
weaker evidence of absence than one that says `no Baker's cyst`, and collapsing both to
zero throws away a real distinction the AUC metric can see.

The **soft tier** exists for two asymmetries found in the gold data. Synovitis is often
not named but reported through its companions — bursal distension, Hoffa fat-pad oedema,
plica irritation — and gold marks those studies positive. Conversely isolated
*subchondral* oedema is usually reactive marrow change beside an arthritic compartment
rather than a traumatic bruise, so it was demoted out of Contusion's assertive tier.

## Compartment-specific osteoarthritis

Three of the twelve targets are OA in a named compartment, so an OA term must be paired
with a compartment cue (`medial femoral condyle`, `retropatellar`, `εξω διαμερισμα`,
`femoropatellair`). Compartment-agnostic phrasings are expanded:

- `tricompartmental` → Medial + Lateral + PF
- `bicompartmental` → Medial + PF
- bare `gonarthrosis` → Medial 0.8, PF 0.5, Lateral 0.4 (medial-dominant in practice)

## Validation

`python scripts/validate_labeler.py`

| Target | pos | AUC | precision | recall |
| --- | --- | --- | --- | --- |
| ACL | 24 | 0.890 | 0.909 | 0.833 |
| MCL | 9 | 0.757 | 0.714 | 0.556 |
| Medial Meniscus | 26 | 0.840 | 0.850 | 0.654 |
| Lateral Meniscus | 23 | 0.749 | 0.867 | 0.565 |
| Medial OA | 15 | 0.728 | 0.875 | 0.467 |
| Lateral OA | 11 | 0.756 | 0.750 | 0.545 |
| PF OA | 21 | 0.761 | 0.765 | 0.619 |
| Effusion | 35 | 0.631 | 0.684 | 0.743 |
| Synovitis | 27 | 0.704 | 0.750 | 0.444 |
| Baker's | 12 | 0.845 | 0.643 | 0.750 |
| Contusion | 19 | 0.667 | 0.500 | 0.737 |
| Fracture | 18 | 0.742 | 0.625 | 0.556 |
| **macro** | | **0.756** | | |

Progression: 0.691 initial → 0.713 (proximity windows) → 0.750 (Greek/Dutch/Turkish
vocabulary) → 0.756 (soft tier).

### The ceiling is not 1.0

The gold labels are **image-derived** — an annotator read the MRI, not the report — and
they sometimes contradict the report outright. One gold study is labelled `Effusion=0`
while its report says `Moderate joint effusion, distended suprapatellar bursa`. Another
is `Synovitis=0` against `mild hypertrophy of the synovium, indicative of synovitis`.

So a large share of the residual "false positives" are not rule bugs, and chasing them
would fit noise across 58 samples. Tuning was stopped once the clear vocabulary and
segmentation gaps were closed.

This also sets expectations downstream: the image model trains against labels that are a
noisy proxy for the annotation process that will score it.

## Output

`python scripts/make_labels.py` → `work/labels.csv`, one row per study, twelve soft
targets, plus `is_gold`. Gold labels override the labeller where they exist, and training
up-weights those 58 rows (`--gold-weight`, default 5).

| Target | mean score | asserted (≥0.9) | soft | negated (<0.05) |
| --- | --- | --- | --- | --- |
| ACL | 0.174 | 525 | 22 | 112 |
| MCL | 0.101 | 192 | 5 | 96 |
| Medial Meniscus | 0.298 | 1124 | 20 | 455 |
| Lateral Meniscus | 0.161 | 480 | 25 | 503 |
| Medial OA | 0.174 | 533 | 18 | 266 |
| Lateral OA | 0.132 | 343 | 20 | 324 |
| PF OA | 0.227 | 787 | 12 | 241 |
| Effusion | 0.446 | 1859 | 0 | 1191 |
| Synovitis | 0.213 | 516 | 492 | 45 |
| Baker's | 0.258 | 962 | 5 | 859 |
| Contusion | 0.261 | 922 | 98 | 521 |
| Fracture | 0.139 | 390 | 10 | 588 |

Prevalences are clinically plausible — effusion most common at ~42%, MCL rarest at ~4%.
