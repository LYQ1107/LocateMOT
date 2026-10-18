# G2 data leakage and label audit

## 2026-10-19 audit result

The historical G2 formal corpus is preserved, but it is not an independent
MOT20-05 experiment. Its manifest has 199,053 rows (SHA256
`b8df7365a1b1cda47489f8044d86c030be9ab17c48724b5b5620befb5c9eb76e`) and
3,317 rows from `MOT20-05`. The corresponding 1,200-step checkpoint and
merged `GroundAnything-DLM-G2-PL` bundle remain valid engineering artifacts;
they must not be reported as a clean held-out result.

The read-only audit also found the following protocol problems in the old
builder:

- MOT rows did not consistently enforce `mark==1` and `class==1` before
  treating a box as a person target.
- TAO-Amodal emitted one category per image, included Novel annotations in the
  old count, and allowed sparse images to look like negative supervision.
- Refer-KITTI-V2 kept only the first valid frame of an expression. The legal
  fit split contains 4,333 expressions and 379,341 labeled expression frames.
- RMOT26 rows used the original natural-language question in the source, but
  the old corpus exposed placeholder-style supervision in its audit scope.
- BURST has no legal TRAIN image root in this checkout and remains an explicit
  source gap. No test image was substituted.

Path-level media overlap was not found by this audit. Content-hash overlap is
not claimed without a complete decoded-media hash pass. Refer-KITTI legal
splits are defined by `locatemot/unified/data/legal_scope.py`; official-reserved
videos remain outside the clean TRAIN manifest.

## Clean manifest

`prepare_g2_clean_corpus.py` writes a new immutable corpus rather than
modifying the old rows or checkpoint:

`/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/`

The clean rows contain 583,251 unique records (SHA256
`f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71`; the
manifest records the authoritative value), with 2,249 rejected records. The
source row counts are:

| Source | Clean rows |
| --- | ---: |
| Refer-KITTI-V2 | 379,341 |
| AerialMind | 69,358 |
| DanceTrack | 41,796 |
| TAO-Amodal Base | 30,781 |
| SportsMOT | 28,574 |
| MOT17 | 15,948 |
| AnimalTrack | 11,543 |
| MOT20 (`MOT20-01/02/03`) | 5,616 |
| RMOT26 | 294 |

The clean policy is `mark==1, class==1` for MOT-like sources; TAO Base only
(`frequency != r`), one row per reliable annotated category, no sparse negative;
all legal Refer-KITTI labeled frames; and exact normalized RMOT questions with
all initial target boxes. MOT20-05 is absent from the clean rows and from the
RMOT copy filter. The clean manifest records the base GroundAnything SHA,
reject SHA, source/video/image coverage, and the explicit BURST gap.

## Evaluation interpretation

The prior short-control attempt stopped before G2 comparison because the
registered G0 service timed out at frame 302 (the first request, 600-second
client timeout). It produced no quality metric. A separate BS8 training probe
was rejected by a real vision OOM (39--42 GiB additional allocation with less
than 8 GiB free); the clean course therefore uses BS2 plus accumulation.
