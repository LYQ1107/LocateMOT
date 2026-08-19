# Stage L11 — RMOT Cross-Domain Results (Refer-KITTI/V2 repair)

Date: 2026-08-20

## Same shared checkpoint (L9-ovmot), official TempRMOT evaluator

Refer-KITTI-V2, 4 official eval sequences (0005,0011,0013,0019), 862
queries:

| metric | before (L10: DLA top-50) | after (L11: whitelist+NMS+top-30+CLIP top-12) | delta |
| --- | ---: | ---: | ---: |
| candidates / frame | ~50 | ~15.3 (pkl); 12 kept by CLIP | -76% |
| HOTA | 3.74 | 6.28 | +2.54 |
| DetA | 0.93 | 3.09 | +2.16 |
| AssA | 16.72 | 13.80 | -2.92 |
| MOTA | -4153 | -766 | +3387 |
| IDF1 | 0.97 | 3.39 | +2.42 |
| DetPr (official) | ~1% | 3.34% | ~3.3x |

Reading:

- The front-end repair is real: DetA is no longer near zero, MOTA is no
  longer a -4000-level disaster, and HOTA improved ~68% with the SAME
  shared UIDM checkpoint.  Query-conditioned CLIP top-12 was calibrated
  on official train sequences only (no eval GT used).
- AssA dropped 16.72 -> 13.80: the CLIP top-12 filter trades recall for
  precision (train-calibrated target recall 65.1%); some correct target
  candidates are filtered, fragmenting identity chains.  A higher top-k
  (e.g., 15-20) would recover recall at lower precision (train
  calibration: 9.5% prec / 71.8% rec at top-15; 8.7% / 78.6% at top-20)
  and is the natural second correction if AssA is prioritized.
- The final Stage L11-L12 report will repeat this table with the
  post-L11 shared checkpoint.

## Refer-Dance (unchanged protocol, L9-ovmot)

| metric | value |
| --- | ---: |
| HOTA | 36.79 |
| DetA | 45.58 |
| AssA | 29.86 |
| MOTA | 29.38 |
| IDF1 | 36.56 |

(Will be refreshed with the final L11 shared checkpoint.)
