# Stage L11 — Refer-KITTI / Refer-KITTI-V2 Candidate-Front-End Repair

Date: 2026-08-19

## 1. Official detector / candidate protocol audit

### Refer-KITTI (v1) and RMOT (CVPR 2023)

- Official repo: https://github.com/wudongming97/RMOT (local mirror
  `LocateMOT_reference_repos/rmot_official`, commit
  `d4fedb35538e79a743ff78ff946abc6c84453cab`, MIT-style LICENSE
  present).
- The official RMOT detector is a custom end-to-end DETR (Deformable
  DETR with referring-query decoding).  Inference parameters read from
  `LocateMOT_reference_repos/temp_rmot/inference.py`:
  - `prob_threshold = 0.6`
  - `area_threshold = 100`
  - `filter_dt_by_ref_scores(..., 0.4)` (referring score threshold)
- Candidates are therefore **query-conditioned by construction**: the
  detector emits boxes only for objects matching the referring
  expression, then a second refer-score threshold selects the output.
  There is no separate 50-candidates/frame public-detector stream.

### Refer-KITTI-V2 / TempRMOT

- Official repo: https://github.com/zyn213/TempRMOT (local mirror
  `LocateMOT_reference_repos/temp_rmot`, commit
  `6a65640d849fdee4a32bb055945ee34c3b0edeb1`; no LICENSE file found,
  no code copied).
- Same end-to-end TransRMOT detector as RMOT; dataset split:
  - train: 17 sequences (0000,0001,0002,0003,0004,0006,0007,0008,0009,
    0010,0012,0014,0015,0016,0017,0018,0020)
  - eval: 4 held-out sequences (0005,0011,0013,0019) with the official
    861-expression seqmap.
- No separate public-detector candidate protocol exists; the official
  pipeline is detector + refer-threshold + TrackEval.

### iKUN (ECCV 2024)

- Official repo: `LocateMOT_reference_repos/iKUN` (commit
  `4db56bfaec703590e0fdfd1684d9769467a67e05`; LICENSE present).
- Uses a modified MOTR-style end-to-end tracker; candidates come from
  its own detector, not a public-detector stream.

### TransRMOT (ECCV 2024)

- Official repo: `LocateMOT_reference_repos/crmot` is the closest
  "causal/transformer" RMOT line; TransRMOT is the TempRMOT detector.
- No public candidate protocol.

### Conclusion for LocateMOT

There is no official "DLA + top-k" candidate protocol for RMOT; the
official methods are end-to-end and query-conditioned.  LocateMOT uses
a shared UIDM over a generic detector stream, so the closest legitimate
repair is to make the **candidate front-end query-conditioned and
precision-oriented** without touching GT identities:

1. LVIS category whitelist (KITTI-relevant classes, data-driven on
   official train sequences);
2. cross-category NMS (Detic emits near-duplicate boxes under different
   LVIS labels);
3. det-score threshold + top-K;
4. query-conditioned CLIP crop-sentence top-k / min-sim (calibrated on
   train sequences only);
5. shared-UIDM evaluation with the official TempRMOT evaluator.

## 2. L10 baseline (before repair)

Refer-KITTI-V2, L9-ovmot shared checkpoint, Detic-SwinB 50
candidates/frame:

| metric | value |
| --- | ---: |
| candidates / frame | ~50 |
| DetPr (official TrackEval) | ~1% |
| HOTA | 3.74 |
| DetA | 0.93 |
| AssA | 16.72 |
| MOTA | -4153 |
| IDF1 | 0.97 |

## 3. Repair stream (after filtering, before query-conditioning)

Builder: `tools/repair_l11_kitti_candidates.py`

- LVIS whitelist ids (LVIS v1 id = DLA label + 1):
  `{173,207,692,800,922,1114,1115,1123,94,703,701,1120,1179,793}`
  (bus, car, minivan, pickup truck, school bus, trailer truck, train,
  truck, bicycle, motorcycle, motor scooter, tricycle, wheelchair,
  person).
- Cross-category NMS IoU >= 0.7, greedy by det score.
- Score >= 0.05 (kept for query-conditioning headroom; the query CLIP
  step does the precision work).
- Top-30 per frame by det score.
- CLIP ViT-B/32 crop features computed for kept candidates.

Result (17 sequences with DLA dets: 13 train + 4 official eval):

| quantity | before | after |
| --- | ---: | ---: |
| total candidates | 284,800 (50/frame) | 86,890 |
| candidates / frame | ~50 | ~15.3 |

## 4. Calibration protocol (no leakage)

- Calibration uses only the 13 available official train sequences with
  DLA dets (0000,0001,0002,0003,0004,0006,0007,0008,0009,0010,0012,
  0014,0015).
- The 4 official eval sequences (0005,0011,0013,0019) are excluded from
  all threshold/top-k choices.
- GT identity is used only to measure precision/recall of the candidate
  stream on train; it is never used to filter candidates.

## 5. Calibration result (train sequences, 783,907 expression-frames)

Query-conditioned CLIP top-k sweep (candidates already whitelist+NMS+
top-30 filtered):

| top-k | query precision | target recall |
| ---: | ---: | ---: |
| 3 | 14.5% | 25.0% |
| 5 | 13.3% | 37.8% |
| 8 | 11.4% | 51.1% |
| 10 | 10.8% | 59.4% |
| **12** | **10.2%** | **65.1%** |
| 15 | 9.5% | 71.8% |
| 20 | 8.7% | 78.6% |

Chosen operating point: **top-k = 12, clip-min = 0.0** (query precision
10.2%, i.e. ~10x the L10 DetPr of ~1%, target recall 65.1%; the L10
baseline candidate stream at 50/frame has far lower query precision).
`results/l11/kitti_calibration.json` stores the full sweep.

The final official TrackEval numbers on the 4 eval sequences are
reported in `reports/l11_rmot_crossdomain_results.md`.
