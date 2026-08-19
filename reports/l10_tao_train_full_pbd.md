# Stage L10 — Full TAO-Train Crop-PBD Cache

Date: 2026-08-17 (status: in progress)

## Goal

Build a full TAO-train OVMOT training stream whose observation
distribution matches the TAO-val full-PBD evaluation protocol:

- candidates = DLA (Detic-SwinB, MASA checkpoint) detections, score>=0.05,
  top-50, on all 500 TAO train videos / 18,274 frames;
- identity supervision = C-TAO continuous annotations (COVTrack ICCV
  2025, `ctao_base.json`, base categories; 490,210 frames / 1,489,637
  boxes for the same 500 videos);
- per candidate: frozen CLIP ViT-B/32 crop feature [512, fp16] and
  LocateAnything-3B crop-PBD box-end token [2048, fp16];
- hard negatives = unmatched DLA detections (expected ~95% of
  candidates, matching the val match rate).

## Pipeline

1. `tools/generate_l9_tao_train_dets_subset.py` (DLA, roi_align patch)
   -> `outputs/l10/cache/tao_train_candidates/train/<ds>/<video>/frameNNNN.pth`
   (18,274 frames).
2. `tools/build_l10_tao_train.py` (DLA pth -> per-video pkl + CLIP crops +
   C-TAO candidate-GT alignment) -> `outputs/l10/data/tao_train/*.pkl`.
3. `tools/run_l10_pbd_cache.py` -> `tools/cache_l9_tao_pbd.py` (8-14
   workers) -> `outputs/l10/cache/tao_train_pbd/` (per-frame safetensors).
4. `tools/merge_l10_train_pbd.py` -> fills `fr["pbd"]` in the pkls
   (`pbd_box_end_last`, fp16, candidate-aligned, finite).

## Candidate / GT stats (to be finalized)

| item | L9 stream | L10 target |
| --- | ---: | ---: |
| videos | 105 | 500 |
| frames | 4,200 | 18,274 |
| candidates | 7,522 | 905,400 -> 322,843* |
| GT source | TAO train (sparse) + L6 boxes | C-TAO continuous (base) |
| detector | LocateAnything | Detic-SwinB (same family as val public dets) |
| expected match rate | ~86% | ~3-5% (val protocol-like) |

*Final training stream is hard-negative capped: all 31,277 GT-matched
candidates + top-16 unmatched by score per frame (17.7 candidates/frame
on average; 99.8% of positives retained).  Rationale: LocateAnything-3B
crop-PBD is the pipeline bottleneck (~0.6-0.8 s/crop, ~40k crops/h on the
4-GPU server); the full 905,400-candidate cache would take ~23 h, while
the capped stream is a 43x supervision increase over L9 and keeps the
hard-negative structure of the val protocol.  Evaluation is unchanged
(full TAO-val candidates).

Val (public Detic dets, 50-video sample): 42.9 dets/frame, 5.0% matched.
The train/val match-rate difference is attributed to C-TAO base-only GT
(novel categories not annotated) and is documented.

## Regression checks

- PBD key: `pbd_box_end_last`, dim 2048, fp32 cache -> fp16 merge.
- Candidate order: `len(pbd) == len(boxes)` per frame (merge asserts).
- Finite: merge asserts `np.isfinite(pbd).all()`.
- `pbd_box_end_last` only (not coord-mean; the L8/L9 PBD rule is
  enforced at the cache and merge level).

## Progress

Filled in the final report with:

- DLA generation completion time / failures;
- builder videos/frames/candidates;
- PBD cache workers, wall-clock, fail rates;
- merge coverage (frames with non-zero PBD / total).
