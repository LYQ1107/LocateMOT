# Stage L11 — Pseudo-Track Quality Report

Date: 2026-08-19

## Scope

First 8 TAO train videos from the L10 index (40 pkl-frames each):

- train-AVA-5BDj0ow5hnA_scene_10_54385-55758
- train-AVA-5BDj0ow5hnA_scene_13_61290-62898
- train-AVA-5BDj0ow5hnA_scene_15_65137-66983
- train-AVA-5BDj0ow5hnA_scene_21_98119-99086
- train-AVA-5BDj0ow5hnA_scene_9_49845-51043
- train-AVA-D8Vhxbho1fY_scene_7_117261-118261
- train-AVA-D8Vhxbho1fY_scene_8_128633-130030
- train-AVA-QotkBTEePI8_scene_2_24728-26072

## Aggregate quality (raw linker, `link_id`)

| metric | value |
| --- | ---: |
| videos / frames | 8 / 320 |
| link candidates | 3,360 |
| training pseudo candidates | 2,745 (81.7% of link cands) |
| kept tracklets | 369 |
| tracklet length mean / median | 9.1 / 5.0 |
| NEW transitions (tracklet starts) | 369 |
| Existing transitions | 2,991 |
| NEW rate | 10.98% |
| unique-ID ratio (ids/cands) | 0.110 |
| cycle pass rate (weighted) | 0.997 |
| mean appearance self-consistency | 0.965 |
| latent-GT coverage (IoU>=0.4) | 11.3% of link cands |
| **pseudo same-ID precision (pair-wise)** | **99.26%** |
| pseudo same-ID precision (majority label) | 99.21% |
| duplicate-identity frames (raw linker) | 98/318 = 30.8% |
| - same-label fragmentation | 1/108 frames |
| - cross-label DLA duplicate detections | 107/108 frames |

The same-ID precision target (>= 90% on the GT-covered subset) is met
with margin (99.26%).  The only notable residual is cross-label DLA
duplicate detections (same physical object fired under two LVIS labels),
which do not enter the final `pseudo_id` training signal when they
overlap GT and are intentionally kept separate when they do not
(category consistency is a precision-preserving choice).

## Comparison with the L10 collapse

| quantity | L10 collapsed model (sample video) | L11 pseudo-track design |
| --- | ---: | ---: |
| unique IDs / prediction rows | 1,612 / 1,650 (97.7%) | 369 IDs / 3,360 cands (11%) |
| NEW / (NEW+Existing) | ~1.0 | 0.110 |
| GT coverage (training) | ~10% (base, IoU 0.5) | ~28% (base_and_novel, IoU 0.3) + 11% high-precision pseudo |

## Threshold headroom

With current filters the linker is already at 99.3% pair precision.
If a later evidence-driven correction is needed, tightening
`MIN_MEAN_APP` / `MIN_CYCLE_RATE` / `MIN_TRACKLET_LEN` gives
additional precision margin at the cost of coverage.
