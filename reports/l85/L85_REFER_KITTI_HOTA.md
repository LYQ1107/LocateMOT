# L85 Refer-KITTI V1 — full-video validation HOTA

## Evidence identity

This is an internal validation result, not official Refer-KITTI test HOTA. The
frozen checkpoint is L85 epoch 09, SHA256
`785e9a5cd7245f502e6e4eae84592c72b7703739c9d8edb599bc35c495a96f7e`, selected
on the video-disjoint internal dev data before this validation run. The frozen
rule is candidate threshold `1.2`, presence threshold `-0.1`, and NULL margin
`0.0`. The prediction source is
`outputs/l85/trackeval/fullvideo_validation_attempt5/refer_kitti_v1/`; the
authoritative TrackEval output is
`outputs/l85/trackeval/trackeval_attempt4/refer_kitti_v1/results/l85/`.

The run evaluates 86 query-sequences from videos `0004` and `0018`, covering
28,029 frame/query records. The local TrackEval adapter uses MOT17's
`pedestrian` class convention for the internal query sequences. It used
HOTA/CLEAR/Identity at IoU threshold `0.5`, with no parallel workers.

## Combined TrackEval result

| metric | value |
|---|---:|
| HOTA (AUC) | 25.0548 |
| DetA (AUC) | 14.9853 |
| AssA (AUC) | 42.4542 |
| LocA (AUC) | 89.8828 |
| DetRe (AUC) | 47.4128 |
| DetPr (AUC) | 17.8400 |
| AssRe (AUC) | 46.5967 |
| AssPr (AUC) | 77.2937 |
| IDF1 | 21.3342 |
| IDR / IDP | 39.0168 / 14.6808 |
| MOTA / MOTP | -170.5276 / 89.8152 |
| IDSW | 2,330 |
| CLR_FP / CLR_FN | 64,866 / 14,690 |

The values above are the combined row from
`pedestrian_detailed.csv`; the machine JSON keeps both raw TrackEval values
and percentage presentation. Counts are not percentage-scaled.

## Per-video descriptive breakdown

The following is a macro-average over query-sequence rows within each video,
not a replacement for the combined TrackEval row.

| video | query-sequences | HOTA | DetA | AssA | DetRe | DetPr | IDF1 | IDSW sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0004 | 45 | 15.0154 | 10.0238 | 30.6070 | 52.4791 | 12.0966 | 12.0665 | 1,402 |
| 0018 | 41 | 24.5560 | 17.0438 | 37.0347 | 43.7495 | 23.7626 | 24.9173 | 928 |

## Interpretation and boundary

The result is below the L85/L18 target of V1 HOTA `45+`; DetPr is also below
20. This is a genuine full-video TrackEval measurement of the frozen internal
validation protocol, but it is not a semantic-gate pass, official test result,
or ordinary MOT result. It does not authorize production promotion. No
token/span-to-region or static/motion alignment is verified (`UNALIGNED`).

Flags: `screening_gt_used=false`, `official_test_labels_read=false`,
`ordinary_mot_ovmot_touched=false`, `hota_trackeval_run=true` in the separate
TrackEval summary, and no ordinary MOT/OVMOT code or checkpoint was modified.
