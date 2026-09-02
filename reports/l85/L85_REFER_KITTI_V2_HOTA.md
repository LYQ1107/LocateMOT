# L85 Refer-KITTI-V2 — full-video validation HOTA

## Evidence identity

This is an internal validation result, not official Refer-KITTI-V2 test HOTA.
The frozen checkpoint is L85 epoch 09, SHA256
`785e9a5cd7245f502e6e4eae84592c72b7703739c9d8edb599bc35c495a96f7e`, selected
on video-disjoint internal dev data before validation. The frozen rule is
candidate threshold `1.2`, presence threshold `-0.1`, and NULL margin `0.0`.
The prediction source is
`outputs/l85/trackeval/fullvideo_validation_attempt5/refer_kitti_v2/`; the
authoritative TrackEval output is
`outputs/l85/trackeval/trackeval_attempt4/refer_kitti_v2/results/l85/`.

The run evaluates 537 query-sequences from videos `0016`, `0017`, and `0020`,
covering 215,521 frame/query records. The local TrackEval adapter uses the
MOT17 `pedestrian` class convention for these internal query sequences. It
used HOTA/CLEAR/Identity at IoU threshold `0.5`, with no parallel workers.

## Combined TrackEval result

| metric | value |
|---|---:|
| HOTA (AUC) | 17.2924 |
| DetA (AUC) | 9.7879 |
| AssA (AUC) | 30.8389 |
| LocA (AUC) | 88.2524 |
| DetRe (AUC) | 45.9492 |
| DetPr (AUC) | 10.9991 |
| AssRe (AUC) | 36.4429 |
| AssPr (AUC) | 65.8326 |
| IDF1 | 12.9430 |
| IDR / IDP | 33.5064 / 8.0206 |
| MOTA / MOTP | -325.1462 / 88.3905 |
| IDSW | 18,374 |
| CLR_FP / CLR_FN | 1,106,999 / 152,170 |

The values above are the combined row from
`pedestrian_detailed.csv`; the machine JSON keeps raw TrackEval values and
percentage presentation. Counts are not percentage-scaled.

## Per-video descriptive breakdown

The following is a macro-average over query-sequence rows within each video,
not a replacement for the combined TrackEval row.

| video | query-sequences | HOTA | DetA | AssA | DetRe | DetPr | IDF1 | IDSW sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0016 | 183 | 10.6496 | 9.1163 | 13.7099 | 28.5437 | 13.2009 | 10.7946 | 3,844 |
| 0017 | 172 | 14.1776 | 9.4442 | 22.9280 | 49.3038 | 10.7687 | 12.3754 | 2,721 |
| 0020 | 182 | 14.9166 | 8.2662 | 34.4045 | 50.8544 | 9.8829 | 10.8951 | 11,809 |

## Interpretation and boundary

The result is below the L85/L18 target of V2 HOTA about `40`; DetPr is
`10.9991`, and the high IDSW/false-positive counts show that the weak semantic
emission and identity interaction remain unresolved. This is a genuine
full-video TrackEval measurement of the frozen internal validation protocol,
not a semantic-gate pass, official test result, or ordinary MOT result. No
token/span-to-region or static/motion alignment is verified (`UNALIGNED`).

Flags: `screening_gt_used=false`, `official_test_labels_read=false`,
`ordinary_mot_ovmot_touched=false`, `hota_trackeval_run=true` in the separate
TrackEval summary.
