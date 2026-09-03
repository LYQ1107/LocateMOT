# L86 TrackEval protocol audit

The authoritative learned internal TrackEval output is
`outputs/l86/trackeval/fullvideo_eval_attempt1/trackeval_summary.json`. It
uses the local `/data1/LWR/vranlee/SERVER_ONLY/avis/TrackEval-master` checkout,
metrics `HOTA`, `CLEAR`, and `Identity`, IoU threshold `0.5`, and the explicit
internal query-sequence adapter. The TrackEval checkout has no verifiable Git
HEAD; it was not modified.

GT was materialized only after checkpoint, emission rule, and prediction
strategy were frozen. The scope is internal full-video validation: V1 videos
`0004` and `0018` (86 sequences), and V2 videos `0016`, `0017`, and `0020`
(537 sequences). It is not official Refer-KITTI test HOTA, screening HOTA, or
a published ordinary-RMOT comparison.

| dataset | sequences | HOTA | DetA | AssA | LocA | DetRe | DetPr | IDF1 | IDSW | CLR_FP | CLR_FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Refer-KITTI V1 internal | 86 | 29.1663 | 20.5210 | 41.8054 | 91.1238 | 69.5047 | 22.3567 | 23.3188 | 1,814 | 71,429 | 7,595 |
| Refer-KITTI V2 internal | 537 | 21.6467 | 13.3584 | 35.2978 | 90.0181 | 42.4989 | 16.1970 | 17.7058 | 11,873 | 651,767 | 163,801 |

Compared with the immutable L85 internal baseline:

| dataset | HOTA delta | DetA delta | AssA delta | DetRe delta | DetPr delta | IDF1 delta | IDSW delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1 | +4.1115 | +5.5357 | -0.6488 | +22.0919 | +4.5167 | +1.9846 | -516 |
| V2 | +4.3543 | +3.5705 | +4.4589 | -3.4503 | +5.1979 | +4.7628 | -6,501 |

The L86 material-improvement descriptor is true because both HOTA values are
at least three points above L85. This does not override the failed fixed
semantic gate and does not authorize screening or production integration.

The first TrackEval CLI/protocol attempt for the GT oracle had an overwritten
seqmap; it is preserved as `semantic_oracle_eval_attempt1/INCOMPLETE.md`. A
minimal seqmap repair produced `semantic_oracle_attempt2`, after which the
learned full-video wrapper completed without a protocol error.

Machine flags for the learned result are
`hota_trackeval_run=true`, `hota_scope=internal_full_video_validation`,
`screening_gt_used=false`, `official_test_labels_read=false`, and
`ordinary_mot_ovmot_touched=false`.
