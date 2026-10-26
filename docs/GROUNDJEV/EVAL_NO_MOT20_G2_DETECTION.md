# G2 perception results

Status: INCOMPLETE. Only officially scored complete registered videos/expressions appear below.

| Dataset | Videos | Complete jobs | Scored query-frames | GT boxes | Recall@.50 | Recall@.75 | FP/frame | Rejected | Detector FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sportsmot | 4 | 4 | 974 | 10655 | 0.988 | 0.894 | 1.252 | 0 | 0.321 |
| animaltrack | 4 | 4 | 1081 | 24043 | 0.450 | 0.261 | 5.060 | 111 | 0.235 |
| refer_kitti_v2 | 1 | 4 | 836 | 3343 | 0.309 | 0.251 | 2.565 | 192 | 0.556 |
| tao_amodal | 3 | 3 | 45 | 178 | 0.000 | 0.000 | PENDING | 45 | 0.004 |

AP50/AP75/mAP: N/A. GroundAnything supplies no genuine per-box confidence. No constant-score ranking AP is reported. TAO is non-exhaustive: unmatched proposals are not all verified false positives. Full per-frame coverage, groups, IoU and empty-query diagnostics are in each local `detection_metrics.json`.
