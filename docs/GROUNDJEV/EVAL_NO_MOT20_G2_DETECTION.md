# G2 perception results

Status: INCOMPLETE. Only officially scored complete registered videos/expressions appear below.

| Dataset | Videos | Complete jobs | Scored query-frames | GT boxes | Recall@.50 | Recall@.75 | FP/frame | Rejected | Detector FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sportsmot | 3 | 3 | 715 | 7868 | 0.987 | 0.888 | 1.169 | 0 | 0.319 |
| animaltrack | 3 | 3 | 780 | 19847 | 0.469 | 0.301 | 3.372 | 109 | 0.254 |
| refer_kitti_v2 | 1 | 3 | 627 | 3005 | 0.261 | 0.207 | 2.869 | 148 | 0.543 |
| tao_amodal | 2 | 2 | 30 | 127 | 0.000 | 0.000 | PENDING | 30 | 0.004 |

AP50/AP75/mAP: N/A. GroundAnything supplies no genuine per-box confidence. No constant-score ranking AP is reported. TAO is non-exhaustive: unmatched proposals are not all verified false positives. Full per-frame coverage, groups, IoU and empty-query diagnostics are in each local `detection_metrics.json`.
