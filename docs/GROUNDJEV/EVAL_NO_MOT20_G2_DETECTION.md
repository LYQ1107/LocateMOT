# G2 perception results

Status: INCOMPLETE. Only officially scored complete registered videos/expressions appear below.

| Dataset | Videos | Complete jobs | Scored query-frames | GT boxes | Recall@.50 | Recall@.75 | FP/frame | Rejected | Detector FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| animaltrack | 8 | 8 | 2308 | 51411 | 0.517 | 0.314 | 4.318 | 229 | 0.222 |
| refer_kitti_v2 | 1 | 8 | 1672 | 5409 | 0.346 | 0.261 | 2.646 | 314 | 0.521 |
| sportsmot | 8 | 8 | 2050 | 22329 | 0.986 | 0.876 | 1.380 | 0 | 0.279 |
| tao_amodal | 7 | 7 | 124 | 328 | 0.000 | 0.000 | PENDING | 124 | 0.003 |

AP50/AP75/mAP: N/A. GroundAnything supplies no genuine per-box confidence. No constant-score ranking AP is reported. TAO is non-exhaustive: unmatched proposals are not all verified false positives. Full per-frame coverage, groups, IoU and empty-query diagnostics are in each local `detection_metrics.json`.
