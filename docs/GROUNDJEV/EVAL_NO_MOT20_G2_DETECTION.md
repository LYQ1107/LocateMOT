# G2 perception results

Status: INCOMPLETE. Only officially scored complete registered videos/expressions appear below.

| Dataset | Videos | Complete jobs | Scored query-frames | GT boxes | Recall@.50 | Recall@.75 | FP/frame | Rejected | Detector FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sportsmot | 7 | 7 | 1779 | 19323 | 0.986 | 0.880 | 1.362 | 0 | 0.286 |
| animaltrack | 8 | 8 | 2308 | 51411 | 0.517 | 0.314 | 4.318 | 229 | 0.222 |
| refer_kitti_v2 | 1 | 7 | 1463 | 5256 | 0.353 | 0.269 | 2.625 | 293 | 0.520 |
| tao_amodal | 7 | 7 | 124 | 328 | 0.000 | 0.000 | PENDING | 124 | 0.003 |

AP50/AP75/mAP: N/A. GroundAnything supplies no genuine per-box confidence. No constant-score ranking AP is reported. TAO is non-exhaustive: unmatched proposals are not all verified false positives. Full per-frame coverage, groups, IoU and empty-query diagnostics are in each local `detection_metrics.json`.
