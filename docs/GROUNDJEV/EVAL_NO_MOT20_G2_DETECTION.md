# G2 perception results

Status: INCOMPLETE. Only officially scored complete registered videos/expressions appear below.

| Dataset | Videos | Complete jobs | Scored query-frames | GT boxes | Recall@.50 | Recall@.75 | FP/frame | Rejected | Detector FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sportsmot | 6 | 6 | 1509 | 16267 | 0.986 | 0.880 | 1.404 | 0 | 0.293 |
| animaltrack | 6 | 6 | 1691 | 40735 | 0.475 | 0.271 | 5.233 | 140 | 0.238 |
| refer_kitti_v2 | 1 | 6 | 1254 | 4835 | 0.334 | 0.258 | 2.498 | 271 | 0.530 |
| tao_amodal | 4 | 4 | 60 | 179 | 0.000 | 0.000 | PENDING | 92 | 0.003 |

AP50/AP75/mAP: N/A. GroundAnything supplies no genuine per-box confidence. No constant-score ranking AP is reported. TAO is non-exhaustive: unmatched proposals are not all verified false positives. Full per-frame coverage, groups, IoU and empty-query diagnostics are in each local `detection_metrics.json`.
