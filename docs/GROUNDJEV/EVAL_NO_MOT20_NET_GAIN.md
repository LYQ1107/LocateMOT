# JEV net association gain

Status: INCOMPLETE. All differences use exactly paired complete native candidate scopes.

| Dataset | Compared arms | Jobs | ΔHOTA | ΔAssA | ΔIDF1 | ΔIDSW | Corrected | Harmed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sportsmot | Simple → j1 | 7 | -0.464 | -0.127 | -0.522 | -500 | 13 | 16740 |
| sportsmot | Simple → coupled | 7 | -0.403 | -0.350 | -0.401 | -21 | 119 | 12201 |
| animaltrack | Simple → j1 | 7 | -0.171 | 0.030 | -0.225 | -2685 | 63 | 14534 |
| animaltrack | Simple → coupled | 7 | -0.174 | -0.169 | -0.195 | 3504 | 237 | 11828 |
| refer_kitti_v2 | Simple → j1 | 7 | -0.058 | -0.008 | -0.071 | -380 | 45 | 790 |

Corrected/Harmed are post-hoc GT-matched temporal-edge diagnostics (IoU>=.5); they do not replace official HOTA/IDSW. Missing detections remain failures. Per-job details include births, occlusion returns and explicitly logged contradictory memory writes.
