# JEV net association gain

Status: INCOMPLETE. All differences use exactly paired complete native candidate scopes.

| Dataset | Compared arms | Jobs | ΔHOTA | ΔAssA | ΔIDF1 | ΔIDSW | Corrected | Harmed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sportsmot | Simple → j1 | 1 | -0.510 | -0.297 | -0.557 | -52 | 3 | 2090 |
| sportsmot | Simple → coupled | 5 | -0.425 | -0.376 | -0.422 | 4 | 93 | 8706 |
| animaltrack | Simple → coupled | 5 | -0.144 | -0.136 | -0.160 | 1246 | 163 | 7348 |
| refer_kitti_v2 | Simple → j1 | 1 | -0.012 | -0.025 | -0.013 | -1 | 0 | 1 |

Corrected/Harmed are post-hoc GT-matched temporal-edge diagnostics (IoU>=.5); they do not replace official HOTA/IDSW. Missing detections remain failures. Per-job details include births, occlusion returns and explicitly logged contradictory memory writes.
