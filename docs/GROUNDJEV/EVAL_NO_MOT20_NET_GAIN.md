# JEV net association gain

Status: INCOMPLETE. All differences use exactly paired complete native candidate scopes.

| Dataset | Compared arms | Jobs | ΔHOTA | ΔAssA | ΔIDF1 | ΔIDSW | Corrected | Harmed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sportsmot | Simple → j1 | 2 | -0.438 | -0.024 | -0.518 | -150 | 3 | 4341 |
| sportsmot | Simple → coupled | 5 | -0.425 | -0.376 | -0.422 | 4 | 93 | 8706 |
| animaltrack | Simple → j1 | 2 | -0.216 | 0.184 | -0.306 | -482 | 18 | 5720 |
| animaltrack | Simple → coupled | 6 | -0.166 | -0.165 | -0.186 | 1975 | 220 | 9400 |
| refer_kitti_v2 | Simple → j1 | 2 | -0.022 | -0.046 | -0.021 | -13 | 0 | 29 |

Corrected/Harmed are post-hoc GT-matched temporal-edge diagnostics (IoU>=.5); they do not replace official HOTA/IDSW. Missing detections remain failures. Per-job details include births, occlusion returns and explicitly logged contradictory memory writes.
