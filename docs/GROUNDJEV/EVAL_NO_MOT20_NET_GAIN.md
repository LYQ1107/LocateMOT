# JEV net association gain

Status: INCOMPLETE. All differences use exactly paired complete native candidate scopes.

| Dataset | Compared arms | Jobs | ΔHOTA | ΔAssA | ΔIDF1 | ΔIDSW | Corrected | Harmed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sportsmot | Simple → j1 | 3 | -0.493 | -0.140 | -0.575 | -171 | 6 | 6777 |
| sportsmot | Simple → coupled | 2 | -0.364 | -0.284 | -0.366 | -77 | 41 | 3166 |
| animaltrack | Simple → j1 | 3 | -0.216 | -0.083 | -0.285 | -1078 | 33 | 5840 |
| animaltrack | Simple → coupled | 3 | -0.207 | -0.208 | -0.258 | 1037 | 85 | 4820 |
| refer_kitti_v2 | Simple → j1 | 3 | -0.023 | 0.042 | -0.039 | -206 | 25 | 264 |

Corrected/Harmed are post-hoc GT-matched temporal-edge diagnostics (IoU>=.5); they do not replace official HOTA/IDSW. Missing detections remain failures. Per-job details include births, occlusion returns and explicitly logged contradictory memory writes.
