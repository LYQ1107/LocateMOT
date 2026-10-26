# JEV net association gain

Status: INCOMPLETE. All differences use exactly paired complete native candidate scopes.

| Dataset | Compared arms | Jobs | ΔHOTA | ΔAssA | ΔIDF1 | ΔIDSW | Corrected | Harmed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sportsmot | Simple → j1 | 3 | -0.451 | -0.155 | -0.516 | -213 | 8 | 6709 |
| sportsmot | Simple → coupled | 3 | -0.446 | -0.396 | -0.458 | -71 | 45 | 5126 |
| animaltrack | Simple → coupled | 4 | -0.170 | -0.171 | -0.202 | 707 | 110 | 5363 |

Corrected/Harmed are post-hoc GT-matched temporal-edge diagnostics (IoU>=.5); they do not replace official HOTA/IDSW. Missing detections remain failures. Per-job details include births, occlusion returns and explicitly logged contradictory memory writes.
