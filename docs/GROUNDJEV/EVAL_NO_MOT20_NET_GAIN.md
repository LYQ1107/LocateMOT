# JEV net association gain

Status: INCOMPLETE. All differences use exactly paired complete native candidate scopes.

| Dataset | Compared arms | Jobs | ΔHOTA | ΔAssA | ΔIDF1 | ΔIDSW | Corrected | Harmed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sportsmot | Simple → j1 | 4 | -0.485 | -0.181 | -0.558 | -234 | 8 | 9157 |
| sportsmot | Simple → coupled | 4 | -0.434 | -0.401 | -0.443 | 101 | 64 | 6656 |
| animaltrack | Simple → coupled | 4 | -0.170 | -0.171 | -0.202 | 707 | 110 | 5363 |

Corrected/Harmed are post-hoc GT-matched temporal-edge diagnostics (IoU>=.5); they do not replace official HOTA/IDSW. Missing detections remain failures. Per-job details include births, occlusion returns and explicitly logged contradictory memory writes.
