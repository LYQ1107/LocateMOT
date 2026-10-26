# JEV net association gain

Status: INCOMPLETE. All differences use exactly paired complete native candidate scopes.

| Dataset | Compared arms | Jobs | ΔHOTA | ΔAssA | ΔIDF1 | ΔIDSW | Corrected | Harmed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| animaltrack | Simple → coupled | 8 | -0.182 | -0.178 | -0.209 | 3589 | 256 | 13404 |
| animaltrack | Simple → j1 | 8 | -0.185 | 0.007 | -0.244 | -2849 | 74 | 16718 |
| refer_kitti_v2 | Simple → j1 | 8 | -0.054 | -0.009 | -0.062 | -385 | 48 | 790 |
| sportsmot | Simple → coupled | 8 | -0.389 | -0.332 | -0.381 | -95 | 131 | 13920 |
| sportsmot | Simple → j1 | 8 | -0.462 | -0.149 | -0.513 | -606 | 16 | 19269 |

Corrected/Harmed are post-hoc GT-matched temporal-edge diagnostics (IoU>=.5); they do not replace official HOTA/IDSW. Missing detections remain failures. Per-job details include births, occlusion returns and explicitly logged contradictory memory writes.
