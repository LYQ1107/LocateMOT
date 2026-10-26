# Fixed final-G2 candidate tracking results

Status: INCOMPLETE. Values are fractions except counts.

| Dataset | Arm | Jobs | Videos | Frames | HOTA | DetA | AssA | IDF1 | IDSW | Association FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sportsmot | simple | 7 | 7 | 1779 | 0.618 | 0.764 | 0.502 | 0.603 | 500 | 273.545 |
| animaltrack | simple | 8 | 8 | 2308 | 0.268 | 0.354 | 0.217 | 0.276 | 3021 | 38.816 |
| refer_kitti_v2 | simple | 7 | 1 | 1463 | 0.156 | 0.170 | 0.146 | 0.144 | 380 | 563.823 |

No unweighted mean of per-video HOTA is used. Only paired complete scopes support net-gain claims. OVMOT requires official TETA, not MOT HOTA.

Current J1 artifact directory: `j1_current_crops_memory_v4`. Historical J1 input variants remain separate diagnostics; their different scopes are never combined into the current arm.

Coupled JEV currently supports ordinary MOT only. RMOT/OVMOT: `NOT_SUPPORTED_BY_CURRENT_CHECKPOINT`. Its selected checkpoint was trained on base-G0 cached features; evaluation on the final-G2 projector is an explicitly recorded feature-distribution shift. J1 step1200 and the separate step8227 OneJev model are not interchanged.

| OVMOT arm | Complete videos | TETA | LocA | AssocA | ClsA |
|---|---:|---:|---:|---:|---:|
| simple | 7 | 0.000 | 0.000 | 0.000 | 0.000 |
| simple base | 7 | 0.000 | 0.000 | 0.000 | 0.000 |
