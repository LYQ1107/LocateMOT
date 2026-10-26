# Fixed final-G2 candidate tracking results

Status: INCOMPLETE. Values are fractions except counts.

| Dataset | Arm | Jobs | Videos | Frames | HOTA | DetA | AssA | IDF1 | IDSW | Association FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| animaltrack | simple | 8 | 8 | 2308 | 0.268 | 0.354 | 0.217 | 0.276 | 3021 | 38.816 |
| refer_kitti_v2 | simple | 8 | 1 | 1672 | 0.150 | 0.159 | 0.145 | 0.137 | 385 | 571.190 |
| sportsmot | simple | 8 | 8 | 2050 | 0.610 | 0.761 | 0.491 | 0.591 | 606 | 273.822 |

No unweighted mean of per-video HOTA is used. Only paired complete scopes support net-gain claims. OVMOT requires official TETA, not MOT HOTA.

Current J1 artifact directory: `j1_current_crops_memory_v4`. Historical J1 input variants remain separate diagnostics; their different scopes are never combined into the current arm.

Coupled JEV currently supports ordinary MOT only. RMOT/OVMOT: `NOT_SUPPORTED_BY_CURRENT_CHECKPOINT`. Its selected checkpoint was trained on base-G0 cached features; evaluation on the final-G2 projector is an explicitly recorded feature-distribution shift. J1 step1200 and the separate step8227 OneJev model are not interchanged.

| OVMOT arm | Complete videos | TETA | LocA | AssocA | ClsA |
|---|---:|---:|---:|---:|---:|
| simple | 7 | 0.000 | 0.000 | 0.000 | 0.000 |
| simple base | 7 | 0.000 | 0.000 | 0.000 | 0.000 |
