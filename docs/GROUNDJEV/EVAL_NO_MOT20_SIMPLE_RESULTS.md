# Fixed final-G2 candidate tracking results

Status: INCOMPLETE. Values are fractions except counts.

| Dataset | Arm | Jobs | Videos | Frames | HOTA | DetA | AssA | IDF1 | IDSW | Association FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sportsmot | simple | 5 | 5 | 1240 | 0.641 | 0.781 | 0.529 | 0.626 | 347 | 283.874 |
| animaltrack | simple | 6 | 6 | 1691 | 0.249 | 0.326 | 0.204 | 0.247 | 2605 | 36.936 |
| refer_kitti_v2 | simple | 6 | 1 | 1254 | 0.157 | 0.169 | 0.147 | 0.147 | 324 | 572.297 |

No unweighted mean of per-video HOTA is used. Only paired complete scopes support net-gain claims. OVMOT requires official TETA, not MOT HOTA.

Current J1 artifact directory: `j1_current_crops_memory_v4`. Historical J1 input variants remain separate diagnostics; their different scopes are never combined into the current arm.

Coupled JEV currently supports ordinary MOT only. RMOT/OVMOT: `NOT_SUPPORTED_BY_CURRENT_CHECKPOINT`. Its selected checkpoint was trained on base-G0 cached features; evaluation on the final-G2 projector is an explicitly recorded feature-distribution shift. J1 step1200 and the separate step8227 OneJev model are not interchanged.

| OVMOT arm | Complete videos | TETA | LocA | AssocA | ClsA |
|---|---:|---:|---:|---:|---:|
| simple | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| simple base | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
