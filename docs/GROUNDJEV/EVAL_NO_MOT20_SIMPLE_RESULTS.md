# Fixed final-G2 candidate tracking results

Status: INCOMPLETE. Values are fractions except counts.

| Dataset | Arm | Jobs | Videos | Frames | HOTA | DetA | AssA | IDF1 | IDSW | Association FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sportsmot | simple | 4 | 4 | 974 | 0.656 | 0.782 | 0.553 | 0.654 | 234 | 287.599 |
| animaltrack | simple | 4 | 4 | 1081 | 0.249 | 0.309 | 0.213 | 0.258 | 1435 | 45.086 |
| refer_kitti_v2 | simple | 4 | 1 | 836 | 0.137 | 0.160 | 0.119 | 0.120 | 254 | 554.441 |

No unweighted mean of per-video HOTA is used. Only paired complete scopes support net-gain claims. OVMOT requires official TETA, not MOT HOTA.

Current J1 artifact directory: `j1_competition_xyxy_v2`. Legacy `j1` outputs remain separate diagnostics after the confirmed competition-coordinate transport bug.

Coupled JEV currently supports ordinary MOT only. RMOT/OVMOT: `NOT_SUPPORTED_BY_CURRENT_CHECKPOINT`. Its selected checkpoint was trained on base-G0 cached features; evaluation on the final-G2 projector is an explicitly recorded feature-distribution shift. J1 step1200 and the separate step8227 OneJev model are not interchanged.

| OVMOT arm | Complete videos | TETA | LocA | AssocA | ClsA |
|---|---:|---:|---:|---:|---:|
| simple | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| simple base | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
