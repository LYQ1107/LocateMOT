# Fixed final-G2 candidate tracking results

Status: INCOMPLETE. Values are fractions except counts.

| Dataset | Arm | Jobs | Videos | Frames | HOTA | DetA | AssA | IDF1 | IDSW | Association FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sportsmot | j1 | 1 | 1 | 226 | 0.137 | 0.074 | 0.251 | 0.078 | 0 | 0.657 |
| sportsmot | coupled | 5 | 5 | 1240 | 0.216 | 0.307 | 0.153 | 0.203 | 351 | 107.088 |
| animaltrack | coupled | 5 | 5 | 1385 | 0.076 | 0.160 | 0.036 | 0.054 | 3672 | 74.339 |
| refer_kitti_v2 | j1 | 1 | 1 | 209 | 0.001 | 0.001 | 0.001 | 0.000 | 0 | 1.408 |

No unweighted mean of per-video HOTA is used. Only paired complete scopes support net-gain claims. OVMOT requires official TETA, not MOT HOTA.

Current J1 artifact directory: `j1_current_crops_memory_v4`. Historical J1 input variants remain separate diagnostics; their different scopes are never combined into the current arm.

Coupled JEV currently supports ordinary MOT only. RMOT/OVMOT: `NOT_SUPPORTED_BY_CURRENT_CHECKPOINT`. Its selected checkpoint was trained on base-G0 cached features; evaluation on the final-G2 projector is an explicitly recorded feature-distribution shift. J1 step1200 and the separate step8227 OneJev model are not interchanged.

| Dataset | Historical J1 arm | Complete jobs | Frames | HOTA | IDF1 |
|---|---|---:|---:|---:|---:|
| sportsmot | j1 | 3 | 715 | 0.183 | 0.104 |
| sportsmot | j1_competition_xyxy_v2 | 5 | 1240 | 0.165 | 0.092 |
| sportsmot | j1_current_crops_v3 | 1 | 226 | 0.137 | 0.078 |
| animaltrack | j1 | 4 | 1081 | 0.063 | 0.021 |
| animaltrack | j1_competition_xyxy_v2 | 2 | 500 | 0.076 | 0.029 |
| refer_kitti_v2 | j1 | 4 | 836 | 0.086 | 0.066 |
| refer_kitti_v2 | j1_competition_xyxy_v2 | 1 | 209 | 0.115 | 0.070 |

| OVMOT arm | Complete videos | TETA | LocA | AssocA | ClsA |
|---|---:|---:|---:|---:|---:|
