# Fixed final-G2 candidate tracking results

Status: INCOMPLETE. Values are fractions except counts.

| Dataset | Arm | Jobs | Videos | Frames | HOTA | DetA | AssA | IDF1 | IDSW | Association FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| animaltrack | j1 | 8 | 8 | 2308 | 0.084 | 0.032 | 0.223 | 0.032 | 172 | 0.324 |
| animaltrack | coupled | 8 | 8 | 2308 | 0.086 | 0.196 | 0.039 | 0.067 | 6610 | 72.946 |
| refer_kitti_v2 | j1 | 8 | 1 | 1672 | 0.096 | 0.069 | 0.135 | 0.075 | 0 | 1.430 |
| sportsmot | j1 | 8 | 8 | 2050 | 0.148 | 0.064 | 0.343 | 0.078 | 0 | 0.652 |
| sportsmot | coupled | 8 | 8 | 2050 | 0.221 | 0.307 | 0.160 | 0.210 | 511 | 89.143 |

No unweighted mean of per-video HOTA is used. Only paired complete scopes support net-gain claims. OVMOT requires official TETA, not MOT HOTA.

Current J1 artifact directory: `j1_current_crops_memory_v4`. Historical J1 input variants remain separate diagnostics; their different scopes are never combined into the current arm.

Coupled JEV currently supports ordinary MOT only. RMOT/OVMOT: `NOT_SUPPORTED_BY_CURRENT_CHECKPOINT`. Its selected checkpoint was trained on base-G0 cached features; evaluation on the final-G2 projector is an explicitly recorded feature-distribution shift. J1 step1200 and the separate step8227 OneJev model are not interchanged.

| Dataset | Historical J1 arm | Complete jobs | Frames | HOTA | IDF1 |
|---|---|---:|---:|---:|---:|
| animaltrack | j1 | 4 | 1081 | 0.063 | 0.021 |
| animaltrack | j1_competition_xyxy_v2 | 2 | 500 | 0.076 | 0.029 |
| refer_kitti_v2 | j1 | 4 | 836 | 0.086 | 0.066 |
| refer_kitti_v2 | j1_competition_xyxy_v2 | 1 | 209 | 0.115 | 0.070 |
| sportsmot | j1 | 3 | 715 | 0.183 | 0.104 |
| sportsmot | j1_competition_xyxy_v2 | 5 | 1240 | 0.165 | 0.092 |
| sportsmot | j1_current_crops_v3 | 1 | 226 | 0.137 | 0.078 |

| OVMOT arm | Complete videos | TETA | LocA | AssocA | ClsA |
|---|---:|---:|---:|---:|---:|
| j1 | 7 | 0.000 | 0.000 | 0.000 | 0.000 |
| j1 base | 7 | 0.000 | 0.000 | 0.000 | 0.000 |
