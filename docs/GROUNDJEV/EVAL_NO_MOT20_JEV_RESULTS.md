# Fixed final-G2 candidate tracking results

Status: INCOMPLETE. Values are fractions except counts.

| Dataset | Arm | Complete jobs | HOTA | DetA | AssA | IDF1 | IDSW | Association FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sportsmot | j1 | 3 | 0.183 | 0.075 | 0.444 | 0.104 | 0 | 0.882 |
| sportsmot | coupled | 2 | 0.255 | 0.298 | 0.218 | 0.252 | 73 | 111.749 |
| animaltrack | j1 | 3 | 0.063 | 0.025 | 0.157 | 0.022 | 0 | 0.480 |
| animaltrack | coupled | 3 | 0.072 | 0.161 | 0.033 | 0.048 | 2115 | 87.755 |
| refer_kitti_v2 | j1 | 3 | 0.083 | 0.055 | 0.125 | 0.059 | 0 | 2.025 |

No unweighted mean of per-video HOTA is used. Only paired complete scopes support net-gain claims. OVMOT requires official TETA, not MOT HOTA.

Coupled JEV currently supports ordinary MOT only. RMOT/OVMOT: `NOT_SUPPORTED_BY_CURRENT_CHECKPOINT`. Its selected checkpoint was trained on base-G0 cached features; evaluation on the final-G2 projector is an explicitly recorded feature-distribution shift. J1 step1200 and the separate step8227 OneJev model are not interchanged.

| OVMOT arm | Complete videos | TETA | LocA | AssocA | ClsA |
|---|---:|---:|---:|---:|---:|
| j1 | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| j1 base | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
