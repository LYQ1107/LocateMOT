# Fixed final-G2 candidate tracking results

Status: INCOMPLETE. Values are fractions except counts.

| Dataset | Arm | Jobs | Videos | Frames | HOTA | DetA | AssA | IDF1 | IDSW | Association FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sportsmot | j1 | 4 | 4 | 974 | 0.172 | 0.079 | 0.372 | 0.096 | 0 | 0.625 |
| sportsmot | coupled | 4 | 4 | 974 | 0.222 | 0.327 | 0.152 | 0.211 | 335 | 105.441 |
| animaltrack | coupled | 4 | 4 | 1081 | 0.079 | 0.151 | 0.043 | 0.056 | 2142 | 78.712 |

No unweighted mean of per-video HOTA is used. Only paired complete scopes support net-gain claims. OVMOT requires official TETA, not MOT HOTA.

Current J1 artifact directory: `j1_competition_xyxy_v2`. Legacy `j1` outputs remain separate diagnostics after the confirmed competition-coordinate transport bug.

Coupled JEV currently supports ordinary MOT only. RMOT/OVMOT: `NOT_SUPPORTED_BY_CURRENT_CHECKPOINT`. Its selected checkpoint was trained on base-G0 cached features; evaluation on the final-G2 projector is an explicitly recorded feature-distribution shift. J1 step1200 and the separate step8227 OneJev model are not interchanged.

| Legacy J1 diagnostic | Complete jobs | HOTA | IDF1 |
|---|---:|---:|---:|
| sportsmot | 3 | 0.183 | 0.104 |
| animaltrack | 4 | 0.063 | 0.021 |
| refer_kitti_v2 | 4 | 0.086 | 0.066 |

| OVMOT arm | Complete videos | TETA | LocA | AssocA | ClsA |
|---|---:|---:|---:|---:|---:|
| j1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| j1 base | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
