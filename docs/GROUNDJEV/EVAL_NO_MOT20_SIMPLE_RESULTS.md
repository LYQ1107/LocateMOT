# Fixed final-G2 candidate tracking results

Status: INCOMPLETE. Values are fractions except counts.

| Dataset | Arm | Complete jobs | HOTA | DetA | AssA | IDF1 | IDSW | Association FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sportsmot | simple | 3 | 0.676 | 0.786 | 0.583 | 0.679 | 171 | 291.777 |
| animaltrack | simple | 3 | 0.278 | 0.332 | 0.241 | 0.307 | 1078 | 37.666 |
| refer_kitti_v2 | simple | 3 | 0.106 | 0.138 | 0.083 | 0.098 | 206 | 529.416 |

No unweighted mean of per-video HOTA is used. Only paired complete scopes support net-gain claims. OVMOT requires official TETA, not MOT HOTA.

Coupled JEV currently supports ordinary MOT only. RMOT/OVMOT: `NOT_SUPPORTED_BY_CURRENT_CHECKPOINT`. Its selected checkpoint was trained on base-G0 cached features; evaluation on the final-G2 projector is an explicitly recorded feature-distribution shift. J1 step1200 and the separate step8227 OneJev model are not interchanged.

| OVMOT arm | Complete videos | TETA | LocA | AssocA | ClsA |
|---|---:|---:|---:|---:|---:|
| simple | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| simple base | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
