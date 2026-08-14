# Stage L8 — Ordinary MOT Results (same shared checkpoint)

Protocol: identical to L6/L7 TrackEval four-domain regression (DanceTrack
val, BDD100K train, MOT17 train, MOT20 train), same LocateAnything-3B
candidate manifest, HOTA(0)/AssA(0)/IDF1/IDSW.

## Table A — Ordinary MOT regression

| Dataset | Method | HOTA | AssA | IDF1 | IDSW |
|---|---|---|---|---|---|
| DanceTrack | L6 PBD UIDM | 0.5546 | 0.3248 | 0.4958 | 5290 |
| DanceTrack | L7 CLIP UIDM | 0.5369 | 0.3045 | 0.4600* | 6164 |
| DanceTrack | L8-B1 sem-in-core | 0.5676 | 0.3405 | 0.4922 | 6503 |
| DanceTrack | L8 v2 shared | **0.5721** | **0.3457** | 0.5012 | 5773 |
| BDD100K | L6 PBD UIDM | 0.4716 | 0.4866 | 0.4110 | 7546 |
| BDD100K | L7 CLIP UIDM | 0.4317 | 0.4077 | — | 11430 |
| BDD100K | L8-B1 sem-in-core | 0.4913 | 0.5283 | 0.4368 | 6234 |
| BDD100K | L8 v2 shared | **0.4790** | **0.5019** | 0.4203 | 7074 |
| MOT17 | L6 PBD UIDM | 0.7084 | 0.6991 | — | — |
| MOT17 | L7 CLIP UIDM | 0.6471 | 0.5840 | — | 369 |
| MOT17 | L8-B1 sem-in-core | 0.7074 | 0.6982 | 0.6202 | 446 |
| MOT17 | L8 v2 shared | 0.7071 | 0.6970 | 0.6237 | 430 |
| MOT20 | L7 CLIP UIDM | 0.5973 | 0.4196 | — | 1799 |
| MOT20 | L8-B1 sem-in-core | 0.6305 | 0.4676 | 0.5572 | 1630 |
| MOT20 | L8 v2 shared | **0.6345** | **0.4734** | 0.5627 | 1619 |

Macro AssA:

- L6 PBD UIDM: **0.4922**
- L7 semantic/CLIP UIDM: 0.4290
- L8-B1 sem-in-core: **0.5087**
- L8 v2 identity-pure: **0.5045**

*L7 IDF1 for DanceTrack from the L7 report where available; "-" means the
value was not printed in the corresponding stage report.

Both L8 variants recover (and slightly exceed) L6. An early apparent
regression of the sem-in-core variant was traced to a PBD evaluation
feature-key bug (coord-mean vs box-end), not to the architecture.

Full per-sequence outputs:
`outputs/l8/trackeval/uidm_l8_v2_fix/{dance,bdd,mot17,mot20}/`
