# Stage L9 — Ordinary MOT results (official TrackEval)

Status: L8 baselines below; L9 rows pending the final L9 checkpoint.

## Protocol

Same four-domain official TrackEval protocol as L6/L7/L8 (DanceTrack val,
BDD100K, MOT17 train, MOT20 train), one shared checkpoint, category spec
text.  Reported: HOTA / DetA / AssA / IDF1 / MOTA / IDSW (full rows in
`outputs/l9/trackeval/` when available).

## AssA rows

| Dataset | L6 PBD | L7 CLIP | L8-B2 | L8-B1 | L9 main |
|---|---|---|---|---|---|
| DanceTrack | 0.3248 | 0.3045 | 0.3457 | | TBD |
| BDD100K | 0.4866 | 0.4077 | 0.5019 | | TBD |
| MOT17 | 0.6991 | 0.5840 | 0.6970 | | TBD |
| MOT20 | — | 0.4196 | 0.4734 | | TBD |
| Macro | 0.4922 | 0.4290 | 0.5045 | 0.5087 | TBD |

Full HOTA/DetA/IDF1/MOTA/IDSW for L8-B1/B2 are in
`reports/l8_mot_results.md`.

## L9 main (v5, cond-gated, official TrackEval)

| Dataset | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|---|---|
| DanceTrack | 0.5763 | | 0.3509 | 0.4991 | | 6282 |
| BDD100K | 0.4832 | | 0.5108 | 0.4293 | | 6476 |
| MOT17 | 0.7095 | | 0.7017 | 0.6256 | | 437 |
| MOT20 | 0.6341 | | 0.4727 | 0.5609 | | 1573 |
| Macro | — | — | **0.5090** | — | — | — |

DetA/MOTA rows are in the TrackEval logs
(`outputs/l9/trackeval/l9_main_v5/ordinary_gpu*.log`) and will be filled
into this table in the final report.
