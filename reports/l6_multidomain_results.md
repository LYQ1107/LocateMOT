# Stage L6 Multi-Domain Results

Status: COMPLETE (tag `uidm_final`, checkpoint `latest.pt` epoch 18).

Protocol: fresh tracker tag + fresh TrackEval directory per run.
Domains: DanceTrack val, BDD100K train subset, MOT17 train, MOT20 train.
Metrics: HOTA / DetA / AssA / IDF1 / IDSW + Macro (domain-equal mean).

| Domain | HOTA | DetA | AssA | IDF1 | IDSW |
|---|---:|---:|---:|---:|---:|
| DanceTrack val | 0.5546 | 0.9468 | 0.3248 | 0.4958 | 5290 |
| BDD100K train | 0.4716 | 0.4571 | 0.4866 | 0.4110 | 7546 |
| MOT17 train | 0.7084 | 0.7179 | 0.6991 | 0.6244 | 434 |
| MOT20 train | 0.6242 | 0.8500 | 0.4584 | 0.5482 | 1645 |

| Macro | U0 (B3) | UIDM | Δ |
|---|---:|---:|---:|
| HOTA | 0.5379 | 0.5897 | +5.2pp |
| AssA | 0.4013 | 0.4922 | +9.1pp |
| IDF1 | 0.4614 | 0.5199 | +5.9pp |

3/4 domains improve on AssA; DanceTrack collapses.  Full comparison
against B0–B4 in `reports/l6_baseline_reconciliation.md` and
`reports/STAGE_L6_FINAL_REPORT.md`.
