# L84 three-seed results

The values below are mean dev-group aggregate metrics over the three paired
seeds.  They are representation diagnostics, not expression-emission
validation and not HOTA.

| state | bag hard violation | hit@1 | recall@5 | multi-target exact | query-swap accuracy | V2 hard | V2 hit@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Z0 | 0.804487 | 0.298077 | 0.615866 | 0.238994 | 0.726974 | 0.813264 | 0.279232 |
| Z1 | 0.705128 | 0.407051 | 0.698678 | 0.308176 | 0.744518 | 0.766143 | 0.315881 |
| Z4 | 0.723291 | 0.389957 | 0.743911 | 0.249476 | 0.731360 | 0.706806 | 0.392670 |
| Z6 | 0.776709 | 0.351496 | 0.711900 | 0.203354 | 0.765899 | 0.815009 | 0.312391 |
| R1 | 0.705128 | 0.407051 | 0.698678 | 0.308176 | 0.744518 | 0.766143 | 0.315881 |
| R4 | 0.753205 | 0.376068 | 0.741823 | 0.218029 | 0.749452 | 0.766143 | 0.375218 |
| R6 | 0.754274 | 0.375000 | 0.721642 | 0.257862 | 0.766447 | 0.783595 | 0.335079 |

The machine source is
`outputs/l84/train/paired_middecoder/paired_stage_metrics.json`.  Z1 and R1
have identical saved dev records and identical trained probe state for every
seed, so the corrected earliest-stage tie-break selects Z1 rather than R1.
The stable gate passed for Z1 and R1; Z4/R4/R6 did not pass the full paired
bootstrap/stability requirements.

The Z1 per-seed deltas against Z0 were hard improvement
`0.083333/0.115385/0.099359`, hit@1 improvement
`0.083333/0.141026/0.102564`, V2 hard improvement
`0.020942/0.062827/0.057592`, and V2 hit improvement
`0/0.057592/0.052356`.  The pooled 10,000-resample bootstrap for Z1 gave
hard-improvement point `0.126881`, 95% CI `[0.061042, 0.193496]`.

This is a paired dev diagnostic.  It does not establish a deployable semantic
gate, full-video performance, TrackEval, HOTA, or ordinary MOT/OVMOT change.
