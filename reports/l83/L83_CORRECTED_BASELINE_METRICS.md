# L83 corrected old-probe baselines

Source: `outputs/l83/baselines/corrected_old_probe_attempt1/`.  These are
video-disjoint fit-derived dev diagnostics recomputed with the L83 target-bag
metric contract, not historical 16/24 validation and not HOTA.

| representation/checkpoint | bag hard | bag hit@1 | bag recall@5 | multi-target exact | swap accuracy | true swap AUC | V2 bag hard | inactive false acceptance | score std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L59 fused ROI | 0.794872 | 0.320513 | 0.572025 | 0.257862 | 0.733553 | 0.738203 | 0.774869 | 0.979592 | 1.588553 |
| L81 candidate evidence | 0.932692 | 0.137821 | 0.315240 | 0.119497 | 0.490132 | 0.493500 | 0.921466 | 1.000000 | 0.208992 |
| L82 candidate reference | 0.807692 | 0.288462 | 0.695198 | 0.207547 | 0.754934 | 0.708157 | 0.832461 | 1.000000 | 1.900849 |

The run covered 138 video-disjoint dev groups and produced no persistent dense
feature cache.  The corrected source JSON reports all candidate rows, finite
scores, and no deletion/truncation.  L59 is the strongest old control on hard
violation and hit@1 among these three, while L82 has a modestly lower hard
violation than its own old-probe comparison but does not meet the faithful
gate.

No fixed historical calibration/validation labels, screening labels, or
official-test labels were read here.
