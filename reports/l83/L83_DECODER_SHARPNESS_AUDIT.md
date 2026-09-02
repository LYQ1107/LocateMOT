# L83 GroundingDINO decoder sharpness audit

## Scope and authoritative attempt

The decoder decomposition was required regardless of the faithful-gate
result.  It is a frozen diagnostic using the same 66,561-parameter faithful
target-bag probe, 524 fit groups, 138 video-disjoint dev groups, ten epochs,
seed `20260829`, and four ranks.  It never changes an external L69 row or
produces a refined detector output.  The authoritative result is
`outputs/l83/audit/decoder_sharpness_attempt9/decoder_sharpness.json` with
1,104 compact dev records (8 stages × 138 groups).

Earlier attempts are retained: full attempts 1 and 2 stopped at the
non-overwrite launcher guard; full attempt3 deadlocked at compact distribution
gather; full attempt4 deadlocked at cleanup.  Full attempt5 completed metrics
but its automatic conclusion used the wrong hit@1 field name and is historical
for decision purposes.  Attempt6 was intentionally terminated before data
collection because its launcher PYTHONPATH omitted `vranlee`.  Targeted
attempt7 exercised the corrected field, targeted attempt8 exercised the
corrected V2 rule, and attempt9 is the complete run.

## Stage definitions

`Z0` is the L59 fused-ROI `visual_seed`; `Zp` adds the pretrained reference
position; `Z1`--`Z6` are fixed-reference GroundingDINO decoder layer outputs.
Native iterative refinement `R0`--`R6` is a frozen comparison only.  All
candidate rows remain in their original order; no NMS, top-k, deletion, or
refined-box output is used.

## Aggregate and V2 sharpness

| stage | aggregate bag hard | aggregate hit@1 | bag recall@5 | multi exact | swap acc | true swap AUC | V2 bag hard | V2 hit@1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Z0 | 0.804487 | 0.307692 | 0.578288 | 0.238994 | 0.736842 | 0.737237 | 0.811518 | 0.293194 |
| Zp | 0.833333 | 0.285256 | 0.530271 | 0.232704 | 0.735197 | 0.680288 | 0.863874 | 0.225131 |
| Z1 | 0.708333 | 0.397436 | 0.680585 | 0.295597 | 0.759868 | 0.732208 | 0.764398 | 0.319372 |
| Z2 | 0.740385 | 0.371795 | 0.726514 | 0.264151 | 0.748355 | 0.737548 | 0.759162 | 0.345550 |
| Z3 | 0.772436 | 0.365385 | 0.743215 | 0.213836 | 0.759868 | 0.744982 | 0.816754 | 0.303665 |
| Z4 | 0.717949 | 0.381410 | 0.741127 | 0.257862 | 0.725329 | 0.729995 | 0.706806 | 0.376963 |
| Z5 | 0.750000 | 0.378205 | 0.709812 | 0.245283 | 0.766447 | 0.725472 | 0.801047 | 0.308901 |
| Z6 | 0.788462 | 0.346154 | 0.709812 | 0.188679 | 0.756579 | 0.722208 | 0.821990 | 0.314136 |

The corrected automated conclusion is `selected_semantic_layer=Z4` for the
decoder diagnostic only.  Relative to corrected L59, the pre-registered
Case-E thresholds are aggregate hard ≤ `0.764872`, aggregate hit@1 ≥
`0.370513`, V2 hard ≤ `0.744869`, and V2 hit@1 ≥ `0.369372`.  Z4 is the
earliest stage satisfying all four.  Z1/Z2 miss the V2 hit condition and Z3
misses the aggregate hard condition.  Because the faithful target-bag gate
failed, Z4 is not authorized as a primary semantic branch and no factorized
model may be built from it in L83.

## Distribution and reference diagnostics

The native query embedding has mean norm approximately `2.428` and effective
rank `11.40`.  Per-rank Z0 effective rank is `9.13`--`10.92` with norm ratio
to the native mean `0.134`; Zp effective rank is `3.13`--`3.38`, norm ratio
`2.46`--`2.51`.  Z0 candidate-pair cosine means are `0.482`--`0.493`, while
Zp is `0.688`--`0.709`.  The post-decoder ranges are Z1 `0.659`--`0.729`, Z2
`0.600`--`0.615`, Z3 `0.582`--`0.593`, Z4 `0.573`--`0.576`, Z5
`0.651`--`0.655`, and Z6 `0.628`--`0.636`.  These are distribution-shift
diagnostics, not causal claims.

Zp is worse than Z0 by hard `0.028846` and hit@1 `0.022436`, below the
pre-registered Case-B `0.03`/`0.03` trigger.  Native refinement deltas are
zero at layer 1 and have mean L2 `3.105`, `3.896`, `4.189`, `3.860`, and
`4.271` for layers 2--6; refined references never replace the candidate bank.
The compact score-entropy means are Z0 `0.9374`, Zp `0.9712`, Z1 `0.8728`,
Z2 `0.8254`, Z3 `0.8468`, Z4 `0.8446`, Z5 `0.8977`, and Z6 `0.8991`.
Candidate similarity and entropy vary by stage, so this audit does not assert
causal “self-attention broadening.”

No fixed historical 16/24 semantic evaluation, factorized energy, task
composition, screening, official test, TrackEval, HOTA, ordinary MOT, or
OVMOT run was performed.
