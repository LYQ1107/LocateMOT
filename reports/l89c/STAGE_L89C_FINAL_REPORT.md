# Stage L89C final report — corrected candidate-vs-NULL replay

Date: 2026-09-08
Project identity: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
Execution worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89C`
Luna thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`

## Executive decision

`L89C_COMPLETE_ZERO_TRAINING_REPLAY / semantic_gate_fail /
STOPPED_PENDING_SUPERVISOR_REVIEW`.

L89C corrected a real evaluation/deployment contract error and completed the
authorized zero-training evidence chain. It did not rescue L89 into a
deployable RMOT system. The corrected gate passes hard-negative improvement,
recall, precision, and multi-positive floors, but fails FP/frame,
predictions/positive, and inactive/no-match behavior.

## What changed

Only the five L89 evaluation/deployment scripts were corrected, plus an
L89C boundary guard. The trained L89 model, checkpoints, bank, raw feature
cache, language cache, loss, tracker, and ordinary MOT/OVMOT paths were not
changed. The common emission contract is:

```text
candidate_energy >= candidate_threshold
candidate_energy - null_logit >= null_margin
presence_logit >= presence_threshold
```

The corrected dev process selected immutable epoch 2 / Rule R, checkpoint SHA
`f8b175597ece8aad1f0ec3ae9d05c70e0759a0f7480dff9af6ea2619a2e3f08b`, with
thresholds `candidate=-0.75`, `presence=-1.0`, `null_margin=0.0`. Selection
was frozen before fixed validation.

## Evidence table

| evidence layer | authoritative output | result |
|:---|:---|:---|
| implementation/static | `tools/l89c_boundary_guard.py`, compile/assert checks | passed; zero training |
| corrected dev selection | `outputs/l89c/dev/final_selection_attempt1/` | epoch 2 / R frozen before fixed labels |
| fixed calibration | `outputs/l89c/eval/fixed_semantic_corrected_attempt2/` | 16 units; threshold frozen |
| fixed validation | same directory | gate failed: recall .838710, precision .086957, FP/frame 11.375, pred/positive 9.6452, hard .692308, multi .833333, inactive FA 1.0 |
| internal TrackEval | `outputs/l89c/internal/trackeval_corrected_final_attempt1/` | V1 HOTA 2.244718%; V2 HOTA .583059% |
| screening / official test | none | not run |

## Historical comparison

Buggy L89 epoch-4/Rule-R fixed validation was recall `.8709677`, precision
`.0794118`, FP/frame `13.0417`, predictions/positive `10.9677`, hard
`.6923077`, multi `.9027778`, inactive FA `1.0`. L89C is not compared by
pretending the same checkpoint was selected: corrected dev selection changed
the frozen checkpoint to epoch 2 and corrected the threshold contract. The
corrected internal HOTA is slightly below buggy L89 (`2.244718/0.583059%`
versus `2.2906/0.5894%` V1/V2) and remains far below L87-A
(`28.5752/22.1300%`).

## Boundary and next step

All 40 fixed keys and candidate rows are complete and finite; no candidate
deletion or truncation occurred. `zero_training=true`,
`screening_gt_used=false`, `official_test_labels_read=false`, and
`ordinary_mot_ovmot_touched=false`. The internal TrackEval result is not a
screening/official-test result and does not establish ordinary RMOT-level
performance. There is no new training or production change.

The sole next action is supervisor review for one separately authorized branch
focused on absence/volume calibration with candidate correspondence. L89C is
stopped and no further L89C replay is warranted.
