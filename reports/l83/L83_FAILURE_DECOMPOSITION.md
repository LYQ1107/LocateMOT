# L83 failure decomposition

## Final status

`faithful_target_bag_training_gate_fail`.

## Evidence chain

1. Source-of-truth snapshot and fixed manifest checks passed.  The manifest
   remains SHA256
   `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
2. The L82 protocol mismatch audit passed after identifying row-level
   duplicate-positive supervision, mislabeled AUC, row-top5 accounting, a
   shared self-score path, and fixed-reference decoder input issues.
3. Target-bag data, loss, duplicate, present-uncovered, inactive, query-axis,
   and independent AUC contracts passed.  Candidate rows were complete and
   duplicate candidate indices were retained.
4. Corrected old-probe dev metrics did not show a strong target-sharp
   representation.  L59 remains the strongest simple control on aggregate
   hard/hit among the three old representations.
5. The faithful target-bag probe was finite, nonzero-gradient and reloadable,
   but none of L59/L81/L82 passed G1--G5; only G6 passed.
6. The mandatory decoder decomposition found Z4 as the earliest layer meeting
   the pre-registered aggregate/V2 Case-E diagnostic thresholds.  This is
   useful localization evidence, but it cannot override the failed faithful
   representation gate and is not a semantic gate.

## First actionable root cause

`grounding_representation_target_separation_insufficient` under corrected
duplicate-aware target-bag supervision.  The result does not justify saying
that the detector or all raw representations are exhausted.  It does justify
stopping the conditional factorized/task-composition branches in this stage.

## Not-run boundaries

No factorized energy, large task-composition training, historical 16/24
semantic gate, screening, official test, TrackEval/HOTA, ordinary MOT, or
OVMOT was run.  Token/span-to-region and static/motion alignment are
`UNALIGNED`.

## Unique next action

`STOPPED_PENDING_SUPERVISOR_REVIEW` — retain all L83 evidence and wait for one
new supervisor-approved structural hypothesis.  Do not increase capacity,
change thresholds, add top-k/NULL filtering, alter the bank, or re-run the
failed conditional branches.
