# L88 Failure Decomposition

## Registered decision

`STOPPED_PENDING_SUPERVISOR_REVIEW` with fixed semantic
`semantic_gate_fail`. The 40-epoch fit and the internal full-video TrackEval
were valid, but L88 did not reach the ordinary-RMOT evidence gate.

## What the evidence rules out

This is not an implementation-smoke failure: the contract, distributed
regression, temporal regression, full training, reloads, candidate-row
retention, and provenance checks passed. It is also not a simple output-volume
failure. L88 lowered the selected semantic output volume and improved
precision, FP/frame, and hard violation on the fixed slice, but candidate-only
recall was only 0.4194 and final-rule recall only 0.1935. Multi-positive recall
was 0.3333 and 0.1667 respectively. The apparent precision gain therefore
comes with missed positives and cannot be called a correspondence repair.

## First actionable bottleneck

The first actionable bottleneck is held-out query-to-candidate membership and
multi-positive calibration under the adapted GroundingDINO representation.
The model did not preserve a usable positive bag while suppressing hard
negatives, most severely in V2. The dev selection can improve target-bag
metrics on its video-disjoint fit/dev scope, but the fixed validation and
internal TrackEval evidence do not transfer that improvement. This is a
correspondence/generalization failure, not evidence that more L88 epochs or a
new threshold would solve it.

Candidate coverage remains an independent limitation: L68/L76 showed that the
budget-40 union is not empty and is broadly adequate for testing language
correspondence, but it is not perfect and present-uncovered units remain
explicit. L88 was not allowed to change proposal acquisition, so missing
candidates cannot be recovered by its scorer. Coverage is not sufficient to
explain away the severe positive/multi-positive collapse, and it must not be
silently folded into semantic metrics.

## Evidence classes kept separate

- Contract and fit smoke: valid implementation evidence only.
- 40-epoch training: fit evidence; not a validation result.
- Cheap dev and full-video dev TrackEval: video-disjoint selection evidence;
  epoch20/Rule B was frozen before fixed validation.
- Fixed 16/24 semantic evaluation: valid calibration/validation evidence and
  a failed semantic gate.
- Internal V1/V2 TrackEval: valid full-video internal validation evidence,
  below L86/L87 references; not screening or official benchmark evidence.
- Oracle, screening, and official-test results: not run in L88.

No token/span-to-region or motion-language annotations were verified;
`UNALIGNED` remains the correct status. No HFF/COAL/STORM external checkpoint
or data was imported.

## Unique next action

Do not extend L88, increase rank, tune threshold/top-k/NULL, change the bank,
or run screening. The single next action is supervisor review and explicit
approval of one new RMOT-specific correspondence/proposal design, chosen from
the measured V2 positive/multi-positive bottleneck. Until that approval, all
L88 evidence is retained and the project remains open.

