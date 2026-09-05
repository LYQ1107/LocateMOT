# L88C Initial Plan — Corrected Candidate-vs-NULL Replay

Date: 2026-09-05  
Project root: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`  
Execution worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L88C`  
Luna thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`  
Base L88 commit: `c9b44c07b9b977de9d0f839fb2ff6363abb0386e`  
Branch: `codex/l88c-candidate-null-corrected-replay-20260905`

## Purpose and fixed boundary

L88C is a zero-training correction and diagnosis stage. The confirmed L88
deployment equation compared `presence_logit - null_logit`, although the
registered/trained contract compares each candidate energy with the
query-level NULL logit. L88C will use the corrected candidate-level rule for
dev rule refitting, video-disjoint full-video dev selection, fixed semantic
evaluation, and internal V1/V2 TrackEval. It will not update LoRA, sidecar,
optimizer, or any checkpoint.

Frozen inputs are the L69 budget-40 candidate bank, the existing L88
query-independent encoder cache, all 20 even L88 checkpoints, L88/L87-A/L86
reports and records, and the fixed manifest. The manifest must remain
SHA256 `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
The immutable L88 branch and its outputs remain untouched.

## Corrected emission contract

For candidate energy `e_i`, presence `p`, NULL `n`, candidate threshold `t`_,
presence threshold `t_p`, and NULL margin `m`, a row emits exactly when:

```text
e_i >= t_  AND  e_i - n >= m  AND  p >= t_p
```

The candidate-only diagnostic is exactly `e_i >= t_`; it has no presence or
NULL gate. NULL is never compared with presence. All candidates, duplicate
rows, and background singleton bags remain in the records.

The fixed grid is candidate and presence thresholds
`[-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]`, with NULL margins
`[0,0.25,0.5,0.75]`. Rule B maximizes corrected target-bag F1 with the
registered tie order; Rule R maximizes distinct target recall at precision at
least `.08`; Rule P maximizes target-bag precision at distinct recall at least
`.60`. The L88 shortlist policy and dev TrackEval selection tuple are
unchanged.

## Execution order

1. Inspect all L88 deployment uses and implement only new L88C helpers.
2. Compile the affected helpers and run the single registered two-candidate
   corrected-mask assertion.
3. Refit all 20 dev checkpoint/rule combinations from existing 9,960 cheap
   records, including the L88C-PURE-GATE control at epoch20/Rule B and the
   original thresholds.
4. Re-run legal full-video dev inference/TrackEval for the corrected shortlist,
   then freeze one checkpoint/rule before reading fixed validation labels.
5. Re-run the fixed 16 calibration/24 validation semantic evaluation once.
6. Re-run full internal V1/V2 inference and TrackEval for the frozen corrected
   strategy, then perform only offline analysis of existing and L88C records.

No screening or official-test labels will be read. No new training, backward,
optimizer step, checkpoint update, rank/layer/loss change, external data,
threshold rescue after validation, top-k/NMS, tracker change, or ordinary
MOT/OVMOT operation is permitted.

## Required decisions

The corrected replay will distinguish: (a) whether the deployment bug alone
recovers L88, (b) whether corrected rule/checkpoint selection changes the
result, and (c) whether the adapted representation still fails before or
after gating. If existing evidence does not uniquely identify the next
structural cause, L88C will write `NEXT_TEST_APPROVAL_REQUEST.md` and stop;
it will not run that test.

Every L88C artifact will state:

```text
zero_training=true
corrected_candidate_vs_null=true
screening_gt_used=false
official_test_labels_read=false
ordinary_mot_ovmot_touched=false
candidate_deletion=false
candidate_truncation=false
token_span_region_alignment=UNALIGNED
static_motion_alignment=UNALIGNED
```

