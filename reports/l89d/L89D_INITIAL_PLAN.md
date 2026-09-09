# L89D initial plan — true full-video timeline repair

Date: 2026-09-08
Project root: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
Execution worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89D`
Luna thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`
Base commit: `613fbe6d5e802fbc08d3ef0b5084f5d6b8337cd5`

## Fixed evidence and question

L89C corrected the candidate-vs-NULL equation and produced a valid fixed
16-calibration/24-validation diagnostic, but its internal TrackEval result is
not valid for architecture comparison: predictions were traversed over sparse
evaluation group keys while GT was materialized over every native L69 frame.
The only L89D variable is the frame universe.  L89D will use every native L69
frame for every legal query sequence, while retaining the frozen L89 model,
checkpoint, rule, bank, language cache, loss, and all selection thresholds.
This is a zero-training evidence repair, not a new model experiment.

## Frozen inputs

All paths below are read-only unless explicitly marked as a new L89D output.
The repository worktree supplies the frozen L89C source; external generated
assets are referenced by absolute path so no large files are copied into this
worktree.

| input | path | contract |
|---|---|---|
| fixed manifest | `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/outputs/l19/protocol/kitti_fast_eval_manifest.json` | SHA256 `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa` |
| base Z1 cache | `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/outputs/l85/features/fit_dev_eval_full_attempt2` | label-free Z1 cache, 1623 groups, 115,651,514 bytes |
| L89 language cache | `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/cache/language_tokens_retry1` | frozen label-free language cache |
| L89C dev selection | `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89C/outputs/l89c/dev/final_selection_attempt1/checkpoint_selection.json` | frozen epoch 2 / Rule R; no validation re-selection |
| L89 checkpoint | `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/train/joint40/checkpoint_l89_epoch002.pt` | SHA256 `f8b175597ece8aad1f0ec3ae9d05c70e0759a0f7480dff9af6ea2619a2e3f08b` |
| L69 bank | `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/outputs/l69/attempt9/budget40_features/kitti` | native `frame_ids/frame_ptr`, all rows retained |
| L82 split | `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/outputs/l82/protocol/fit_video_train_dev_split.json` | dev video/group scope |

The legal full-video scopes are V1 videos `0004,0018` and V2 videos
`0016,0017,0020`; dev uses only L82 dev videos from fit units.  No screening
or official-test labels will be read.  The fixed manifest is re-hashed before
every executable phase.

## New outputs and execution order

Only the following new source files are authorized: the nine `tools/l89d_*`
files specified by the L89D prompt.  Reports and artifacts are under
`reports/l89d/` and `outputs/l89d/`.  A dense Z1 supplement, if required, is
written only under `/data2/usr_for_deadline/locatemot_l89d/` and is a compact,
label-free derived cache; it is not copied into the repository.

1. Implement the shared native-timeline, cache-resolver, provenance, and
   boundary contracts.
2. Compile new scripts and run the boundary guard.
3. Run CPU/read-only dense coverage audits for dev and internal.  If either
   scope is incomplete, build only the audited missing groups with the proven
   L85 capture, finalize, and rerun both audits.
4. Run one true-timeline dev smoke (one video/frame and at most two queries),
   then formal all-dev true full-video inference over every native frame and
   legal query.  Run the new TrackEval matrix and freeze the registered dev
   selection without rereading or refitting fixed validation.
5. Run the unchanged fixed semantic evaluator with the new L89D selection.
   Then run internal true full-video inference and the new TrackEval matrix
   over exactly the five legal validation videos and the expected 86 V1 / 537
   V2 query sequences.
6. Generate the final L89D report and boundary/provenance checks, then commit
   and push this branch to GitHub before returning the evidence.

## Timeline and data contracts

The frame universe is loaded from each L69 bank's native `frame_ids`, sorted
and unique.  For each legal video, every legal query is paired with every
native frame.  The expected pair count is therefore
`native_frame_count * legal_query_count`; actual scored pairs must equal it.
The native frame-id list is hashed in sorted JSON form.  Candidate rows are
constructed from each frame's own `frame_ptr` slice, in native row order, and
the immutable row key retains the bank path and row offset.  No L49 sparse
`begin/end`, top-k, NMS, query/source shortcut, candidate deletion, or
truncation is permitted.  `history_frame_id <= current_frame_id` is asserted.

The layered resolver accepts a complete base item or a complete supplement
item for a native group.  A supplement may cover only missing groups, but
each selected item must contain every legal query exactly once, matching
sentences, candidate count, row offsets, `[Q,N,256]` Z1 and finite text/frame
globals.  It never zero-fills a missing or partial group.  If a full group is
present in both layers, the overlap is compared with `allclose(atol=rtol=2e-3)`
before inference.

## Immutable branch/selection rules

The L89C source shortlist is used only for dev source inference.  Its existing
epoch-2/8/20/32/40 checkpoint/rule fits and source SHA are verified; L89D does
not refit rules.  The new selector uses exactly

```text
(HOTA, DetA, AssA, distinct_target_recall,
 -inactive_false_acceptance, -epoch, deterministic_rule_order)
```

with B > R > P only as the final deterministic rule tie-break.  Internal uses
exactly the resulting single checkpoint/rule.  Fixed semantic evaluation is
performed by the unchanged old evaluator after selection is frozen; it remains
separate from dev selection and internal TrackEval.

## Stop/validity thresholds

Any boundary, manifest, native timeline, cache parity, candidate row, strict
checkpoint, or label-boundary violation is a hard `INCOMPLETE`/`INVALID` stop;
only the first actionable cause may be repaired in a new attempt.  A dev
smoke is not accepted by the TrackEval wrapper as full-video evidence.  The
formal inference summary may claim `full_video=true` only when all legal
videos, native frames, legal queries, and expected pairs are present and the
timeline proof is true.  A missing pair is a hard error, not an empty result.

L89D does not alter the semantic gate.  The fixed semantic result is reported
as pass/fail using its registered output, without changing thresholds or using
screening labels.  Internal TrackEval is valid only if its timeline proof and
sequence counts pass; otherwise the final report must say incomplete and make
no final HOTA comparison.  No screening, official test, further training,
HOTA selection on validation, MOT, or OVMOT work is authorized by this stage.

## Resource and safety record

Before cache construction, record `df -h /data1 /data2` and one `nvidia-smi`
check.  Use at most four GPUs for a supplement, only when available; start
with the minimum actual idle allocation.  Query batch 8 may be reduced only
on an observed OOM, in order 8→4→2→1.  No detector/feature cache is written
to `/data1`; inference streams one native frame group at a time and releases
tensors.  All long commands use one blocking shell invocation and are awaited
to completion.

## Required status flags

Every machine-readable output must retain:

```json
{
  "zero_training": true,
  "checkpoint_weights_changed": false,
  "new_checkpoint_created": false,
  "screening_gt_used": false,
  "official_test_labels_read": false,
  "ordinary_mot_ovmot_touched": false
}
```

The report will clearly separate label-free coverage, dev TrackEval
selection, fixed calibration/validation, and internal validation-scope
TrackEval.  It will end at
`L89D_COMPLETE_ZERO_TRAINING_TRUE_FULLVIDEO_REPAIR / <semantic gate pass/fail> /
STOPPED_PENDING_SUPERVISOR_REVIEW` and will not close the Luna task.
