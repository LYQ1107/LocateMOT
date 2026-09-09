# L89D failure decomposition and evidence disposition

Date: 2026-09-08
Project: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
Worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89D`
Luna thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`

## Primary L89C validity problem

L89C prediction traversal used sparse dev/validation group keys, while its
ground-truth materialization iterated the complete native L69 `frame_ids`
timeline. Its internal HOTA values are therefore invalid as full-video system
evidence, although the L89 training, checkpoint, corrected candidate-vs-NULL
equation, and sparse fixed semantic diagnostic remain historical evidence.

L89D changed only the frame universe: every legal query is paired with every
native L69 frame, and the same native frame universe is used for GT
materialization after prediction. The L89 model, loss, thresholds, bank,
tracker, UIDM and production entries were not changed.

## Implementation/resource failures retained

1. The first L89D dev inference attempt stopped with `ENOSPC` while writing
   its prediction audit. The output is retained at
   `outputs/l89d/dev/true_fullvideo_infer_attempt1` (moved recoverably to
   `/data2/usr_for_deadline/locatemot_l89d/dev_true_fullvideo_infer_attempt1`).
   It is not authoritative.
2. The first internal supplement build exited with code 137 after 1,111
   items, and the first sharded retry exited with code 137 after seven items;
   both are retained outside the repository. The streaming, four-shard retry
   completed 1,844/1,844 internal groups. The dev supplement completed
   2,385/2,385 groups. No frozen asset was removed.
3. The first dev TrackEval attempt stopped at `36/146` for V1 because the
   inference runner's multi-video `query_map` dictionary comprehension
   overwrote earlier videos in the GT/seqmap map. This was the first
   actionable code root cause. A new `_query_map_from_scopes()` helper now
   aggregates all videos per dataset while retaining key-only rows. A
   synthetic multi-video targeted regression passed before rerunning the
   full dev matrix.
4. The first selector invocation passed the TrackEval directory instead of
   its `trackeval_matrix.json`; it was an invocation error and is retained in
   `outputs/l89d/dev/final_selection_attempt1`. The corrected new invocation
   completed in `final_selection_attempt2` without changing selection logic.

## Authoritative post-fix evidence

- Dev dense coverage: 2,385/2,385 native frame groups and 230,538/230,538
  query-frame pairs.
- Internal dense coverage: 1,844/1,844 native frame groups and
  243,550/243,550 query-frame pairs.
- Dev true full-video inference: five frozen checkpoints, six legal dev
  videos, 1,152,690/1,152,690 query-frame pairs; all candidate rows retained.
- Dev TrackEval: 15 complete checkpoint/rule results. The registered
  selector froze epoch 8, Rule P, without threshold refitting.
- Fixed semantic replay: complete 16-calibration/24-validation diagnostic,
  `semantic_gate_fail`; validation recall `0.5484`, precision `0.1518`,
  FP/frame `3.9583`, predictions/positive `3.6129`, hard violation `0.6923`,
  multi-positive recall `0.4861`, inactive false acceptance `0.6667`.
- Internal true full-video inference: 243,550/243,550 pairs over V1 0004/0018
  and V2 0016/0017/0020, one frozen checkpoint/rule.
- Internal TrackEval: V1 HOTA `25.0799%`, V2 HOTA `22.6374%`; combined
  unweighted HOTA `23.8586%`. This is valid internal validation-scope
  TrackEval, not screening or official-test evidence.

## Interpretation

The timeline repair explains the L89C near-zero full-video HOTA: the repaired
result is in the same broad range as L86/L87-A rather than 0--5. It does not
make L89 a semantic-gate pass. The fixed semantic result still loses recall
and multi-positive coverage, while the internal TrackEval result shows a real
remaining detection/volume and association gap. No claim of ordinary-level
RMOT completion is supported.

The root cause for stopping L89D is therefore `semantic_gate_fail` after a
valid full-video repair, not a timeline or checkpoint-integrity failure.

## Boundary and next action

`zero_training=true`; `checkpoint_weights_changed=false`;
`new_checkpoint_created=false`; `screening_gt_used=false`;
`official_test_labels_read=false`; `ordinary_mot_ovmot_touched=false`;
`hota_trackeval_run=true` only for the legal dev-selection/internal-validation
artifacts; no screening or official-test labels were read. Token/span-region
and static/motion alignment remain `UNALIGNED`.

Single next action: supervisor review and, only with separate authorization, a
bounded absence/volume-calibration study. Do not extend L89, retune its
threshold, add NULL filtering, alter the bank/tracker, or touch MOT/OVMOT.
