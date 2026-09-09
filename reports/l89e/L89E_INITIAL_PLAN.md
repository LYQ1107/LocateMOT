# L89E initial plan — phase-consistent temporal/history replay

Date: 2026-09-09
Project root: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
Execution worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89E`
Luna thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`
Base L89D commit: `d0960e1d1679414765f79203bef444f0f9968928`
Planned branch: `codex/l89e-phase-consistent-temporal-replay-20260909`

## Scope and scientific question

L89E is a zero-training protocol correction.  It changes no L89 model, loss,
candidate bank, tracker, threshold formula, UIDM, ordinary MOT, OVMOT or
production entrypoint.  The only question is whether the L89 QSC-D result
survives when every checkpoint is evaluated with the temporal flag and history
tensor contract used during its training phase.

The confirmed training contract is:

| epochs | phase | temporal flag | history |
|---|---|---:|---|
| 1–8 | S | off | all-zero, mask false, frame id -1 |
| 9–20 | T | on | last four causal observations packed into 8 slots |
| 21–40 | J | on | last four causal observations packed into 8 slots |

`tools/l89_train_full_rmot.py` defines this phase schedule, and
`locatemot/rmot/l86_clip_data.py::_clip_history` defines the tensor packing.
The L89 model consumes history in `candidate_prior` even when temporal
delta/gate output is disabled, so disabling only the flag would be incorrect.
L89D used `temporal_enabled=True` for every checkpoint and directly reused the
raw 8-slot history.  Therefore its timeline is valid but its phase/history
contract is inconsistent.  L89E rescoring is limited to S epochs 2/4/6/8;
the historical T/J sparse rows are reused because their existing
`build_frame(..., temporal_enabled=True)` path already supplied last-four
history.

## Frozen evidence and inputs

- Fixed manifest:
  `outputs/l19/protocol/kitti_fast_eval_manifest.json`, SHA256
  `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
- L89 even checkpoints: read-only files under
  `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/train/joint40`
  (epochs 2 through 40); S replay uses only 2/4/6/8.
- Frozen base Z1 cache:
  `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/outputs/l85/features/fit_dev_eval_full_attempt2`.
- Frozen L89 language cache:
  `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/cache/language_tokens_retry1`.
- Frozen legal dense supplements are reused, never rebuilt:
  `/data2/usr_for_deadline/locatemot_l89d/supplement_dev_full_attempt1` and
  `/data2/usr_for_deadline/locatemot_l89d/supplement_internal_full_attempt4`.
- Historical T/J sparse scores:
  `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/eval/dev_scores_joint40/score_records.jsonl`.
- L89D true-native timeline evidence and coverage audits are read-only.  The
  L89C/L89D report's older base-commit text is historical; this plan records
  the actual L89D code base as `d0960e1`.

## New files and protocol

Only the following new source names are permitted: `tools/l89e_*`.  Reports
and machine outputs are confined to `reports/l89e/**` and `outputs/l89e/**`.
The new `l89e_phase_policy.py` is the single phase/history source of truth and
will call the frozen `_clip_history(..., length=4)` implementation, then
assert S zero history or T/J last-four causal history.  No runtime import of
the training script's phase helper is used.

The Stage-S rescorer will produce exactly 4 checkpoints × 138 legal dev groups
and 498 records per checkpoint (1,992 total), using fit/dev rows only.  The
merge will discard old S rows, retain old T/J rows for epochs 10–40, and
produce exactly 20 epochs × 498 records (9,960 total).  Unchanged corrected
candidate-vs-NULL B/R/P rules will be refit on fit/dev rows only to create a
maximum-five registered shortlist containing fixed epochs 8/20/40 plus the
registered best-rule candidates after deduplication.

The L89D native timeline/resolver, row ordering, candidate-vs-NULL emission,
GT materialization ordering and TrackEval math are reused unchanged in new
L89E wrappers.  The new runner will pass the selected checkpoint's policy to
both `history_for_batch` and `model(..., temporal_enabled=...)`.  Dev selection
will use the registered HOTA/DetA/AssA/distinct-target/inactive/epoch/rule
tuple on true-full-video dev only.  Fixed semantic uses the frozen selection,
then attaches the 16 calibration labels before its registered threshold/rule
report and the 24 validation labels afterward.  Internal TrackEval uses one
frozen selected checkpoint/rule over all native internal frames.

## Required gates and stopping rule

Before formal replay: compile all L89E tools, run the boundary guard, and run
one CPU-only S/T history equality check against `L86ClipStore`; no model
forward is used in this check.  After that check passes, no additional
testing is planned before the formal replay.

The semantic gate remains unchanged: recall ≥ `.7233333`, precision ≥
`.0830188679`, FP/frame ≤ `11.125`, predictions/positive ≤ `4.069`, hard
violation ≤ `.8666667`, multi-positive recall ≥ `.7894444`, inactive false
acceptance < 1, and complete finite rows.  This is a fixed semantic gate, not
HOTA.  Full-video TrackEval is valid only after native timeline and policy
checks pass; it is internal validation-scope evidence, not screening or
official-test evidence.

No screening labels, official-test labels, training, optimizer, checkpoint
creation, cache rebuild, LoRA, new head, threshold grid, tracker change,
ordinary MOT/OVMOT benchmark or production modification is authorized.  On
completion the status will be
`STOPPED_PENDING_SUPERVISOR_REVIEW`; if a new structural study is needed it
will be requested separately rather than started here.

## Resource and provenance flags

Use one available GPU where necessary, `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, and one blocking command per long replay.  Existing dense
Z1 supplements are reused; no new dense/raw cache is written.  Every output
will record `zero_training=true`, `new_checkpoint_created=false`,
`checkpoint_weights_changed=false`, `screening_gt_used=false`,
`official_test_labels_read=false`, `ordinary_mot_ovmot_touched=false`, and
the applicable `hota_trackeval_run` value.  Token/span-to-region and
static/motion alignment remain `UNALIGNED`.
