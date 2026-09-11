# R1 legal-dev replay plan

Date: 2026-09-11
Thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`
Worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_R1`

This continuation is the registered read-only legal-development replay after
the completed R1 formal fit.  It is not screening, official-test evaluation,
or a production RMOT result.

## Frozen inputs and scope

- Frozen L89E Stage-S anchor:
  `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/train/joint40/checkpoint_l89_epoch004.pt`
  SHA256 `5ab3cb344b73b320b34de7b4bb41622ce665ecb17c4d90c1999640a318e69aa8`.
- R1 aligned cache:
  `outputs/r1/cache/eval_attempt2/`; it contains only label-free Z0/Z1/Z4
  tensors and summaries.  The prior process-loss directory
  `outputs/r1/cache/eval_attempt1/` remains historical and is not overwritten.
- Frozen R0A/R1 visual cache and its 14-frame fixed supplement are read-only;
  no detector, CLIP, or GroundingDINO forward is performed by this replay.
- Legal-dev videos are exactly V1 `{0008,0010,0020}` and V2
  `{0000,0008,0009}` from the audited dev indexes.  The forbidden official
  videos `{0005,0011,0013,0019}` are rejected by the index contract.
- All L69 rows are retained in native frame-pointer order.  The replay uses
  `(dataset, video, query_id, frame_id, bank_path, row_offset)` provenance and
  never uses old L49 ranges, top-k, NMS, or candidate deletion.

## Frozen replay and selection protocol

For each dataset, epoch `{1,2,4,6}` R1 adapter checkpoint is loaded strictly;
the frozen anchor is instantiated once and remains outside the optimizer.
Temporal mode is the registered Stage-S zero-history anchor mode.  Rule B is
loaded from the authoritative L89E selection artifact and remains exactly:
candidate energy `>=1.0`, presence `>=0.5`, and energy-minus-null `>=0.0`.
Every native query/frame pair and every candidate row is scored before any
legal-dev target record is opened.  Predictions are written to a separate
large-output root on `/data2`; legal GT is materialized only after all
prediction files are complete.

The checkpoint is selected separately for V1 and V2 with the preregistered
tuple:

`(HOTA, DetA, AssA, distinct_target_recall, -inactive_false_acceptance, -epoch)`.

The first three fields come from the local TrackEval checkout.  The two
descriptor fields are computed after prediction completion from legal safe
targets: target-frame box hit at IoU 0.5 and query-frame inactive acceptance.
Epoch 1 is retained as a preview; epochs 2/4/6 are eligible.  No fixed
calibration/validation or screening label is read for this selection.

## Hard gates and stopping condition

Before the full replay, a one-group regression must prove anchor/sidecar
shapes, finite outputs, Rule-B masks, native row order, and strict checkpoint
reload.  The full replay must complete all expected legal query/frame pairs for
both datasets, produce all four checkpoints per dataset, and keep
`candidate_deletion=false` and `candidate_truncation=false`.  Any first
actionable technical error is preserved in a new attempt and only that error
is repaired.

After legal-dev selection, the selected epoch per dataset is frozen.  Only
then will the R1 fixed 16-calibration/24-validation candidate-bag diagnostic
be run with the L69 rows and the fixed visual supplement.  No screening/test
labels, HOTA claim for the fixed slice, ordinary MOT/OVMOT path, or larger R1
training is authorized by this continuation.

Machine outputs will carry:

`screening_gt_used=false`, `official_test_labels_read=false`,
`ordinary_mot_ovmot_touched=false`, `training_run=false` for replay/selection,
and `hota_trackeval_run=true` only for the legal-dev TrackEval artifact.
