# L89E — phase-consistent temporal/history replay final report

## Executive decision

`L89E_COMPLETE_ZERO_TRAINING_PHASE_CONSISTENT_REPLAY / semantic_gate_fail / STOPPED_PENDING_SUPERVISOR_REVIEW`

L89E repaired the checkpoint phase/history evaluation contract and completed
the authorized true-full-video dev selection, fixed semantic replay, and
internal V1/V2 TrackEval.  It did not train or change the QSC-D model.

## Source and boundary

- base L89D commit: `d0960e1d1679414765f79203bef444f0f9968928`
- branch: `codex/l89e-phase-consistent-temporal-replay-20260909`
- thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`
- fixed manifest SHA: `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`
- selected checkpoint: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/train/joint40/checkpoint_l89_epoch004.pt`
- selected checkpoint SHA: `5ab3cb344b73b320b34de7b4bb41622ce665ecb17c4d90c1999640a318e69aa8`
- selected epoch/phase/rule: `4` / `S` / `B`
- selected policy: `{"epoch": 4, "history_length": 0, "history_mode": "zero_history", "phase": "S", "temporal_enabled": false}`

The older L89D report contains historical base-commit text from before its
final commit; this report uses the actual L89D base commit above.

## Phase/history correction

| checkpoint | train phase | train temporal | eval temporal | eval history | consistent |
|---|---|---:|---:|---|---|
| 2/4/6/8 | S | off | off | zero | yes |
| 10–20 | T | on | on | last4 causal | yes |
| 22–40 | J | on | on | last4 causal | yes |

Stage-S produced exactly 1992 sparse records.  The
historical T/J score rows were reused unchanged, producing 9960
records over 20 epochs.  No new checkpoint or cache was created.

## True-full-video dev and fixed semantic

The dev matrix has 15 complete TrackEval results.
Selection used only fit/dev TrackEval and the registered tuple, then froze
epoch `4` / Rule `B` before fixed validation.

Fixed validation metrics:

| recall | precision | FP/frame | pred/positive | hard | multi-positive | inactive FA | gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.4516129 | 0.1458333 | 3.4166667 | 3.0967742 | 0.7692308 | 0.3611111 | 0.8333333 | `semantic_gate_fail` |

Recall and multi-positive recall fail the registered floors.  The result is
therefore not a semantic correspondence pass and is not used to authorize a
new training run.

## Internal TrackEval and historical comparison

| evidence | V1 HOTA | V2 HOTA | authority |
|---|---:|---:|---|
| L86 | 29.1663 | 21.6467 | historical full-video comparison |
| L87-A | 28.5752 | 22.1300 | historical full-video comparison |
| L89D phase/history-inconsistent | 25.0799 | 22.6374 | valid timeline, inconsistent phase/history |
| L89E phase-consistent | 28.7628 | 21.8385 | valid internal validation-scope TrackEval |

L89E internal metrics:

- V1: HOTA `28.7628%`, DetA `18.7680%`,
  AssA `44.3927%`, DetRe `80.3045%`,
  DetPr `19.5312%`.
- V2: HOTA `21.8385%`, DetA `13.3358%`,
  AssA `35.9990%`, DetRe `45.1160%`,
  DetPr `15.8179%`.

These are not screening or official-test results.  No HOTA fast-screening
claim is made; they are legal internal TrackEval measurements only.

## Final boundaries and next action

`zero_training=true`, `new_checkpoint_created=false`,
`checkpoint_weights_changed=false`, `screening_gt_used=false`,
`official_test_labels_read=false`, `ordinary_mot_ovmot_touched=false`.
Token/span-region and static/motion alignment remain `UNALIGNED`.

The unique next action is supervisor review of one separately authorized
absence/volume-calibration study.  Do not extend L89/L89C/L89E, change the
threshold, add a new NULL head, rebuild the bank, or modify ordinary
MOT/OVMOT in this stage.

## Artifact paths

- Stage-S: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89E/outputs/l89e/dev/stage_s_rescore_attempt4`
- merged scores: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89E/outputs/l89e/dev/merged_scores_attempt2`
- shortlist: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89E/outputs/l89e/dev/shortlist_attempt4`
- dev true-full-video TrackEval: `/data2/usr_for_deadline/locatemot_l89e/dev_trackeval_matrix_attempt1`
- frozen selection: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89E/outputs/l89e/dev/selection_attempt1`
- fixed semantic: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89E/outputs/l89e/eval/fixed_semantic_attempt1`
- internal TrackEval: `/data2/usr_for_deadline/locatemot_l89e/internal_trackeval_attempt1`
