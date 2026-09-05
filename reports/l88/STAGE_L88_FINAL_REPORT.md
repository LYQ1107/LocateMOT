# Stage L88 Final Report — RMOT-Aware Grounding Adaptation

Date: 2026-09-04  
Luna thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`  
Project root: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`

## Executive result

L88 completed its registered RMOT-only path: contract smoke, distributed and
temporal regressions, 40-epoch V1/V2 joint training, all-checkpoint dev
scoring, video-disjoint full-video dev selection, fixed 16-calibration/24-
validation semantic evaluation, and internal full-video V1/V2 TrackEval.

The selected method is GroundingDINO LoRA plus the frozen L86/L87 sidecar,
checkpoint epoch20, Rule B. The fixed semantic gate **failed** because recall
and multi-positive recall collapsed despite better precision, FP/frame,
predictions/positive, and hard-violation numbers. Internal TrackEval also fell
below the L86 and L87-A references. L88 is therefore not ordinary-level RMOT
completion and is stopped pending supervisor review.

## Evidence summary

| layer | authoritative evidence | result |
|---|---|---|
| label-free/fit contract | `outputs/l88/audit/contract_smoke_attempt8/` and distributed/temporal retries | finite, frozen base, nonzero LoRA/sidecar gradients, reload and keys pass |
| fit training | `outputs/l88/train/joint40_world4_retry1/` | 40 epochs, 2,640 optimizer steps, 20 checkpoints complete |
| dev selection | `outputs/l88/dev/final_selection_attempt1/` | epoch20/Rule B selected before fixed validation |
| calibration/validation | `outputs/l88/eval/fixed_semantic_attempt3/` | semantic gate fail; recall 0.1935, multi-positive 0.1667 under final rule |
| internal TrackEval | `outputs/l88/internal/trackeval_matrix_attempt2/` | HOTA 26.0914 V1 / 20.2386 V2 under selected Rule B |
| screening/official test | not run | no evidence |

## Interpretation

The experiment gives a clean negative result for the registered L88 hypothesis
at this capacity and protocol. It demonstrates that the adapted GroundingDINO
layers can receive gradients and produce finite candidate/temporal outputs,
but does not demonstrate a stable expression-to-target correspondence signal
on held-out V1/V2. The primary failure is positive-bag membership and
query-to-candidate generalization, with V2 the clearest stress case. Lower
output volume and higher precision are not sufficient because recall and
multi-positive preservation are mandatory gates.

The L69 candidate bank and L88 temporal fields were not altered. Candidate
coverage remains a separate upper-bound consideration; it is not a license to
attribute every missed target to language correspondence. Sequence-level
TrackEval confirms that the frame-level failure is not repaired by the L88
sidecar.

## Boundary and reproducibility flags

All authoritative L88 artifacts record:

```text
screening_gt_used=false
official_test_labels_read=false
ordinary_mot_ovmot_touched=false
candidate_deletion=false
candidate_truncation=false
groundingdino_lora_used=true
groundingdino_backbone_trainable=false
bert_body_trainable=false
bbox_head_trainable=false
decoder_layers_2_to_6_trainable=false
token_span_region_alignment=UNALIGNED
static_motion_alignment=UNALIGNED
```

The fixed manifest remained SHA256
`06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
Ordinary MOT, OVMOT/TAO, UIDM, historical banks/checkpoints, and their
entrypoints were not modified.

## Artifact index

- preregistration: `reports/l88/L88_PREREGISTERED_PLAN.md`;
- literature and code audit: `reports/l88/L88_LITERATURE_AND_CODE_AUDIT.md`;
- training: `reports/l88/L88_TRAINING_REPORT.md`;
- checkpoint/rule selection: `reports/l88/L88_CHECKPOINT_SELECTION.md`;
- fixed semantic: `reports/l88/L88_FIXED_SEMANTIC_REPORT.md`;
- internal TrackEval: `reports/l88/L88_TRACKEVAL_REPORT.md`;
- runtime deviations: `reports/l88/L88_RUNTIME_DEVIATION.md`;
- failure decomposition: `reports/l88/L88_FAILURE_DECOMPOSITION.md`;
- machine-readable final semantic decision:
  `outputs/l88/eval/fixed_semantic_attempt3/gate_decision.json`;
- machine-readable internal TrackEval:
  `outputs/l88/internal/trackeval_matrix_attempt2/trackeval_matrix.json`.

## Final status

`STOPPED_PENDING_SUPERVISOR_REVIEW`

The only next action is supervisor approval of one new structural
correspondence/proposal experiment. No L88 continuation, screening, official
test, or ordinary MOT/OVMOT regression is started automatically.

