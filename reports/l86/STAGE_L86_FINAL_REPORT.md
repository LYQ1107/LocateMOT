# LocateMOT Stage L86 — final execution report

## Final status

`full_rmot_hota_complete_improved`

L86 completed the authorized faithful factorized full-RMOT repair: contract
smoke, 40-epoch V1/V2 fit, dev-only checkpoint/rule selection, fixed semantic
validation, full internal V1/V2 inference, and local TrackEval. The internal
full-video HOTA descriptor improved by more than three points in both domains,
but the fixed candidate semantic gate failed on held-out recall and
multi-positive preservation. This is not final ordinary RMOT completion and
does not authorize screening or official testing.

## Git/base/branch

- Project: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
- Luna thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`
- Branch: `codex/l86-faithful-full-rmot-repair-20260903`
- Base L85 commit: `d54ffaa51a9c3e123fcd59fca0828a764a92ff3f`
- Fixed manifest SHA256: `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`

## Frozen assets and changes

The L69 budget-40 bank, L84 Z1 representation, L85/L49 data contracts, shared
UIDM, old checkpoints, TrackEval source, and ordinary MOT/OVMOT entrypoints
were treated as read-only. New L86 code is confined to the L86 namespace. No
GroundingDINO LoRA, detector change, candidate-bank change, raw/dense cache,
or old-score semantic input was used.

The three L85 implementation issues corrected by L86 were: duplicate-sensitive
row ranking instead of unique target-bag supervision; coupled candidate/
presence/NULL energies; and history-length pseudo-target training of a sigmoid
gate. L86 instead uses unique target bags, independent candidate/presence/
NULL energies, and causal target-identity pairs.

## Semantic-oracle TrackEval

The GT-privileged oracle is separate from learned L86. It uses the same legal
internal candidate rows and target-consistent diagnostic assignment, without
changing boxes. Oracle HOTA was V1 `66.8724` (AssA `72.8335`) and V2 `53.9669`
(AssA `61.2163`). These are upper-bound diagnostics, not model performance,
not official test results, and not evidence of deployable target identity.

## L86 architecture and objective

`L86FullRMOT` has 2,061,413 trainable parameters, hidden size 256 and causal
history length 8. It consumes frozen Z1, text/frame global inputs and the
1,432-D L69 observation vector. It emits static and temporal semantic energy,
a centered query-independent candidate prior, independent presence and NULL
logits, temporal gate/delta/state, and history state. All current candidate
rows remain in the output. IDs are used only by the data index to assemble
causal history and are not semantic features.

The fixed loss is
`1.00 semantic_total + .30 semantic_static + 1.00 membership + .50 presence
+ .50 NULL + .10 temporal_identity + .01 delta_regularization`. It includes
all positive target bags, inactive/no-match supervision, explicit
present-uncovered masking, causal real target identities, and the documented
all-negative fallback because same-class metadata is unavailable.

## Minimal contract and training

The authoritative contract is
`outputs/l86/audit/contract_smoke_attempt6/contract.json`: finite loss and
gradients, duplicate target-bag behavior, causal history, complete rows,
strict reload, and frozen input checks passed. The full fit is
`outputs/l86/train/joint40/`, 40/40 epochs, S/T/J phases, seed `20260829`,
actual world size 3, effective clip batch 9, BF16 after smoke, and 2,360 local
optimizer steps. Every epoch summary was finite and every epoch saw both
domains and all four strata. The final checkpoint is step 2,360 with SHA256
`a87b076692798020857e86cb7d291103b84cb33c0355b8e2325f78ac09423552`.

## Dev selection and fixed semantic validation

Cheap dev scoring covered 138 internal dev groups and all 20 even checkpoints
(9,960 row records); full-video dev HOTA was unavailable. The frozen choice
was epoch 14 / step 826, checkpoint SHA256
`b9a6d659e6b5315696370f5a8350f1abce1716eec8ccd8e12244f339b3e26be5`, Rule B,
candidate threshold `0.75`, presence threshold `0.0`, NULL margin `0.0`.
This was frozen before fixed validation labels were read.

The authoritative fixed semantic output is
`outputs/l86/eval/fixed_semantic_attempt2/`. The L29 control is directly
recomputed from accepted immutable records; L86 scores all 40 fixed units and
retains all candidate rows. Final frozen-rule validation is:

| metric | L29 | L86 | gate |
|---|---:|---:|---|
| recall | 0.7333333 | 0.3225806 | FAIL |
| precision | 0.0830189 | 0.1176471 | pass |
| FP/frame | 10.1250 | 3.1250 | pass |
| predictions/positive | 8.8333 | 2.7419 | pass |
| hard violation | 0.9166667 | 0.8461538 | pass |
| multi-positive recall | 0.8194444 | 0.2638889 | FAIL |
| inactive false acceptance | 1.0000 | 0.8333333 | pass |

Candidate-only L86 recall was `.3548387` and candidate-only inactive false
acceptance was `1.0`; the final frozen presence/NULL rule lowered volume but
did not restore recall. The simultaneous decision is `semantic_gate_fail`.
There was no top-k, NMS, candidate deletion, or threshold rescue.

## Full-video validation HOTA

Full inference scored 15,576,721 candidate rows across 243,550 query-frame
records and emitted predictions only after the strategy was frozen. The
authoritative learned TrackEval output is
`outputs/l86/trackeval/fullvideo_eval_attempt1/trackeval_summary.json`.

| dataset | sequences | HOTA | DetA | AssA | LocA | DetRe | DetPr | IDF1 | IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Refer-KITTI V1 internal | 86 | 29.1663 | 20.5210 | 41.8054 | 91.1238 | 69.5047 | 22.3567 | 23.3188 | 1,814 |
| Refer-KITTI V2 internal | 537 | 21.6467 | 13.3584 | 35.2978 | 90.0181 | 42.4989 | 16.1970 | 17.7058 | 11,873 |

L85→L86 deltas are V1 HOTA `+4.1115`, DetA `+5.5357`, AssA `-0.6488`,
DetRe `+22.0919`, DetPr `+4.5167`, IDF1 `+1.9846`, IDSW `-516`; V2 HOTA
`+4.3543`, DetA `+3.5705`, AssA `+4.4589`, DetRe `-3.4503`, DetPr `+5.1979`,
IDF1 `+4.7628`, IDSW `-6,501`. Thus the pre-registered material-fullvideo
descriptor is true, while semantic correspondence remains below its gate.

## Root-cause decomposition

The first actionable bottleneck is held-out query-to-candidate emission,
especially multi-positive target coverage and presence calibration. L86's
lower output volume improved precision/DetPr and internal HOTA, but it is not
the same as correspondence success. The candidate/oracle ceiling and zero
candidate deletion rule exclude a simple proposal-volume explanation; the
remaining learned semantic output does not preserve enough positives.

## Published-method context and limitations

L86's structural context and revision-level provenance are in
`L86_2025_2026_RMOT_COMPARISON.md`. TempRMOT, DKGTrack, FlexHook, ReferDINO,
STORM, COAL and GroundingDINO/MMDetection are references only; no external
weights or code path was imported. The local MMDetection/TrackEval checkouts
do not have independently verifiable HEADs where noted. The frozen-bank
protocol is not a leaderboard-equivalent reproduction of those systems.
Token/span-to-region and static/motion alignment remain `UNALIGNED`.

## Evidence boundaries and unique next action

| evidence bucket | result |
|---|---|
| oracle | complete, GT-privileged internal ceiling only |
| fit smoke | complete and passed |
| 40-epoch fit | complete |
| calibration/dev selection | complete before fixed validation |
| fixed semantic validation | complete; simultaneous gate failed |
| full internal TrackEval | complete for V1/V2; HOTA above is internal validation |
| screening | not run |
| official test | not run |
| ordinary MOT/OVMOT | untouched; no regression run |

Flags: `screening_gt_used=false`, `official_test_labels_read=false`,
`ordinary_mot_ovmot_touched=false`, `hota_trackeval_run=true` only for the
separate internal TrackEval output, `z1_representation_changed=false`,
`groundingdino_lora_used=false`, `token_span_region_alignment=UNALIGNED`, and
`static_motion_alignment=UNALIGNED`.

The unique next action is supervisor review of the improved internal HOTA,
failed semantic gate and oracle ceiling, followed by one explicitly approved
RMOT-only hypothesis. Do not start L86-R1, screening, official testing,
ordinary MOT/OVMOT regression, or any unregistered threshold/loss/tracker
change automatically.

## Machine pointers

- Final status: `outputs/l86/final_status.json`
- Training provenance: `outputs/l86/train/joint40/provenance.json`
- Dev selection: `outputs/l86/eval/dev_selection_attempt1/checkpoint_selection.json`
- Semantic: `outputs/l86/eval/fixed_semantic_attempt2/semantic.json`
- Learned inference: `outputs/l86/trackeval/fullvideo_validation_attempt1/summary.json`
- Learned TrackEval: `outputs/l86/trackeval/fullvideo_eval_attempt1/trackeval_summary.json`
- Oracle TrackEval: `outputs/l86/trackeval/semantic_oracle_eval_attempt2/trackeval_summary.json`
