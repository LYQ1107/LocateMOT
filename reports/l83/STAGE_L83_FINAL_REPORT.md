# Stage L83 final report

## Final status

`faithful_target_bag_training_gate_fail`

L83 established a corrected target-bag evidence chain and completed the
mandatory decoder-sharpness diagnostic.  It did not establish deployable
expression correspondence, HOTA, ordinary RMOT performance, or project
completion.

## Corrected baseline and faithful probe

| representation | corrected old bag hard | faithful new bag hard | corrected old hit@1 | faithful new hit@1 | old multi exact | new multi exact | old V2 hard | new V2 hard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L59 fused ROI | 0.794872 | 0.794872 | 0.320513 | 0.285256 | 0.257862 | 0.270440 | 0.774869 | 0.806283 |
| L81 candidate evidence | 0.932692 | 0.919872 | 0.137821 | 0.157051 | 0.119497 | 0.132075 | 0.921466 | 0.942408 |
| L82 candidate reference | 0.807692 | 0.778846 | 0.288462 | 0.336538 | 0.207547 | 0.207547 | 0.832461 | 0.816754 |

The exact G1--G6 decisions are in
`outputs/l83/train/faithful_bag_attempt1/faithful_gate.json`: G6 passed for
all, while every representation failed at least G1 and the complete set of
G1--G5.  Strict reload passed at
`outputs/l83/audit/faithful_reload_attempt1/reload_audit.json`.

## Decoder sharpness

Authoritative output:
`outputs/l83/audit/decoder_sharpness_attempt9/decoder_sharpness.json`.
Z0/Zp/Z1/Z2/Z3/Z4/Z5/Z6 all have complete finite metrics for 138 dev groups.
Z4 is the earliest diagnostic layer satisfying the pre-registered Case-E
thresholds relative to corrected L59 and in V2 (aggregate hard `0.717949`,
hit@1 `0.381410`; V2 hard `0.706806`, hit@1 `0.376963`).  This is a diagnostic
candidate only; the failed faithful gate means no primary representation was
authorized and no downstream factorized/task-composition model was run.

## Phase status table

| evidence type | status | authoritative path |
|---|---|---|
| source snapshot / protocol audit | complete | `outputs/l83/preregister/source_of_truth.json`, `outputs/l83/audit/l82_protocol_mismatch_retry1/` |
| target-bag data/loss contracts | complete | `outputs/l83/audit/target_bag_data_contract_attempt1/`, `outputs/l83/audit/loss_contract_final/` |
| corrected old-probe dev baseline | complete | `outputs/l83/baselines/corrected_old_probe_attempt1/` |
| faithful fit probe | complete, gate fail | `outputs/l83/train/faithful_bag_attempt1/` |
| strict reload | complete | `outputs/l83/audit/faithful_reload_attempt1/` |
| decoder sharpness | complete, diagnostic Z4 | `outputs/l83/audit/decoder_sharpness_attempt9/` |
| factorized energy | not run | faithful gate prerequisite failed |
| large task-composition training | not run | faithful/factorized prerequisites not met |
| historical 16/24 semantic | not run | conditional branch not authorized |
| screening / official test | not run | not authorized |
| TrackEval / HOTA | not run | not authorized |

## Boundaries and comparability

The corrected L83 metrics are video-disjoint fit-derived dev evidence, not the
L29/L64/L65/L66/L70/L71/L72/L73/L74/L75/L77/L78/L79/L80/L81/L82 historical
semantic gates.  No claim is made that target-bag dev scores translate to
HOTA.  L69/L81/L82 banks, L48 text cache, GroundingDINO, UIDM, old checkpoints,
fixed manifest, ordinary MOT, OVMOT/TAO, and TrackEval were left unchanged.
`candidate_deletion=false`, `candidate_truncation=false`, and all decoder
stage rows are finite and complete.  No dense/raw cache was serialized.

`screening_gt_used=false`; `official_test_labels_read=false`;
`hota_trackeval_run=false`; `ordinary_mot_ovmot_touched=false`;
`l81_modified=false`; `l82_modified=false`;
`uidm_shared_checkpoint_modified=false`;
`token_span_region_alignment=UNALIGNED`;
`static_motion_alignment=UNALIGNED`.

## Single next action

`STOPPED_PENDING_SUPERVISOR_REVIEW` — preserve the corrected target-bag and
decoder evidence and wait for supervisor direction.  Do not start factorized
energy, task composition, historical semantic evaluation, screening, or
TrackEval in this stage.
