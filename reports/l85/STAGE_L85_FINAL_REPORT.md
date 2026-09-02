# LocateMOT Stage L85 — Final Execution Report

## Executive decision

L85 completed the authorized RMOT-only full-video experiment and produced a
valid internal validation TrackEval result, but the fixed semantic gate
failed. The branch is stopped below the deployable RMOT gate; it is not
project completion and does not authorize screening, official-test evaluation,
or ordinary MOT/OVMOT changes.

Machine-readable status: `outputs/l85/final_status.json`.

## Scope and frozen inputs

The experiment used the L84-selected Z1 fixed-reference decoder
representation, the immutable L69 budget-40 candidate bank, L49 V1/V2
expression-level fit supervision, and the fixed manifest
`06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
Candidate rows remained complete and query-independent. No L29/L70/L73 score,
source/pool/group/query/track ID or test-only information was used as a neural
semantic input.

No screening/official-test labels were read. Ordinary MOT, OVMOT, TAO, UIDM,
PBD and legacy TrackEval entrypoints were not modified. Fine-grained
token/span-to-region and static/motion alignment remain `UNALIGNED`.

## Architecture and training

The factorized model contains static representation energy, an eight-observation
causal history correction, a centered query-independent candidate prior, a
query/frame presence term, candidate membership energy, a diagnostic NULL
term, and temporal auxiliary loss. The compact adapter has 1,795,174
parameters. It preserves all candidate rows and supports multi-positive target
bags.

The 100-step smoke and forward/loss contracts passed. The registered S/T/J
40-epoch run completed 20,960/20,960 finite and nonzero-gradient steps with
strictly reloadable epoch checkpoints. The long run used one GPU0 process due
to an unrelated GPU1 occupant; the four-GPU one-step DDP contract passed. The
actual S/T/J trainer uses causal per-row histories rather than a separate
batched clip-4 tensor, which is disclosed in
`L85_ARCHITECTURE.md` and `L85_TRAINING_REPORT.md`.

## Calibration and fixed semantic validation

Dev selection froze epoch09/step4716 and the global rule candidate threshold
`1.2`, presence threshold `-0.1`, null margin `0.0` before fixed validation.
The 24-unit validation output was:

| Metric | L85 | Requirement | Gate |
|---|---:|---:|---|
| Recall | 0.4193548 | >= 0.7233333 | FAIL |
| Precision | 0.1300000 | >= 0.0830189 | pass |
| FP/frame | 3.6250 | <= 11.125 | pass |
| Predictions/positive | 3.2258 | <= 4.069 | pass |
| Hard violation | 0.7692308 | <= 0.8666667 | pass |
| Multi-positive recall | 0.4861111 | >= 0.7894444 | FAIL |
| Inactive false acceptance | 1.0000 | < 1.0 | FAIL |

The result is a simultaneous `semantic_gate_fail`. Precision and volume
improved, but recall, multi-positive coverage and inactive/no-match behavior
failed. No threshold, top-k, NMS or candidate deletion was used to manufacture
the result.

## Full-video validation HOTA

Once the semantic selection was frozen, full-video inference and local
TrackEval were run on internal validation videos only. The authoritative output
is `outputs/l85/trackeval/trackeval_attempt4/`.

| Dataset | Sequences | HOTA | DetPr | DetRe | AssA | IDF1 | IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|
| Refer-KITTI V1 | 86 | 25.0548 | 17.8400 | 47.4128 | 42.4542 | 21.3342 | 2,330 |
| Refer-KITTI V2 | 537 | 17.2924 | 10.9991 | 45.9492 | 30.8389 | 12.9430 | 18,374 |

This is internal full-video validation HOTA, not a screening or official-test
score. It is not compared as a published ordinary-RMOT result and was not
used to select the checkpoint.

## Evidence separation

| Evidence type | Status |
|---|---|
| Candidate oracle | Internal validation upper bound only; not a model result |
| Fit smoke | Passed |
| Full fit | 40 epochs complete |
| Calibration/dev selection | Complete before fixed validation |
| Fixed semantic validation | Failed simultaneous gate |
| Full-video validation TrackEval | Complete; HOTA above is internal validation |
| Screening | Not run |
| Official test | Not run |
| Ordinary MOT/OVMOT regression | Not run |

## Literature and reproducibility

The structural context and source revisions are listed in
`L85_2025_2026_RMOT_COMPARISON.md`. The actual implementation reused only the
local audited GroundingDINO/L69/L84 interfaces and local TrackEval API; public
methods were structural references, not claimed reproductions. The local
MMDetection checkout has no verifiable local Git HEAD; the public v3.3.0
reference is recorded separately. The harmless checkpoint warning about
`language_model.language_backbone.body.model.embeddings.position_ids` is
retained as provenance.

## Final status and next action

Status: `full_rmot_hota_complete_below_semantic_gate`.

The unique next action is supervisor review and explicit authorization of one
new RMOT-only correspondence/presence-calibration hypothesis. Do not
automatically extend L85 or start screening/official testing. The overall
LocateMOT objective remains open.

```json
{
  "screening_gt_used": false,
  "official_test_labels_read": false,
  "ordinary_mot_ovmot_touched": false,
  "no_hota_or_trackeval": false,
  "hota_scope": "internal_full_video_validation_only",
  "token_span_region_alignment": "UNALIGNED",
  "static_motion_alignment": "UNALIGNED"
}
```
