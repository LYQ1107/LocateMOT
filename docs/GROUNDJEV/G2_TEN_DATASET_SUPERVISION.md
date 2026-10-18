# G2 multidomain supervision and coverage

## Clean V2 protocol (2026-10-19)

The clean course is registered in
`/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/g2_clean_train_manifest.json`.
It supersedes the old row counts for new training only; the old manifest and
checkpoint remain immutable evidence. Clean rows are TRAIN-only and use the
following corrected coverage:

| Dataset | Clean rows | Policy |
| --- | ---: | --- |
| Refer-KITTI-V2 | 379,341 | all legal fit labeled frames per expression |
| AerialMind | 69,358 | existing dense/referring official adapter |
| DanceTrack | 41,796 | TRAIN sequences |
| TAO-Amodal Base | 30,781 | one reliable Base category per row; Novel and sparse negatives excluded |
| SportsMOT | 28,574 | TRAIN sequences |
| MOT17 | 15,948 | `mark==1`, `class==1` person boxes |
| AnimalTrack | 11,543 | TRAIN videos |
| MOT20 | 5,616 | MOT20-01/02/03 only; MOT20-05 frozen holdout |
| RMOT26 | 294 | exact normalized source questions and initial target boxes |

Total clean rows: 583,251; rejects: 2,249. BURST remains an explicit zero-row
gap because no legal TRAIN image root is available. The clean manifest records
source/video/image coverage and both row/reject SHA values.

The previous table below describes the historical 199,053-row engineering
corpus. It is retained for reproducibility and is not a clean held-out claim.

The G2 corpus is TRAIN-only. Every emitted row has the official four-key schema `id/messages/images/source`; messages carry canonical GroundAnything prompts and answers, while the collator decodes media online from absolute paths. The row file and source counts are registered in `/data3/user/LocateMOT-runs/GroundAnythingG2/G2_PROJECTOR_QWEN3_LORA_FULLDATA/g2_train_manifest.json`.

| Dataset | Supervision in current corpus | Rows | Coverage decision |
| --- | --- | ---: | --- |
| MOT17 | dense person boxes | 15,948 | TRAIN sequences only |
| MOT20 | dense person boxes | 8,931 | TRAIN sequences only |
| DanceTrack | dense person boxes | 41,796 | TRAIN sequences only, symlink resolved |
| SportsMOT | dense object/person tracking boxes | 28,574 | TRAIN sequences only |
| AnimalTrack | dense species boxes from decoded TRAIN frames | 11,543 | 32 TRAIN videos, test split unopened |
| Refer-KITTI-V2 | referring expressions and target boxes | 4,333 | fit sequences only; official evaluation videos excluded |
| AerialMind | dense object boxes plus referring expressions | 45,625 + 23,733 | image/label intersection audited |
| RMOT26 | referring target boxes from complete annotation rows | 296 | 20 unrelated failed-download entries remain recorded; unusable rows are rejected |
| TAO-Amodal | dense open-vocabulary category boxes | 18,274 | all 18,274 TRAIN images resolved across `/data0`–`/data3` |
| BURST | none yet | 0 | annotations exist, but no legal TRAIN image root is present; excluded to prevent TAO test leakage |

The resulting corpus contains nine datasets and a transparent BURST gap. No test/validation media is substituted for BURST. The excluded rows and causes are written to `g2_train_rejects.jsonl`; source-balanced scheduling can therefore be reproduced without hiding missing data.

## Post-builder contract audit

The clean JSONL was re-audited after the formal owner was launched. The
independent audit reports 583,251 unique rows, zero MOT20-05 rows, zero
placeholder RMOT questions, and no cross-source absolute media-path overlap.
The raw MOT20 checkout still contains the MOT20-05 directory because it is the
evaluation source tree; the clean builder excludes that sequence before row
creation. Evidence is retained at
`/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/g2_clean_data_contract_audit_v2.json`.

## Latest submission audit

A read-only audit receipt is written at
/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/latest_submission_audit_20261019.json.
It verifies the exact clean row SHA and base-model SHA used by the live owner,
with no non-finite, skip, or truncation event detected in the observed G2
events. The audit preserves the explicit BURST gap and MOT20-05 exclusion.
