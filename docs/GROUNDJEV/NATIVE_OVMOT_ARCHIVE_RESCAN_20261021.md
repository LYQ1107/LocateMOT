# Native OVMOT archive rescan — 2026-10-21

This is a targeted read-only rescan of `/data2/user/LocateMOT_GroundJEV/outputs/groundjev` before opening the corrected three-task training queue. It inventories the TAO candidate outputs that actually exist and checks their split, terminal receipt, candidate count, and native-feature contract.

The result is unchanged at the decision boundary: **no complete TAO TRAIN native candidate archive was found**. The corrected queue remains `PAUSED_FOR_NATIVE_OVMOT_ROWS` with zero optimizer updates. No validation output was relabeled as TRAIN and no teacher-only fallback was promoted to native training data.

| Archive | Evidence found | Eligibility |
|---|---|---|
| `multitask_candidate_controls/tao_exact_geometry_v1` | 4 `frames/val` rows; geometry grouping only; native hidden features and per-box scores were not exported | Ineligible |
| `multitask_candidate_controls/tao_scored_queries_v1` | 4 `frames/val` rows, 1,773 retained query candidates; GT-free query control, but no exported native per-box feature/score fields | Ineligible |
| `multitask_runs/tao_base_novel63_full_v1/native_candidates` | Empty candidate file; terminal status `FULL_BASE_NOVEL_VIDEO_FAILED` | Ineligible |
| `multitask_runs/tao_base_novel63_full_v2_expression_none/native_candidates` | 40 `frames/val` rows, 35,228 candidates, 1,885,903,934-byte feature cache; legal validation stream only | Ineligible for TRAIN |
| `multitask_runs/tao_exact_geometry_alias40_v1` | 40-frame `frames/val` exact-geometry control | Ineligible for TRAIN |
| `multitask_runs/tao_base_novel63_full_v3_empty_crop_cpu` | 40-frame legal validation-video control | Ineligible for TRAIN |
| `training_data/tao_native_ovmot_train_scope_v1` | 18,274 TRAIN images from 500 videos registered, but native export was never started and `native_ovmot_training_rows=false` | Scope only |

The strict join gate therefore remains correct. The next valid transition is to run the native TAO TRAIN exporter during an exclusive GPU window, verify that it emits real native candidates without GT labels or fabricated scores, and only then rebuild the joint MOT/RMOT/OVMOT corpus. The large validation stream is retained as validation evidence and is not reused for training.

Evidence details are in the JSON companion and the earlier cross-disk audit `NATIVE_OVMOT_ARCHIVE_AUDIT_20261021`.
