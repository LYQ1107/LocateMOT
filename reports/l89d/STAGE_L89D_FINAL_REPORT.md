# L89D — True Full-Video Timeline Repair Final Report

## Final status

`L89D_COMPLETE_ZERO_TRAINING_TRUE_FULLVIDEO_REPAIR / semantic_gate_fail / STOPPED_PENDING_SUPERVISOR_REVIEW`

This stage repaired the evidence timeline only. It did not change the L89 model, loss, candidate bank, tracker, threshold fits, UIDM, ordinary MOT, OVMOT, or production entrypoints. The fixed semantic gate remains separate from formal internal TrackEval.

## 1. Provenance

- project root: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
- L89D worktree/branch checkout: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89D` / `codex/l89d-true-fullvideo-timeline-repair-20260908`
- Luna thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`
- L89D base commit: `613fbe6d5e802fbc08d3ef0b5084f5d6b8337cd5`
- fixed manifest SHA256: `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`
- L89C shortlist source: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89C/outputs/l89c/dev/final_selection_attempt1/checkpoint_selection.json` (SHA `c227a604cfc4cbbab18d6087d91d0aab298ebccc47fccab9aa3709dfd3ef2a38`)
- L89C historical epoch-2 source SHA verified: `f8b175597ece8aad1f0ec3ae9d05c70e0759a0f7480dff9af6ea2619a2e3f08b`
- L89D selected checkpoint: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/train/joint40/checkpoint_l89_epoch008.pt` (epoch 8, SHA `28d4c35a189474d446a26fc096bbafbad77e1298429674383ce7ce8b14677d99`)
- base Z1 cache: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/outputs/l85/features/fit_dev_eval_full_attempt2`; summary SHA `e1c0d2b688f6b097528850a7f4ef4812965f2b015a435fd2cc722f57090f9f99`
- dev Z1 supplement: `/data2/usr_for_deadline/locatemot_l89d/supplement_dev_full_attempt1`
- internal Z1 supplement: `/data2/usr_for_deadline/locatemot_l89d/supplement_internal_full_attempt4`
- language cache: frozen L89 retry1 cache; no language regeneration

## 2. Root cause confirmed

The invalid L89/L89C system evidence used sparse dev/validation group frames for prediction, while `materialize_gt()` wrote GT on every native L69 frame. That produced a sequence/timeline mismatch and made the old near-zero HOTA invalid for full-video comparison.

L89D uses the same native L69 `frame_ids/frame_ptr` timeline for prediction and GT. Every legal query is paired with every native frame; sparse group keys are used only to define legal query scope.

## 3. Dense Z1 coverage

| scope | before | after |
|---|---|---|
| dev | dev: groups 0/2385, missing 2385, pairs 0/230538, dense_ready=False | dev: groups 2385/2385, missing 0, pairs 230538/230538, dense_ready=True |
| internal | internal: groups 0/1844, missing 1844, pairs 0/243550, dense_ready=False | internal: groups 1844/1844, missing 0, pairs 243550/243550, dense_ready=True |

The base cache was sparse/partial for the native universe. L89D built only the audited missing groups as label-free Z1 supplements on `/data2`, then passed the post-build coverage audit. The dev supplement finalized 2,385 groups / 230,538 query-frame pairs and is about 6.09 GB; the internal supplement finalized 1,844 groups / 243,550 query-frame pairs and is about 8.50 GB. No raw pixels or dense detector maps were persisted.

## 4. No-training and boundary contract

- `zero_training=true`; no optimizer step or checkpoint was created by L89D.
- L89 model weights, QSC-D architecture, loss, thresholds/rule fits, L69 bank, Z1 definition, tracker IDs and boxes were unchanged.
- `candidate_deletion=false`, `candidate_truncation=false`; all native candidate rows were scored.
- `screening_gt_used=false`, `official_test_labels_read=false`, `ordinary_mot_ovmot_touched=false`.
- token/span-to-region and static/motion alignment remain `UNALIGNED`.
- TrackEval used the local checkout with no verifiable Git HEAD; this is recorded in machine outputs.

## 5. True-full-video dev TrackEval matrix

Percent-valued metrics are local TrackEval values multiplied by 100; IDSW is a count.

| checkpoint/rule | HOTA | DetA | AssA | DetRe | DetPr | IDF1 | IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|
| epoch 8 / B | 28.0834 | 17.6694 | 44.9330 | 52.3861 | 20.8534 | 23.9180 | 3848.5 |
| epoch 8 / P | 28.5060 | 17.5421 | 46.7466 | 63.3221 | 19.4291 | 24.1055 | 4635.5 |
| epoch 8 / R | 22.8234 | 10.8910 | 48.7654 | 78.3718 | 11.1991 | 16.2972 | 6265.0 |
| epoch 20 / B | 26.1035 | 15.4607 | 44.3306 | 47.7117 | 18.4970 | 21.9308 | 3962.0 |
| epoch 20 / P | 23.2495 | 12.8769 | 42.1943 | 27.9671 | 19.1416 | 19.3328 | 2293.5 |
| epoch 20 / R | 24.7105 | 13.2343 | 46.6487 | 59.2874 | 14.5209 | 19.4204 | 4885.0 |
| epoch 40 / B | 25.0920 | 15.7423 | 40.3069 | 39.6501 | 20.4847 | 22.5053 | 4701.0 |
| epoch 40 / P | 23.1752 | 13.7501 | 39.3445 | 29.7425 | 20.1795 | 20.3703 | 3433.0 |
| epoch 40 / R | 25.5727 | 15.1331 | 43.5751 | 47.9914 | 17.9922 | 21.8333 | 5061.5 |
| epoch 32 / B | 25.9295 | 16.1873 | 41.8678 | 43.4743 | 20.2904 | 23.0741 | 4938.5 |
| epoch 32 / P | 23.9246 | 14.1653 | 40.7261 | 31.9141 | 20.1086 | 20.9651 | 3503.5 |
| epoch 32 / R | 26.0137 | 15.3130 | 44.5990 | 52.2148 | 17.7038 | 22.0565 | 5360.5 |
| epoch 2 / B | 26.9766 | 15.1974 | 48.0541 | 59.3939 | 16.8693 | 21.6227 | 4227.5 |
| epoch 2 / P | 27.6591 | 15.7627 | 48.7384 | 56.8124 | 17.8069 | 22.7314 | 4016.5 |
| epoch 2 / R | 19.5173 | 8.0864 | 47.6031 | 84.9279 | 8.1825 | 12.2942 | 6753.5 |

All five shortlist checkpoints and B/R/P rules completed over the six L82 dev videos. The dev matrix contains 15 results and passed the native full-video timeline checks.

## 6. New frozen L89D selection

- selected epoch: `8`
- selected rule: `P`
- checkpoint SHA256: `28d4c35a189474d446a26fc096bbafbad77e1298429674383ce7ce8b14677d99`
- candidate threshold: `0.75`
- presence threshold: `-1.0`
- null margin: `0.75`
- dev selection HOTA/DetA/AssA (%): `28.5060` / `17.5421` / `46.7466`

The selection tuple was exactly `(HOTA, DetA, AssA, distinct_target_recall, -inactive_false_acceptance, -epoch, deterministic_rule_order)`. L89D did not refit thresholds or use fixed validation labels for selection.

## 7. Fixed 16-calibration / 24-validation semantic replay

This is a fixed semantic diagnostic, not HOTA or screening. The unchanged evaluator attached calibration labels after score construction and validation labels afterward.

| metric | L29 control | L89D validation |
|---|---:|---:|
| recall | 0.7333 | 0.5484 |
| precision | 0.0830 | 0.1518 |
| FP/frame | 10.1250 | 3.9583 |
| predictions/positive | 8.8333 | 3.6129 |
| hard violation | 0.9167 | 0.6923 |
| multi-positive recall | 0.8194 | 0.4861 |
| inactive false acceptance | — | 0.6667 |
| empty rate | — | 0.2500 |

Decision: `semantic_gate_fail`. Recall and multi-positive recall were below the registered floors, although precision, FP/frame, predictions/positive and hard violation met their individual floors. This is not a deployment or ordinary-level RMOT pass.

## 8. Final internal true-full-video TrackEval

This is legal internal validation-scope TrackEval only: V1 videos 0004/0018 (86 query sequences) and V2 videos 0016/0017/0020 (537 query sequences). It is not screening or official-test evidence.

| dataset | HOTA | DetA | AssA | LocA | DetRe | DetPr | AssRe | AssPr | IDF1 | IDR | IDP | IDSW | FP | FN | MOTA | MOTP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| refer_kitti_v1 | 25.0799 | 16.3984 | 38.6984 | 90.4354 | 67.6963 | 17.6572 | 44.6961 | 75.9443 | 19.0120 | 45.9513 | 11.9855 | 1834.0 | 93827.0 | 8047.0 | -242.6212 | 90.2383 |
| refer_kitti_v2 | 22.6374 | 13.8605 | 37.1939 | 89.7389 | 46.8155 | 16.3337 | 44.2185 | 70.9503 | 18.3743 | 35.5194 | 12.3925 | 10841.0 | 710115.0 | 149332.0 | -189.6181 | 89.7079 |

Combined unweighted mean: HOTA `23.8586%`, DetA `15.1294%`, AssA `37.9462%`, DetRe `57.2559%`, DetPr `16.9954%`, IDF1 `18.6932%`, IDSW `6337.5`.

## 9. Historical comparison and interpretation

| evidence | V1 HOTA (%) | V2 HOTA (%) | authority |
|---|---:|---:|---|
| L86 historical full-video | 29.1663 | 21.6467 | historical comparison |
| L87-A historical full-video | 28.5752 | 22.1300 | historical comparison |
| L89C internal result | 2.244718 | 0.583059 | invalid for full-video comparison — sparse prediction/ dense GT mismatch |
| L89D repaired internal result | 25.0799 | 22.6374 | valid native-timeline internal TrackEval |

The repaired timeline explains the L89C near-zero system result: L89D recovers to a meaningful full-video range rather than 0–5 HOTA. The remaining gap is real and must be separated into detection/volume (DetA/DetPr), association (AssA/IDSW), and inactive/no-match calibration. The fixed semantic replay independently shows a hard/multi-positive trade-off, so the timeline fix does not establish a correspondence or ordinary-RMOT solution.

## 10. Evidence boundary and next action

No screening, official-test, HOTA fast-screening gate, or ordinary MOT/OVMOT regression was run. The L89D branch is complete and stopped. The single next action is supervisor review followed by one separately authorized absence/volume-calibration study; do not extend L89, retune its threshold, add NULL filtering, or change the tracker in this stage.

### Artifact status

- dev true-full-video inference: `outputs/l89d/dev/trackeval_true_fullvideo_attempt2`
- dev selection: `outputs/l89d/dev/final_selection_attempt2`
- fixed semantic: `outputs/l89d/eval/fixed_semantic_attempt1`
- internal true-full-video TrackEval: `outputs/l89d/internal/trackeval_true_fullvideo_attempt1`
- screening: not run
- official test: not read
- formal production RMOT: not claimed
