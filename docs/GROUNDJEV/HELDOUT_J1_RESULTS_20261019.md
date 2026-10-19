# GroundJEV-MOT / TrackJEV held-out results (2026-10-19)

Gate status: `HELDOUT_ALL_REGISTERED_TASKS_COMPLETE`; `failed=None`.

## Official registered metrics

| Evaluation | Frames | Metrics | Predictions / GT |
|---|---:|---|---|
| dt1203 | 1203 | HOTA=0.0, DetA=0.0, AssA=0.0, IDF1=0.0, IDSW=0, FP=0, FN=4586 | None / None |
| rmot_single8 | 8 | HOTA=0.0, DetA=0.0, AssA=0.0, IDF1=0.0, IDSW=0, FP=0, FN=5 | None / None |
| rmot_multi12 | 12 | HOTA=0.0, DetA=0.0, AssA=0.0, IDF1=0.0, IDSW=0, FP=0, FN=30 | None / None |
| ovmot_video4 | 40 | TETA=0.0, LocA=0.0, AssocA=0.0, ClsA=0.0, LocRe=0.0, LocPr=0.0, AssocRe=0.0, AssocPr=0.0, ClsRe=0.0, ClsPr=0.0 | 0 / 146 |
| ovmot_video63 | 40 | TETA=0.0, LocA=0.0, AssocA=0.0, ClsA=0.0, LocRe=0.0, LocPr=0.0, AssocRe=0.0, AssocPr=0.0, ClsRe=0.0, ClsPr=0.0 | 0 / 160 |

## Checkpoint and dependency state

- J1 final step: `1200`; rows: `43533` (identity 21,168; RMOT 16,677; OVMOT 5,688).
- Merged checkpoint: `/data2/user/LocateMOT_OneJev_CVPR/outputs/groundjev/full_data_v2/post_j1_heldout_gate_v1/merged_3task_J1_v4_birth`.
- MOT20 G0: `VIDEO_INFERENCE_COMPLETE_WITH_REJECTED_PERCEPTION`, frames `1658`.
- Clean G2: latest event `3100`/18,227, final receipt exists `False`.
- GPU5 G1 queue: `WAITING_G2_CLEAN_SUCCESS`.

## Root cause of zero scores

- RMOT/OVMOT outputs were rejected by the strict wire parser on duplicate coordinate tuples or duplicate phrase entries; the registered empty-and-record policy produced empty predictions.
- DT1203 had detector candidates on 1,201/1,203 frames, but the J1 association head selected `NO_MATCH` for every candidate, so no tracks reached the evaluator.
- These receipts are protocol-faithful evidence, not a claim of model quality. Any parser or calibration repair must use a new run and preserve this gate.
