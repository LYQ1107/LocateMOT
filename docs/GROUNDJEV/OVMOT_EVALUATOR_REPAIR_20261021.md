# OVMOT evaluator repair receipt — 2026-10-21

- Code commit: `16e20851dfa0110a4b7f45c89abfe02a9a8194e9`
- Evaluator SHA-256: `e750bdc972a4a4f5cb97aa8c0b298cdf5cad1a9411c7730eacf396668d716b6e`
- TETA commit: `b498aa87b252bfb75d7cc0d8d6435c3e1e74e4de`
- Official-test labels read: `false`
- Historical output directories were not modified.

The two recorded TETA failures came from non-default evaluator configuration: disabling `OUTPUT_PER_SEQ_RES` removed the `COMBINED_SEQ` structure expected by the summary reader, and disabling `OUTPUT_SUMMARY` left the official TETA return variable unassigned. The current evaluator explicitly keeps `OUTPUT_SUMMARY`, `OUTPUT_TEM_RAW_DATA`, and `OUTPUT_PER_SEQ_RES` enabled. Fresh copies of the inference artifacts were evaluated below.

## video63_simple_control

- Scope: `full_legal_validation_video_annotation_frames`; video `63`; `40` frames.
- Sequence: `val/HACS/Making_a_sandwich_v_SMk4troNhwE_scene_0_0-2482`
- Backend/checkpoint: `simple` / `None`
- Fresh output: `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/multitask_runs/ovmot_evaluator_repair_v1/video63_simple_control`
- Evaluator status: `OFFICIAL_REGISTERED_MULTITASK_SCOPE_EVALUATION_COMPLETE`; GT boxes `160`; predictions `35228`.
- Rejected frames: `[]`.
- TETA metrics: `{"TETA": 18.64, "LocA": 7.0114, "AssocA": 28.618, "ClsA": 20.29, "LocRe": 89.375, "LocPr": 7.0629, "AssocRe": 30.888, "AssocPr": 60.706, "ClsRe": 20.29, "ClsPr": 63.333}`
- Base/novel TETA: base `23.372500000000002`; novel `9.1737`.
- This result is `official_complete_dataset_result=False`.

## video4_native_rejected_4frame

- Scope: `registered_short_functional_clip`; video `4`; `4` frames.
- Sequence: `val/YFCC100M/v_25685519b728afd746dfd1b2fe77c`
- Backend/checkpoint: `simple` / `None`
- Fresh output: `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/multitask_runs/ovmot_evaluator_repair_v1/video4_native_rejected_4frame`
- Evaluator status: `OFFICIAL_REGISTERED_MULTITASK_SCOPE_EVALUATION_COMPLETE`; GT boxes `18`; predictions `0`.
- Rejected frames: `[751, 781, 811, 841]`.
- TETA metrics: `{"TETA": 0.0, "LocA": 0.0, "AssocA": 0.0, "ClsA": 0.0, "LocRe": 0.0, "LocPr": 0.0, "AssocRe": 0.0, "AssocPr": 0.0, "ClsRe": 0.0, "ClsPr": 0.0}`
- Base/novel TETA: base `0.0`; novel `None`.
- This result is `official_complete_dataset_result=False`.

## Video 4 boundary

The required full legal video-4 scope has 40 registered frames, but only the existing four-frame native diagnostic is available. All four native OneJev frames were rejected by the strict canonical coordinate-tuple parser (`751, 781, 811, 841`) and produced zero predictions. The full 40-frame native result is therefore not claimed; the parser/output repair remains a concrete blocker. The four-frame TETA zero is a diagnostic, not a performance result.

## Video 63 boundary

The 40-frame video-63 run is an independent TAO validation control and has a valid official TETA receipt. It uses the existing simple backend, not a G2/G1/OneJev adapted model, so it cannot support a GroundJEV adaptation claim.
