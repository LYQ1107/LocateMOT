# J1 zero-metric repair and verified OVMOT boundary (2026-10-19)

**Status: PARTIAL_WITH_VERIFIED_OVMOT_BLOCKER.** The original J1 held-out gate is preserved and immutable. The repair run is recorded as a new experiment; no short clip or malformed decoder output is presented as a quality claim.

## Completed evidence

- The real OneJev J1 three-task training reached step 1200. The original gate remains immutable.
- The original gate was all zero: DT1203 HOTA/DetA/AssA/IDF1=0 with FN=4586; RMOT single8 all zero with FN=5; RMOT multi12 all zero with FN=30; OVMOT video4/video63 all TETA components zero with zero predictions.
- The repaired parser removes only exact duplicate phrase/coordinate entries when the wire payload is otherwise complete. The association policy is `no_match_policy=only_if_omitted`. RMOT smoke replay became nonzero: single8 HOTA=0.3792207792 and IDF1=0.6666666667; multi12 HOTA=0.1598656815 and IDF1=0.1463414634.
- The full repaired DT1203 run completed all 1203 frames and official TrackEval: HOTA=0.1585619177, DetA=0.1976921614, AssA=0.1272432780, IDF1=0.1282768263, IDSW=0, FP=110, FN=3560.
- OVMOT video4 completed its registered 40-frame TAO validation scope and the unmodified official TETA evaluator. All 40 frames were rejected by the strict perception wire parser, so the result has 0 predictions, 146 GT boxes, TETA/LocA/AssocA/ClsA=0, and end-to-end FPS=0.0037955. This is a verified failure boundary, not a model-quality claim.

## OVMOT failure boundary

The decoder returned malformed wire payloads, including missing coordinate or box-start tokens, unclosed object references, duplicate coordinate tuples, duplicate phrases, invalid non-degenerate boxes, and missing entry separators. Exact duplicate tuples can be normalized safely; missing tokens and invalid boxes cannot be reconstructed without inventing detections or changing the protocol. The official receipt retains every rejected frame.

Video4 frame latency was 239.58--288.17 seconds, with 40/40 rejections. To avoid consuming GPU indefinitely on invalid output, video63 was not started after this verified boundary. It remains a separately named pending decoder/schema replay; no metric is fabricated for it.

## Pending work

1. Diagnose the original space-token/generation/schema incompatibility and run a separately named decoder repair on video4/video63.
2. Complete J1 A--E parser/NO_MATCH ablations on the same checkpoint and candidate archive, with G0+Simple/Compact references.
3. Start real YesTrack training and compare no verifier, original verifier, and fine-tuned verifier on independent rows.

## Evidence locations

- Repaired suite: `/data2/user/LocateMOT_OneJev_CVPR/outputs/groundjev/j1_parser_nomatch_repair_20261019/full_repaired_v3`
- DT receipt: `.../full_repaired_v3/dt1203/result.json`
- OVMOT video4 receipt: `.../full_repaired_v3/ovmot_video4/official_metrics.json`
- Immutable original gate: `/data2/user/LocateMOT_OneJev_CVPR/outputs/groundjev/full_data_v2/post_j1_heldout_gate_v1`
