# J1 zero-metric repair and live failure report (2026-10-19)

**Status: IN_PROGRESS.** This report is an auditable snapshot of the J1 repair
work. It deliberately records empty outputs and malformed decoder responses;
it does not turn an unfinished OVMOT run into a completed result.

## What was already completed

- The real OneJev J1 three-task training reached step 1200. Its original
  registered held-out gate is preserved and immutable.
- The original gate was all zero: DT1203 HOTA/DetA/AssA/IDF1=0 with FN=4586;
  RMOT single8 all zero with FN=5; RMOT multi12 all zero with FN=30; OVMOT
  video4/video63 all TETA components zero with zero predictions.
- The parser/association repair added the pinned
  `official_scoring_repaired` policy and `no_match_policy=only_if_omitted`.
  RMOT smoke replay became nonzero: single8 HOTA=0.3792207792 and
  IDF1=0.6666666667; multi12 HOTA=0.1598656815 and IDF1=0.1463414634.
- The full repaired DT1203 run completed all 1203 frames and official
  TrackEval. It is no longer all zero: HOTA=0.1585619177, DetA=0.1976921614,
  AssA=0.1272432780, IDF1=0.1282768263, IDSW=0, FP=110, FN=3560.

## Current full-repair failure evidence

The same full-repair suite is still processing the registered OVMOT scopes.
At the latest snapshot, `ovmot_video4/result.json` and its official metrics
receipt were not present, and the second OVMOT scope had not started. The
process is alive, but individual OVMOT frames take roughly 250--281 seconds
because the GroundAnything decoder returns malformed wire output and the
pipeline records an empty candidate set.

Observed decoder errors include:

- `expected coordinate token ... found ', <|object_ref_s'`
- `duplicate coordinate tuples are not canonical`
- `expected '<|box_start|>' ... found '<|object_ref_end'`
- `bbox must be non-degenerate ... (0, 483, 34, 476)`
- `duplicate phrase entry: 'aquarium'`

These errors are upstream decoder/schema failures. Exact duplicate tuples can
be safely normalized, which is what the repaired parser does where the wire
payload is complete. Missing tokens, unclosed object references, and invalid
boxes do not contain enough information to reconstruct a detection without
changing the evaluation protocol, so those frames remain empty and are
recorded as rejected. The current run is therefore evidence collection, not a
quality claim.

## Pending items

1. Finish the current OVMOT replay and run its official evaluator only after a
   complete receipt exists.
2. Add a separately named, targeted decoder/schema repair for the malformed
   OVMOT payloads and replay the affected registered scopes. Do not overwrite
   the original zero gate or this failure snapshot.
3. Start the real YesTrack verification/finetuning job after the J1 services
   are released. No YesTrack-specific process is running in this snapshot.

## Evidence locations

- Current repaired suite:
  `/data2/user/LocateMOT_OneJev_CVPR/outputs/groundjev/j1_parser_nomatch_repair_20261019/full_repaired_v3`
- Completed DT receipt:
  `.../full_repaired_v3/dt1203/result.json`
- Current OVMOT partial archive:
  `.../full_repaired_v3/ovmot_video4/`
- Immutable original all-zero gate:
  `/data2/user/LocateMOT_OneJev_CVPR/outputs/groundjev/full_data_v2/post_j1_heldout_gate_v1`

