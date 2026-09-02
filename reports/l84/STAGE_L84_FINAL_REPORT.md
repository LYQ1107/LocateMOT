# Stage L84 final report — paired mid-decoder verification

## Status

`original_content_seed_selected`
`L85_AUTHORIZED=true`
`NEXT_STAGE=L85_FULL_RMOT`

Selected semantic representation: **Z1** (fixed-reference decoder layer 1).
No-refPE: **not selected**.

The authoritative source-of-truth snapshot is
`outputs/l84/preregister/source_of_truth.json`.  The paired run is
`outputs/l84/train/paired_middecoder/`; the corrected selection/no-refPE run
is `outputs/l84/train/selection_correction_attempt2/`.

## Answers to the preregistered questions

* **Does the mid-decoder advantage survive paired initialization?** Yes for
  Z1 and R1 under the diagnostic stable gate.  Relative to Z0, the three-seed
  Z1 mean bag-hard improvement is `0.099359`, hit@1 improvement is `0.108974`,
  V2 hard improvement is `0.047120`, and V2 hit improvement is `0.036649`.
  The paired bootstrap hard-improvement 95% CI is `[0.061042, 0.193496]`.
* **Are seeds consistent?** Yes for the registered gate: Z1 hard and hit
  improve for all three seeds; V2 hard improves for all three and V2 hit
  improves for two (the third is unchanged).  The exact per-seed deltas are
  in `stable_gate.json`.
* **Is R4 better than Z4?** No evidence authorizes that claim.  R4 has worse
  aggregate hard/hit/multi means than Z4 and fails the bootstrap/stability
  criteria; native refinement is diagnostic only.
* **Is native refinement worth retaining?** R1 is exactly tied with Z1 in
  saved dev records and checkpoint state, while R4/R6 fail stability.  It is
  not retained as a distinct semantic representation.
* **Is no-refPE better?** No.  The corrected Z1 no-refPE test failed its
  preregistered hard/hit/V2 rule, so original-content Z1 remains selected.

## Protocol and limitations

The contract audit passed: all seven state families were finite, candidate
rows were retained in L69 order, history/candidate counts were unchanged,
canonical initialization differences were zero, and three 10,000-resample
paired bootstrap manifests were written.  A known harmless checkpoint load
warning (`language_model...position_ids`) was recorded; it is the only
unexpected key and was not silently called a perfect key match.  The local
MMDetection checkout has no verifiable local HEAD; public v3.3.0 reference
commit `44ebd17b145c2372c4b700bfb9cb20dbd28ab64a` is recorded in the
literature audit.

All L84 labels were L49 fit labels attached after complete feature/state
construction.  No screening or official-test labels were read.  No HOTA,
TrackEval, full-video RMOT, ordinary MOT or OVMOT result exists yet.  Token
span-to-region and static/motion alignment remain `UNALIGNED`.

The initial selection output selected R1 due a code-level reverse-sort
mistake.  It is preserved, not overwritten.  The corrected selection audit
selected Z1 by the exact registered earliest-stage tie-break and ran the
authorized no-refPE test for Z1.  This is a protocol correction, not a new
scientific branch.

## L85 handoff

L84 is the final representation-only probe.  L85 must now use Z1 in the new
factorized full-RMOT model, retain complete query-independent candidate tracks,
add causal temporal history and factorized candidate/presence energies, and
run the authorized full-video TrackEval/HOTA protocol if its contracts pass.
L85 must report candidate oracle, training, calibration/dev selection and each
legal Refer-KITTI evaluation split separately.  It must not change ordinary
MOT/OVMOT or use hidden official-test labels.

Flags: `screening_gt_used=false`, `official_test_labels_read=false`,
`hota_trackeval_run=false`, `ordinary_mot_ovmot_touched=false`,
`candidate_deletion=false`, `candidate_truncation=false`.
