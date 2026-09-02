# L85 source of truth

## Identity and frozen boundary

This stage belongs to Luna thread `01a02014-fce8-7f51-8414-e7ed6ab44745`,
project root `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`, branch
`codex/l85-full-rmot-hota-20260902`, starting from L84 commit
`c65af026c02fbe7fd24e72a315963d89373dcd4c`. The fixed fast manifest is
`outputs/l19/protocol/kitti_fast_eval_manifest.json` with expected SHA256
`06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.

L84 selected the fixed-reference decoder representation Z1. Its stable paired
diagnostic is evidence for a representation to test, not a deployable RMOT
result. L85 therefore uses a factorized full-video RMOT sidecar and must not
rewrite the frozen banks, detector, UIDM, TrackEval checkout, or ordinary
MOT/OVMOT entrypoints.

## Data and legal evaluation scope

The candidate source is the validated L69 budget-40 view at
`outputs/l69/attempt9/`; V2 files may resolve to the audited L69 attempt4
objects through existing symlinks. The bank is query-independent and GT-free.
Rows are reconstructed from each bank's native `frame_ptr` and `frame_ids`.
The observation input is 1432-dimensional:
`clip[512] + history_clip[512] + uidm_h[384] + geometry[7] + motion[8] +
lifecycle[8] + objectness[1]`.

L85 fit uses only L49 `split=fit` expression-level labels. Internal,
video-disjoint dev groups are declared by the existing L82 split file. Legal
full-video internal validation scope is V1 videos `0004,0018` and V2 videos
`0016,0017,0020`; hidden official-eval videos and screening labels are not
read. Any use of labels is separated as fit, dev/calibration, or validation.

## Current implementation truth

The new code validates and reuses the complete L69 bank and captures compact
Z1/text summaries with the verified GroundingDINO runtime. It does not copy
detector weights or persist raw maps. The current compact model implements
`S=A+B+R_total`, a causal history correction, centered query-independent
candidate prior, query/frame presence energy, shared-energy NULL diagnostic,
and semantic/static ranking terms. Z1 is currently a frozen compact input;
the L84 optional detector-side LoRA is not silently claimed as trained when it
is absent from the captured cache. This distinction is reported in all
training results.

No token/span-to-region or static/motion alignment is verified: status is
`UNALIGNED`. No candidate is deleted, top-k/NMS is applied, or ID is supplied
as a semantic input. Track IDs are used only to reconstruct causal history.

## Required truth flags

`candidate_bank_gt_conditioned=false`, `candidate_bank_query_conditioned=false`,
`screening_gt_used=false`, `official_test_labels_read=false`,
`ordinary_mot_ovmot_touched=false`. Before a valid TrackEval invocation there
is no HOTA claim. If only internal legal sequences are evaluated, the metric
name is **full-video validation HOTA**, not official test HOTA.

## Final L85 evidence pointers

The 40-epoch run completed at
`outputs/l85/train/joint_curriculum40_gpu0/`, dev selection froze epoch 09 at
`outputs/l85/eval/dev_selection_attempt2/`, and fixed semantic evaluation is
at `outputs/l85/eval/fixed_semantic_attempt1/`. Full-video internal
prediction generation completed at
`outputs/l85/trackeval/fullvideo_validation_attempt5/`; the authoritative
TrackEval result is `outputs/l85/trackeval/trackeval_attempt4/`.

The fixed semantic gate failed before HOTA: validation recall was `0.4193548`,
precision `0.1300000`, FP/frame `3.625`, predictions/positive `3.2258065`,
hard-negative violation `0.7692308`, multi-positive recall `0.4861111`, and
inactive false acceptance `1.0`. The full-video validation HOTA was later run
because the L85 plan requires a genuine TrackEval result even when the
surrogate gate is weak: V1 HOTA `25.0548`, V2 HOTA `17.2924`. Neither result
is an official test or a production promotion.
