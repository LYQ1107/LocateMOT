# GroundJEV fixed-candidate MOT video diagnostic — 2026-10-21

The current coupled readout was replayed causally on all 128 registered frames
538–665 of MOT20-03.  The evaluator consumed the immutable native candidate
boxes and 14,784 cached 2,560-dimensional region features; it did not read GT
until after prediction generation, and it loaded no second VLM.  The replay
made 128 cached feature calls and emitted 152 track boxes on 76 frames.

TrackEval reports HOTA **0.027901**, DetA **0.009989**, AssA **0.077947**, IDF1
**0.005261**, IDSW **1**, FP **1**, and FN **14,524**.  The decision histogram
contains 11,462 explicit `REJECT` actions, two `NEW` actions, and 3,320
existing-track choices.  The large FN count is a real failure of the current
admission/readout policy, not a missing-frame shortcut.

For a same-input reference, the existing offline compact association
checkpoint on the identical candidate archive reports HOTA **0.336515**, DetA
**0.309884**, AssA **0.365702**, IDF1 **0.359322**, IDSW **159**, FP **119**, and
FN **9,832**.  This is a historical multi-component control, not a
single-backbone GroundJEV result; it shows that the current coupled readout is
far below the established physical tracking baseline and needs a bounded
admission/identity repair before any larger training run.

This clip is registered TRAIN scope and its feature archive predates the
current G2 Clean terminal checkpoint.  It is therefore a fixed-candidate
association/runtime diagnostic only.  It does not establish G2 Clean
performance, independent generalization, end-to-end Grounding improvement, or
the value of identity-conditioned Grounding.  The result is retained as the
first complete video-level failure boundary for the coupled path and must be
compared against a fixed-candidate G0+Simple control before any scientific
claim.

The reproducible entrypoint is
`tools/groundjev/coupled/evaluate_fixed_candidate_clip.py`; the raw receipt is
under the registered output directory at
`/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/fixed_candidate_mot20_03_phaseb_v1/official_metrics.json`.
