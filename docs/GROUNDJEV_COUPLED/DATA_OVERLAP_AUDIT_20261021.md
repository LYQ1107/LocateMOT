# GroundJEV joint-corpus overlap audit — 2026-10-21

The 1,116-row Phase-B joint corpus was compared against all 583,251 rows in
the protected G2 Clean TRAIN manifest (`f682519a…06e71`). The audit found 678
joint rows sharing an exact image path with G2 TRAIN, covering 290 unique
images. The overlaps are 221 identity-pair rows and 119 occlusion-recovery
rows from DanceTrack/MOT17, plus 338 RMOT rows from Refer-KITTI-V2. There are
no exact OVMOT overlaps because the 430 OVMOT rows are Base teacher-category
crop rows rather than native TAO frame rows.

These rows are valid TRAIN-scope engineering controls, but they are not
independent G2 generalization evidence. The overlap audit read no official
test labels and leaves `independent_generalization_claim_allowed=false`.
Future held-out evaluation must use video/image paths outside this overlap
set and keep the same parser, candidate protocol and assignment contract.

The reproducible CPU audit is
`tools/groundjev/coupled/audit_joint_corpus_overlap.py`; its raw receipt was
written to the registered external output directory and is not committed as a
large data artifact.
