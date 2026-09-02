# L85 internal-dev checkpoint and rule selection

Selection was performed only on the predeclared video-disjoint internal dev
groups.  It was completed before the fixed 16-calibration/24-validation
semantic report and before internal full-video validation GT was materialized.
The authoritative machine record is
`outputs/l85/eval/dev_selection_attempt2/checkpoint_selection.json`.

## Frozen selection

| item | selected value |
|---|---|
| checkpoint | `outputs/l85/train/joint_curriculum40_gpu0/checkpoint_l85_epoch09.pt` |
| step | 4,716 |
| checkpoint SHA256 | `785e9a5cd7245f502e6e4eae84592c72b7703739c9d8edb599bc35c495a96f7e` |
| candidate threshold | 1.2 |
| presence threshold | -0.1 |
| NULL margin | 0.0 |
| adapter norm | 78.19966218564967 |

The registered tuple was lower dev hard-negative violation, higher top-1,
higher multi-target exact, lower inactive false acceptance, earlier epoch,
then smaller parameter norm.  Dev full-video HOTA was unavailable at this
selection pass and was explicitly **not used**.  No validation, screening, or
official-test labels entered this selection.

## Dev evidence

The selected rule was evaluated on 498 internal-dev units and 24,495 candidate
rows: 750 positive rows, 2,396 emitted rows, 422 TP and 1,974 FP.  Candidate
precision was `0.1761268781`, recall `0.5626666667`, FP/frame `3.9638554217`,
predictions/positive `3.1946666667`, top-1 `0.3846153846`, top-5
`0.8141025641`, hard-negative violation `0.7820512821`, multi-positive recall
`0.5643578290`, multi-target exact `0.3503184713`, empty rate `0.1887550201`,
and inactive false acceptance `0.6632653061`.  These are dev selection
statistics, not fixed validation or HOTA metrics.

The fixed semantic evaluation uses this checkpoint/rule without any later
change.  The subsequent full-video validation result, if the TrackEval format
passes, is reported in the two dataset-specific HOTA reports and is not used
to revise this selection.

## Provenance flags

`dev_full_video_hota_used_for_selection=false`, `screening_gt_used=false`,
`official_test_labels_read=false`, `ordinary_mot_ovmot_touched=false`,
`token_span_region_alignment=UNALIGNED`, `static_motion_alignment=UNALIGNED`.
