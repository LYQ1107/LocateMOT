# L85 Failure Decomposition

## Status

`full_rmot_hota_complete_below_semantic_gate`

L85 completed its registered compact full-RMOT training, fixed calibration/
validation evaluation, legal internal full-video inference, and local
TrackEval run. It did not pass the fixed semantic emission gate and is not a
claim of ordinary-level RMOT performance.

## Evidence chain

- Selected frozen representation: L84 Z1, fixed-reference GroundingDINO
  decoder layer 1.
- Candidate input: immutable L69 budget-40 bank, with all rows and native
  query-independent row order retained. The bank was reused; it was not
  copied or rebuilt.
- Label-free semantic cache: 1,623 complete groups, with no labels, pixels or
  dense feature maps. Internal-validation candidate oracle only: unit
  coverage `.8385300668`, target micro coverage `.8792302588`.
- Forward/loss contract: passed. The authoritative smoke at
  `outputs/l85/train/smoke100_retry3/` was 100/100 finite and
  nonzero-gradient with strict reload, both domains and all four strata.
- Full training: 40 epochs, 20,960/20,960 finite and nonzero-gradient steps;
  every epoch checkpoint was written. The long run used one process on GPU0
  because GPU1 was occupied by an unrelated process. A separate four-GPU
  one-step DDP contract passed.
- Implementation qualification: S/T/J was implemented as causal per-row
  history (current row for S; last four valid observations for T/J), not as a
  separately batched clip-4 tensor. The result is therefore reported as a
  causal-history curriculum, not as a full explicit clip-batch experiment.

## Fixed semantic validation

The dev-selected checkpoint was epoch 09 / step 4716:

`outputs/l85/train/joint_curriculum40_gpu0/checkpoint_l85_epoch09.pt`

SHA256: `785e9a5cd7245f502e6e4eae84592c72b7703739c9d8edb599bc35c495a96f7e`

Frozen rule: candidate threshold `1.2`, presence threshold `-0.1`, null
margin `0.0`. The fixed 16-calibration/24-validation result was:

| Metric | L85 validation | Registered requirement | Result |
|---|---:|---:|---|
| Candidate recall | 0.4193548 | >= 0.7233333 | FAIL |
| Candidate precision | 0.1300000 | >= 0.0830189 | pass |
| FP/frame | 3.6250 | <= 11.125 | pass |
| Predictions/positive | 3.2258 | <= 4.069 | pass |
| Hard-negative violation | 0.7692308 | <= 0.8666667 | pass |
| Multi-positive recall | 0.4861111 | >= 0.7894444 | FAIL |
| Inactive false acceptance | 1.0000 | < 1.0 | FAIL |

The simultaneous gate therefore fails. The lower output volume and lower hard
violation do not compensate for recall loss, multi-positive loss, and
universal inactive acceptance. This is not a threshold/top-k/NMS repair and
no candidate rows were deleted during scoring.

## First actionable root cause

The first actionable bottleneck is **cross-video query-to-candidate semantic
emission/presence calibration**: the factorized score can suppress volume and
improve aggregate hard-negative/FP statistics, but its learned presence/null
behavior and target-bag ranking do not preserve held-out positives, especially
multi-positive targets. Candidate coverage is an upper-bound limitation for
some units but does not explain the full failure; the internal oracle coverage
is materially above the required recall floor.

The causal-history implementation qualification is recorded separately and is
not used to relabel the metric failure as an implementation-only failure.

## Full-video validation TrackEval

After the semantic result was frozen, full-video inference was run only on
legal internal validation videos: V1 `0004`, `0018`; V2 `0016`, `0017`,
`0020`. The local TrackEval API completed at
`outputs/l85/trackeval/trackeval_attempt4/trackeval_summary.json`.

| Dataset | HOTA | DetPr | DetRe | AssA | IDF1 | IDSW |
|---|---:|---:|---:|---:|---:|---:|
| Refer-KITTI V1 | 25.0548 | 17.8400 | 47.4128 | 42.4542 | 21.3342 | 2,330 |
| Refer-KITTI V2 | 17.2924 | 10.9991 | 45.9492 | 30.8389 | 12.9430 | 18,374 |

These are **internal full-video validation HOTA** values, not official-test
values. They were not used for checkpoint or threshold selection. No screening
labels, official-test labels, ordinary MOT/OVMOT regression, or production
path changes were used.

## Unique next action

Stop L85 and request supervisor review for exactly one new RMOT-only
correspondence or presence-calibration hypothesis. Do not extend L85, add
NULL/top-k rescue, alter the bank, or change ordinary MOT/OVMOT without a new
approved stage.

## Provenance flags

```json
{
  "screening_gt_used": false,
  "official_test_labels_read": false,
  "ordinary_mot_ovmot_touched": false,
  "hota_trackeval_run": true,
  "hota_scope": "internal_full_video_validation_only",
  "token_span_region_alignment": "UNALIGNED",
  "static_motion_alignment": "UNALIGNED"
}
```
