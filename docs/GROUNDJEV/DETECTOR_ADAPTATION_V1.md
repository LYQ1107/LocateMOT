# GroundAnything detector adaptation V1

## Scope

G0 is the frozen pretrained GroundAnything DLM. G200 is the historical narrow projector adaptation. G1 uses the same pretrained initialization with a frozen vision/language body and a trainable multimodal projector over MOT17, MOT20, DanceTrack, and Refer-KITTI legal TRAIN clips. Held-out DT1203, Refer-KITTI 0007, MOT20-05 evaluation frames, and TAO40 are excluded.

## Measured state

- G1 completed 200/200 steps with a resumable state and a materialized serving directory at `models/GroundAnything-DLM-TrackProjG1`.
- The same-frame G0/G1 comparison is in `outputs/groundjev/full_data_v2/detector_comparison/g1_vs_g0/comparison.json`.
- G1 did not improve G0: the measured recall deltas are within about 1.2 percentage points and are mostly negative. This is retained as a valid negative result; it is not promoted as the formal detector.
- Before the later user-authorized control, GPU5 was occupied by another user's `/data2/OCD/TALON` jobs; that historical snapshot did not stop or touch the foreign processes.

## Next detector action

The next useful detector experiment is expression-conditioned RMOT native export. The previous RMOT cache queried only `pedestrian`, so its semantic quality gate correctly rejected the resulting rows. The new pilot manifests are in `outputs/groundjev/full_data_v2/candidate_export_rmot_expression_v1/manifests/` and are not yet inferred.

`full_goal_complete=false`.

## Addendum — GPU5 short functional retention control (2026-10-18)

At the user's direction, GPU5 was shared with the existing `/data2/OCD/TALON`
processes after a memory check (about 68.6 GiB free; the G1 training peak was
about 18.3 GiB). The G1 service was launched from the GroundJEV checkout so the
shared model link passed the service's isolation check. The owner completed all
six registered controls in
`outputs/groundjev/full_data_v2/g1_tracking_controls_20261018_gpu5_retry2/`:
MOT4, RMOT single8, and RMOT multi12, each with G0 and G1 and unchanged
TrackEval/scoring contracts. The service was stopped by the owner afterwards;
the foreign TALON jobs were not touched.

| short scope | G0 HOTA / IDF1 | G1 HOTA / IDF1 | G1−G0 HOTA pp | G1−G0 IDF1 pp |
|---|---:|---:|---:|---:|
| MOT4 | 0.7485 / 0.8289 | 0.7333 / 0.8134 | −1.52 | −1.55 |
| RMOT single8 | 0.2718 / 0.4000 | 0.2117 / 0.2105 | −6.02 | −18.95 |
| RMOT multi12 | 0.2179 / 0.2222 | 0.2284 / 0.2273 | +1.05 | +0.51 |

This is a registered short functional control, not a full-dataset score or a
promotion claim. It reinforces the existing same-frame detector result: G1 is
not uniformly better than G0 and remains a measured negative/neutral ablation.
