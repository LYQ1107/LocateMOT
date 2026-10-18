# GroundAnything detector adaptation V1

## Scope

G0 is the frozen pretrained GroundAnything DLM. G200 is the historical narrow projector adaptation. G1 uses the same pretrained initialization with a frozen vision/language body and a trainable multimodal projector over MOT17, MOT20, DanceTrack, and Refer-KITTI legal TRAIN clips. Held-out DT1203, Refer-KITTI 0007, MOT20-05 evaluation frames, and TAO40 are excluded.

## Measured state

- G1 completed 200/200 steps with a resumable state and a materialized serving directory at `models/GroundAnything-DLM-TrackProjG1`.
- The same-frame G0/G1 comparison is in `outputs/groundjev/full_data_v2/detector_comparison/g1_vs_g0/comparison.json`.
- G1 did not improve G0: the measured recall deltas are within about 1.2 percentage points and are mostly negative. This is retained as a valid negative result; it is not promoted as the formal detector.
- GPU5 is not available for a repeat: it is occupied by another user's `/data2/OCD/TALON` jobs. No other user's process is stopped.

## Next detector action

The next useful detector experiment is expression-conditioned RMOT native export. The previous RMOT cache queried only `pedestrian`, so its semantic quality gate correctly rejected the resulting rows. The new pilot manifests are in `outputs/groundjev/full_data_v2/candidate_export_rmot_expression_v1/manifests/` and are not yet inferred.

`full_goal_complete=false`.
