# Stage L6 Failure Analysis

Status: COMPLETE (final checkpoint).

## 1. DanceTrack collapse (AssA 0.4169 -> 0.3248)

Switch diagnosis on DanceTrack val (12,543 switches):

- 11,525 (92%) switches occur at **gap = 1**: the identity was observed in
  the previous frame without any detection gap.  These are
  consecutive-frame association errors, not occlusion/reappearance errors.
- Mean IoU with another GT object in the previous frame = 0.42
  (dense, overlapping same-class dancers).
- Detection-gap fraction only 8.1%.

Interpretation: on DanceTrack the model over-trusts PBD appearance
evidence between similar-looking dancers and under-uses motion/geometry
competition, so it swaps identities while both objects are continuously
visible.  The learned cue reliability is dataset-conditioned in effect
(without a dataset ID): appearance helps BDD/MOT17/MOT20, hurts Dance.

## 2. BDD ID switches (IDSW 11042 -> 7546, still frequent)

Switch diagnosis on BDD (9,556 switches):

- Most switches occur after gaps of 6–10 frames (detection miss then
  re-birth/re-match); detection-gap fraction 100%.
- BDD cross-spec drift improved from 53.2% to 17.0%: the dynamics model
  is more spec-consistent than any previous stage.

Interpretation: BDD switches are largely detection-gap re-identification
errors; a long-gap memory (beyond MAX_AGE=30) and better re-identification
would further reduce them.

## 3. MOT17/MOT20 IDSW regression (259->434, 2406->1645)

MOT17 IDSW rises despite AssA +9.4pp: the model creates more short-lived
track fragments (births) than U0 while matching more correctly overall.
IDSW-optimal thresholds differ from AssA-optimal thresholds; a
tracking-level threshold calibration (e.g., NEW margin) could trade
fragments for switches.

## 4. Exposure bias / model-in-the-loop

Training uses scheduled sampling (teacher 1.0 -> 0.4).  The final model
shows no evidence of rollout instability (loss converged, no divergence),
but Dance collapse shows that student-rollout errors (wrong matches)
propagate through persistent memory — the failure is in cue reliability,
not in state explosion.
