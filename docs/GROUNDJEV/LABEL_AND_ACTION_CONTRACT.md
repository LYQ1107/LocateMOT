# Global physical identity labels and actions, version 1

Milestone: `GOAL_GLOBAL_IDENTITY_DECISION_V1`, subordinate to the unchanged
`GOAL_GROUNDJEV_UNIFIED_ONLINE_TRACKING`. Source tree `d2d4c00`, verified source
GitHub commit `4881bbc`; implementation is isolated in
`codex/global-identity-decision-v1`. Original running sources remain untouched.

The whole-frame audit covers all 43 existing legal MOT TRAIN units, 1,056 frames
and 48,051 original GroundAnything candidates. Every original candidate and frame
is retained. Original global matching is reconstructed with the original Python
overlap/float64 arithmetic, including ties. The failed float32 prefix is preserved.
GT is external TRAIN supervision; it never replaces a predicted historical state.

## Admission and entity support

Candidate support is the complete set of annotated pedestrian GT boxes with
IoU >= 0.5. One supported GT gives a unique physical supervision identity. Multiple
supported identities are ambiguous and masked. A one-to-one global winner is an
allocation target, not the definition of objectness.

|Label|Meaning|Audited candidates|
|---|---|---:|
|VALID_OBJECT|Unique supported identity and original global winning proposal|36,397|
|DUPLICATE_COMPETITOR|Unique supported identity with another global winning proposal|5,567|
|BACKGROUND_OR_INVALID|Certified background or invalid box geometry only|0|
|UNKNOWN_SUPERVISION|Ambiguous, unmatched, or unassigned support without a certified target|6,087|

Both VALID_OBJECT and DUPLICATE_COMPETITOR have positive entity support. A duplicate
is suppressed only as an extra allocation, not relabeled as physical background.
The original 5,832 GT-supported IGNORE rows include ambiguous overlaps and are not
identical to the new unique-identity duplicate class. No negative background labels
are fabricated from the remaining unmatched candidates. Consequently this version
does **not** establish learned background rejection; that needs certified negatives.

## Identity actions

Runtime actions are MATCH(existing ID), REACTIVATE(existing lost ID), NEW and
REJECT(reason). No DEFER action or unspecified delayed-output state is enabled.
Supervision distinguishes duplicate rejection, invalid geometry and unknown evidence.

* A confirmed historical identity is bound once at a unique, GT-supported predicted
  birth. Wrong or unknown predicted births remain wrong/unknown in subsequent inputs.
* All confirmed matching live IDs are acceptable positives in ID CE. Supervision
  does not arbitrarily pick one erroneous duplicate historical ID by input index.
* A known existing identity outside the old Top-K is UNKNOWN for that local question,
  never NEW. Report live-pool and Top-K coverage separately.
* NEW requires a first supported identity in the registered causal clip, no confirmed
  existing identity, and no unresolved live birth that could conflict. It does not
  claim the object first appeared in the entire source video.
* Previously observed identities with no confirmed live anchor are UNKNOWN, including
  retirement and unknown-birth cases; they are not fabricated newborn examples.

The audit gives 554 NEW, 19,428 local MATCH/REACTIVATE, 5,567 duplicate REJECT and
22,502 UNKNOWN action rows. Among 28,507 candidates with confirmed live identity
support, the original Top-K covers 22,040. New V1 allocation first measures retrieval
scores, then scores **all** available ACTIVE/LOST identities in blocks, avoiding a
training label that requires an unseen omitted identity. This is O(NM) association
work and storage for the global cost matrix, not an O(N²) candidate attention bank.

## Global competition and allocation

For every candidate, use the union of its top 32 IoU, 32 center-distance and 32
native-appearance neighbors over the entire frame. Score blocks are 32×N and
there are at most 96N directed edges. Canonical geometric ordering breaks retrieval
ties before original-index fallback. No GT determines runtime neighbors.

All 4,804 original cross-group cases recover their **complete** GT-supported
competition sets. All 12,882 supported competitor sets are covered in this TRAIN
audit. The graph has 2,600,231 streamed directed edges across all frames. This is
coverage evidence, not a tracking-quality result or a claim on arbitrary scenes.

A1 trains ID/NEW/REJECT CE. A2 adds learned sparse messages, same-identity duplicate
BCE and GT-quality ordering. The discrete runtime uses one representative per
predicted duplicate component and Hungarian with private NEW/REJECT columns.
Uniqueness is guaranteed for predicted components and old IDs; correctness of the
learned duplicate grouping must be measured. Hungarian is outside the gradient path.

## Assignment-conditioned memory and lifecycle

Reuse `final-assigned-existing-identity-memory-update-v1` and its existing callback
after final global assignment, before any current-frame write. UPDATE supervision
requires a known unique current identity matching the immutable assigned birth and
reliable visual evidence. Known wrong binding is HOLD; unknown identity is masked.
The audit retains 13,052 existing certified assigned-write targets, with ambiguous
support masked. The separate validated RMOT component retains all 1,428 identity
and 1,454 assigned-write questions and their semantic controls.

Always-update, fixed-confidence, original JEV-update and assignment-conditioned
policies require actual video trajectories and UPDATE/HOLD flip counts; different
probabilities alone do not establish an algorithmic improvement. Sigmoid outputs
are uncalibrated until a separate legal TRAIN calibration/held-out check is reported.

NEW creates TENTATIVE, repeated observations confirm ACTIVE, absence gives LOST,
and the registered fixed timeout gives RETIRED. No unknown birth creates a permanent
exit label. MOT/RMOT clocks are physical seconds; TAO uses annotation steps with
unverified source FPS. Tentative output rules and timeout effects must be recorded
as runtime interventions, not hidden changes to the old protocol.

## Evidence and storage

`outputs/groundjev/global_identity_v1/label_audit.json` and `labels_v1.csv.gz` retain
all frame/candidate support, source allocation, duplicate sets, retrieval coverage,
action masks and original write-target provenance. Labels occupy 314,611 bytes.
All non-protected source files count conservatively as 22,583,722,572 bytes at the
read-only observation, including unclassified files and videos. Protected scientific
predictions/identity states and checkpoints are enumerated separately, never deleted.
The old reservations plus the new 2 MiB stage reservation total 26,842,702,814 bytes,
below 26,843,545,600 bytes. No new visual feature bank or GPU task is introduced.

## Implementation references

[MOTIP runtime](https://github.com/MCG-NJU/MOTIP/blob/main/models/runtime_tracker.py)
uses identity prediction, explicit newborn choices and global assignment; its
[ID criterion](https://github.com/MCG-NJU/MOTIP/blob/main/models/motip/id_criterion.py)
masks invalid labels and trains CE/focal losses. This motivates separating a birth
option from identity labels, not importing its detector scores or thresholds.

[NOVA assignment](https://github.com/xifen523/NOVA/blob/main/nova/tracking/assignment.py)
separates legal association masks from costs. Its
[lifecycle](https://github.com/xifen523/NOVA/blob/main/nova/tracking/lifecycle.py)
uses explicit tentative/active/lost states and fixed age rules. Its
[association model](https://github.com/xifen523/NOVA/blob/main/nova/models/association_model.py)
injects geometry into an LM and exposes probabilities. It is a 3D implementation;
this review does not establish matched 2D benchmark superiority.

[SAM3 association](https://github.com/facebookresearch/sam3/blob/main/sam3/perflib/associate_det_trk.py)
distinguishes matched tracks, new detections and overlap relations. Its
[video base](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_video_base.py)
contains tracklet confirmation and duplicate suppression. We borrow the separation
of allocation and duplicate evidence, not its mask-specific thresholds or weights.
