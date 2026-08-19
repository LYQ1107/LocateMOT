# Stage L11 — Pseudo-Track Design

Date: 2026-08-19

## Problem

L10 expanded OVMOT supervision (500 TAO train videos / 18,274 frames /
322,843 training candidates) collapsed association (AssocA 7.26 / 7.86)
because only 3.46% of raw DLA candidates match C-TAO base GT, and the
old target scheme marked every unmatched detection as relevance
positive + NEW birth.  L11 replaces "unmatched -> NEW" with a
high-precision temporal pseudo-track supervision signal.

## Data sources

- DLA candidates: `outputs/l10/cache/tao_train_candidates/`
  (Detic-SwinB MASA checkpoint, same family as TAO-val public dets).
- Training pkls: `outputs/l10/data/tao_train/*.pkl` with boxes, det
  scores, LVIS labels, C-TAO base `cand_gt`, `gt_boxes`, CLIP [512] and
  full-PBD [2048] features (1.6 GB total).
- Dense GT: C-TAO base_and_novel
  (`LocateMOT_reference_repos/covtrack/saved_models/ctao_dataset/
  ctao_base_and_novel.json`, 500 videos / 490,210 frames; used for
  class-A matching and evaluation-only latent identity).

## Candidate classes

- **A (GT-matched)**: greedy one-to-one IoU >= 0.30 match to C-TAO
  base_and_novel (same matcher as L10 but with denser GT and a
  threshold justified by DLA/C-TAO box-style mismatch).  On the 8-video
  calibration set this raises coverage from ~10% (L10 base, IoU 0.5) to
  ~28%.  Full normal supervision; relevance 1.0.
- **B (high-confidence pseudo-track)**: unmatched candidates linked
  forward and backward with cycle consistency, appearance
  (CLIP+PBD cosine), motion (constant-velocity IoU gate), category
  consistency, detector score; confidence-weighted pseudo identity.
- **C (uncertain)**: any unmatched candidate without a reliable
  tracklet, and every GT-overlapping duplicate (max GT IoU >= 0.30 but
  not matched one-to-one).  `ignore_flag=1`; identity/lifecycle IGNORE;
  NEVER NEW; relevance 0.
- **D (clear negative/background)**: no separate class is needed in this
  stream; candidates rejected by the above rules receive relevance 0 and
  identity IGNORE, which is the conservative choice given no
  background-label source.

## Linking algorithm (clean reimplementation)

1. For each video, each frame, each candidate:
   - Greedy match to C-TAO base_and_novel at IoU >= 0.30 (class A).
   - Compute `gt_max_iou` (max IoU to any GT box) for exclusion.
2. Suppress near-duplicate unmatched detections: same LVIS label and
   IoU >= 0.50, keep highest det score.
3. Forward linking (greedy, one-to-one):
   - Active tracklets with last observation <= 2 pkl-steps ago.
   - Same LVIS category required.
   - Motion gate: IoU(det, constant-velocity predicted box) >= 0.15.
   - Appearance gate: mean (CLIP + PBD) cosine >= 0.70 (in [0,1]).
   - Combined score = 0.25*motion + 0.45*appearance + 0.20*category
     + 0.10*gap-penalty; threshold 0.62.
4. Backward cycle check: each forward link (a -> b) must be the best
   backward link from b's frame (score ties broken by identity); a link
   is kept only if confirmed.  This is the Walker-style bidirectional
   consistency.
5. Tracklet filter: length >= 3 observations, mean appearance >= 0.80,
   cycle pass rate >= 0.80, mean det score >= 0.25.
6. Confidence = min(0.97, mean_gen * mean_app * sqrt(cycle_rate)).
7. Training sidecar (`pseudo_id`) is assigned only to candidates with
   `gt_max_iou < 0.30`.  Candidates in the same tracklet but overlapping
   GT are dropped from pseudo supervision (their identity is ambiguous
   with an existing GT track) while the raw linker output is preserved
   as `link_id` for the quality audit.

## NEW / NO_MATCH supervision rules (as implemented in trainer changes)

- NEW is only taught when a candidate has a GT id or a reliable pseudo
  id whose first observation has no trusted predecessor in the window
  (birth_confidence).
- Existing/NO_MATCH/reactivation uses the same lifecycle machinery as
  GT tracks (slot persists while alive; NO_MATCH when no candidate is
  observed within MAX_AGE; reactivation when the id reappears).
- Unmatched candidates without pseudo id are excluded from col_valid
  and NEW; their relevance target is 0.

## Output format

Sidecar per video (`outputs/l11/data/pseudo_tracks/<video>.pkl`):

- `gt_id [N]` : class-A GT track id or None
- `pseudo_id [N]` : final training pseudo id or None
- `link_id [N]` : raw linker id (for audit only)
- `pseudo_conf / birth_conf / cont_conf [N]`
- `ignore_flag [N]` : 1 = identity/lifecycle IGNORE
- `rel_target [N]` : 1.0 (GT), pseudo_conf (pseudo), 0.0 (other)
- `tracklet_stats` : per-tracklet length / cycle_rate / mean_app /
  mean_gen / frames

## Why this design (vs alternatives)

- IoU-only Hungarian pseudo labels are explicitly shown to fail under
  fast motion/occlusion by COVTrack++ (arXiv 2603.24016); we add
  appearance, category, motion and cycle filters.
- U2MOT shows risky associations should be excluded rather than
  reweighted into supervision; our uncertain class is IGNORE.
- Walker shows bidirectional graph walks improve identity consistency;
  our cycle check is the minimal version of that idea.
- PS-MOT shows confidence-gated pseudo labels with temporal feedback
  are reliable; we keep only high-confidence tracklets and weight their
  transitions by confidence.
