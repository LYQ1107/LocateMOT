# Stage L11 — Temporal Pseudo-Track Supervision Literature & Code Audit

Date: 2026-08-19
Project: LocateMOT (Stage L11, OVMOT temporal-supervision repair)

## Purpose

L10 established that the expanded TAO-train OVMOT stream collapses
association (AssocA 7.26 / 7.86 vs L9's 29.34) because only ~3.46% of
DLA candidates match C-TAO base GT, while the old target scheme taught
every unmatched detection as a NEW birth.  Stage L11 must replace
"unmatched -> NEW" with high-precision temporal pseudo-tracklets.
This audit records the real 2024–2026 literature and official code that
informs the pseudo-track design, per the project's external-reference
rules.  All claims below were verified against the cited URLs, local
clones, or both; no paper/GitHub is invented.

## Summary of what is borrowed

- **U2MOT-style uncertainty-aware association**: verify risky inter-frame
  links; only confident links become pseudo-tracklets.  We adopt the
  principle (not code).
- **Walker-style bidirectional/cycle consistency**: forward link must be
  confirmed by backward link; random-walk/cycle pass rate is used as a
  tracklet-quality filter.
- **COVTrack++ evidence**: IoU-only pseudo labels suffer propagation
  errors under fast motion/occlusion, so we use appearance + motion +
  category + cycle filters instead of IoU-only Hungarian.
- **PS-MOT style temporal feedback**: pseudo labels should be refined
  with temporal evidence (prompt evolution in PS-MOT; here, temporal
  linking of DLA candidates) and only high-confidence labels are kept.
- **MASA/OVTrack protocol**: the TAO val evaluation uses public
  Detic-SwinB detections; the train pseudo-tracklets are built on the
  same detector family so the observation distribution matches.

## References (verified)

### 1. U2MOT — Uncertainty-aware Unsupervised Multi-Object Tracking

- Paper: arXiv:2307.15409 (2023); ICCV 2023 (official code README).
- Official URL: https://github.com/alibaba/u2mot
- Local clone: `LocateMOT_reference_repos/u2mot`
- Commit: `7411211d17cb893f5fcd6be39cd4e5f91cfe1586` (2026-08-19 clone)
- License: MIT (LICENSE file present).
- Relevant mechanism (read from `yolox/core/trainer.py`,
  `yolox/tracker/u2mot_tracker.py`, README):
  - Unsupervised tracker builds frame-by-frame associations with a
    tracklet-guided augmentation objective.
  - `uncertainty` values are computed from the matching risk
    (`risk - threshold`); risky matches are excluded from supervision
    (`uncertain` matches become unpaired), and uncertainties are stored
    per sample for loss reweighting.
  - Pseudo-tracklets are used as supervision for the embedding head; the
    quality of tracklets is explicitly controlled.
- Borrowed design: uncertainty-gated link acceptance; only low-risk
  links form pseudo-tracklets; uncertain candidates are IGNORE (not
  NEW).  No code copied.

### 2. Walker — Self-Supervised MOT by Walking on Temporal Object Appearance Graphs

- Paper: ECCV 2024 (Segu et al.; DOI 10.1007/978-3-031-73242-3_1).
- Official URL: https://github.com/mattiasegu/walker
- Local clone: `LocateMOT_reference_repos/walker`
- Commit: `27006bca95b7eaad02c1f6e9c3f3fd1cb48e90a5` (2026-08-19 clone)
- License: no LICENSE file; README only (code marked "coming soon").
- Relevant mechanism (paper, verified via ECCV 2024 PDF snippet):
  - Quasi-dense Temporal Object Appearance Graph (TOAG) connects object
    RoIs on key/reference frames.
  - Multi-positive contrastive objective optimizes random walks on the
    graph; walks enforce temporal identity consistency (bidirectional
    paths).
- Borrowed design: forward-backward walk / cycle consistency as a
  precision filter.  No code available, so this is a clean
  reimplementation of the mechanism only.

### 3. COVTrack (ICCV 2025) + COVTrack++ (arXiv 2026)

- COVTrack paper: ICCV 2025 (Qian et al.), "Continuous Open-Vocabulary
  Tracking via Adaptive Multi-Cue Fusion".
  - Official URL: https://github.com/zekunqian/COVTrack (repo mirrored at
    `LocateMOT_reference_repos/covtrack`)
  - Local commit: `9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b`
  - License: Apache-2.0 (LICENSE present).
  - C-TAO annotations (used by L10/L11 as the dense GT source) come from
    this project; `ctao_base.json` covers 500 videos / 490,210 frames /
    1,489,637 boxes / 2,588 tracks.
- COVTrack++ paper: arXiv:2603.24016 (2026), "Learning Open-Vocabulary
  Multi-Object Tracking from Continuous Videos via a Synergistic
  Paradigm" (Qian, Feng, Han, Hou).
  - Official code: not yet found (paper says "code will be available").
  - Verified quote from arXiv HTML (ar5iv):
    "Pseudo labels, while providing temporal continuity, suffer from
    propagation errors and lack temporal consistency, especially in
    dynamic scenarios with rapid motion or severe occlusions where IoU
    matching becomes unreliable."
  - Also: Temporal Confidence Propagation (TCP) recovers flickering
    detections via high-confidence tracked objects.
- Borrowed design: do NOT use IoU-only pseudo labels; combine motion,
  appearance, semantic cues; use temporal confidence propagation for
  short flickers.  C-TAO remains the GT basis for matched candidates.

### 4. MASA — Matching Anything by Segmenting Anything

- Paper: CVPR 2024 (Li et al.).
- Official URL: https://github.com/siyuanliii/masa (mirror at
  `/data1/LWR/vranlee/SERVER_ONLY/avis/masa`).
- License: MIT (per repo).
- Relevant mechanism: universal instance appearance model trained
  without tracking labels via exhaustive data transformations; MASA
  adapter enables any detector to track.  MASA's TAO val protocol
  (public Detic-SwinB detections) is the protocol used for our OVMOT
  evaluation.
- Borrowed design: protocol alignment (candidate distribution) rather
  than code; the pseudo-track generator runs on the same DLA dets as
  evaluation.

### 5. OVTrack

- Paper: CVPR 2023 (Li et al.).
- Official URL: https://github.com/SysCV/OVTrack
- Local copy: `LocateMOT_reference_repos/covtrack/ovtrack` (vendored in
  COVTrack repo); also `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/ovtrack`.
- License: Apache-2.0.
- Relevant mechanism: open-vocabulary tracker with region-level
  appearance + semantic fusion; TETA evaluation protocol on TAO.
- Borrowed design: evaluation metric/protocol (already used in L7–L10).

### 6. MOTIP / MOTIP-2 — Multiple Object Tracking as ID Prediction

- Paper: CVPR 2025.
- Official URLs:
  - https://github.com/MCG-NJU/MOTIP
  - https://github.com/GISer-WB/MOTIP-2
- Local clone: `LocateMOT_reference_repos/motip`
- Commit: `ffc0e905ac196a603027eca8d18fb0dff48c8bcc`
- License: Apache-2.0 (LICENSE present).
- Relevant mechanism (read from README/code): in-context ID prediction;
  trajectories carry ID information; current detections are decoded
  against history; NEWBORN is an explicit output with a threshold.
- Borrowed design: identity is a persistent track-level concept; NEW is
  a rare, explicit transition rather than a per-detection default.
  (Architecture already reflected in UIDM since L6.)

### 7. LaMOT — Language-Guided Multi-Object Tracking

- Paper: ICRA 2025 (Li et al.).
- Official URL: https://github.com/Nathan-Li123/LaMOT
- Local clone: `LocateMOT_reference_repos/lamot`
- Commit: `5300242c06381c8d5ad36865dd88f4b62c2482bc`
- License: no LICENSE file found in repo (annotation-only release).
- Relevant mechanism: 1,660 sequences / 18.9k trajectories of
  language-guided MOT; annotations are `{targets: {frame: [track_ids]},
  language}`; source datasets include TAO, MOT17, SportsMOT, VisDrone.
- Borrowed design: RMOT evaluation/generation conventions (already used
  in L8/L10); no code copied.

### 8. PLOT — Pseudo-Labeling via Object Tracking (monocular 3D detection)

- Paper: arXiv:2507.02393 (2025); third-party listing as ECCV 2026 not
  independently verified from an official page.
- Official URL: https://arxiv.org/abs/2507.02393
- License/code: not available at audit time (paper describes
  training-free framework).
- Relevant mechanism: dense point tracking -> background/object
  trajectory decomposition -> global object memory for long-range
  identity consistency.
- Borrowed design: trajectory-level memory for identity consistency
  (3D-specific details not adopted).

### 9. PS-MOT — Point-Supervised MOT with Temporal-Feedback Prompting

- Paper: arXiv:2606.30476 (2026).
- Official URL: https://github.com/xifen523/PS-MOT
- Local clone: `LocateMOT_reference_repos/ps_mot`
- Commit: `163e9ee31967e6b043b4d71917e833704cda4d19`
- License: MIT (LICENSE present).
- Relevant mechanism (read from `generate_mot_pseudo_boxes.py`):
  - Static point annotations are turned into temporally consistent
    pseudo boxes with SAM; Gaussian noise is added to the prompt center,
    negative prompts within a radius are used, and retries with offsets
    keep only confident masks.
  - TFP (Temporal-Feedback Prompting) uses negative spatial cues and
    motion priors to evolve points into consistent pseudo-labels.
- Borrowed design: temporal feedback and confidence-gated pseudo-label
  acceptance; relevant again in L12 for point-prompt interface.
  No code copied.

### 10. PL-MCT — Pseudo-Labeling and Multi-Frame Consistency Training

- Paper: The Visual Computer 41(6), 2025 (Zhao et al.).
- Official URL: https://github.com/HYQ-hyq222/PL-MCT
- Local clone: `LocateMOT_reference_repos/pl_mct`
- Commit: `782c1fc4ba6f1d7f5318d085eccf11e4b3386d06`
- License: no LICENSE file found at audit time.
- Relevant mechanism: pseudo-labeling + multi-frame consistency
  regularization for semi-supervised (single-object) visual tracking.
- Borrowed design: multi-frame consistency as a filter principle.
  No code copied.

## Search topics actually run (2026-08-19)

pseudo track supervision MOT; temporal pseudo labels multi-object
tracking; self-supervised tracklet association; semi-supervised MOT
pseudo track; open-vocabulary MOT pseudo labels; TAO temporal pseudo
tracking; trajectory self-training open-vocabulary tracking;
cycle-consistent tracklet pseudo labels; teacher-student temporal
tracking; high precision tracklet generation MOT filtering.

No official 2025/2026 system was found that publishes exactly
"high-precision temporal pseudo-track supervision for sparsely
annotated OVMOT"; the closest verified systems are U2MOT (unsupervised
pseudo-tracklets), Walker (cycle-consistent graph walks), PS-MOT
(temporal pseudo boxes), and COVTrack++ (evidence that IoU-only pseudo
labels fail).

## Design implication for L11 pseudo-track generator

The L11 generator will implement (all as clean reimplementation):

1. Per-frame DLA candidates with C-TAO/TAO GT match (A) keep GT ids.
2. Unmatched candidates (B/C/D) are linked forward by a motion-gated
   affinity (IoU + constant-velocity motion + CLIP/PBD cosine +
   category consistency), then backward (reverse) links must confirm
   forward links (cycle consistency, Walker principle).
3. Tracklets are filtered by: length >= 3 observations, appearance
   self-consistency, motion smoothness, category purity, cycle pass
   rate, and minimum mean confidence (U2MOT/PS-MOT confidence gating).
4. Remaining uncertain candidates are IGNORE for identity/lifecycle and
   relevance-negative unless clear-background evidence exists; they are
   never NEW by default.
5. GT-matched candidates keep full supervision; high-confidence pseudo
   tracklets receive confidence-weighted transition supervision.
