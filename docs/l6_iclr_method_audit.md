# Stage L6: 2025–2026 ICLR/CVPR/ECCV/NeurIPS Method Audit

Audit date: 2026-08-11
Project: LocateMOT Stage L6 — Learned Causal Identity Dynamics Model (UIDM)

The purpose of this audit is not to collect 50 papers.  It answers one
question: **"if we design a modern Unified MOT main model to the
2025/2026 standard, what should its architecture, training objective and
lifecycle mechanism look like?"**  Every design decision in
`docs/l6_uidm_design.md` is traceable to one of the audited official
implementations below.

---

## 1. Verified official repositories (priority 1, author/official org)

### 1.1 UniTrack (ICLR 2026)

- Paper: *UniTrack: Differentiable Graph Representation Learning for
  Multi-Object Tracking* (OpenReview forum id XpddZpGck9, ICLR 2026).
  Repo README also states *"UniTrack: Enhanced Multi-Object Tracking with
  Hinge Loss"*; the code we audited is the hinge-loss universal criterion.
- Official URL: https://github.com/ostadabbas/UniTrack
- Local path: `references/l6/UniTrack-ICLR2026`
- Clone date: 2026-08-11
- Branch: default (main)
- Commit SHA: `afdd9869d31ff115d2fe03b14dd36e0b4f366557`
- License: MIT (repo root `LICENSE`; per-UT README files also MIT)
- Files inspected:
  - `README.md` (integration claims: 7 frameworks)
  - `unitrack_criterion.py` (the full criterion: tracking score,
    spatial consistency, temporal consistency)
  - `INTEGRATION_EXAMPLES.md` and `UT-MOTR/`, `UT-GTR/`, `UT-BYTE/`
    wrappers (how the loss is attached to different trackers)
- Actual architecture: **not an architecture**; it is a framework-agnostic
  differentiable criterion.  Given per-frame predicted boxes + predicted
  track IDs + GT boxes + GT track IDs, it computes:
  1. a tracking score penalising FP / FN / low-IoU localisation / high-IoU
     ID switches (hinge-style penalties),
  2. per-track spatial consistency (box size/shape stability),
  3. per-track temporal consistency (acceleration penalty).
  It does **not** build a graph, does not model persistence, and does not
  replace association.
- What can be adopted: the tracking-level loss philosophy — directly
  penalise ID switches / FP / FN instead of only row-wise classification.
  We will implement a vectorised clean version of the same principle
  (differentiable soft assignment + per-track box smoothness), not copy
  their Python-loop implementation.
- What cannot be adopted: it assumes predicted boxes and predicted track
  IDs already exist; it is a loss plug-in, not an identity dynamics model.
- Collision check: no collision with "learned causal identity dynamics";
  UniTrack is orthogonal (a loss).

### 1.2 Samba / SambaMOTR (ICLR 2025 Spotlight)

- Paper: *Samba: Synchronized Set-of-Sequences Modeling for End-to-end
  Multiple Object Tracking* (arXiv 2410.01806)
- Official URL: https://github.com/mattiasegu/sambamotr
- Local path: `references/association_2025_2026/SambaMOTR`
- Commit SHA: `f1c139a653c00a55c0873a64d7c59a67d7dbad44`
- License: AGPL-3.0 (read-only reference; do not copy)
- Files inspected:
  - `models/query_updater.py` (UnifiedQueryUpdater: active-track
    selection, query embedding update, hidden-state carry)
  - `models/query_updaters/samba.py` (SambaBlock: synchronized Mamba)
  - `models/query_updaters/attention.py` (cross-sequence attention)
  - `models/runtime_tracker.py` (MaskObs uncertainty handling, query
    lifecycle at inference)
- Actual architecture: per-track **query embedding + hidden state +
  conv history** carried across frames; each frame the query updater runs
  Samba (SSM) on every track sequence **synchronously**, with
  self-attention over the per-track **state vectors** so that tracklets
  exchange information at every time step (linear-time cross-sequence
  mixing); MaskObs zeroes uncertain observations; matched detections
  update the query, unmatched tracks keep a decaying state.
- Training formulation: end-to-end DETR-style (track queries, Hungarian
  matching, detection + identity losses).
- What can be adopted: (a) the principle that MOT is a *set of interacting
  sequences* whose per-track states must be synchronised; (b) the
  persistent hidden-state carry; (c) the explicit uncertainty/masking of
  observations.  We do **not** depend on the Mamba SSM kernel (AGPL, CUDA
  build); we implement synchronization with standard attention over
  per-track states, which is a clean, license-safe equivalent of the same
  scientific idea.
- What cannot be adopted: code (AGPL), detector-coupled query decoder,
  dataset-specific pretraining, 3D deformable attention dependency.

### 1.3 MOTIP (CVPR 2025)

- Paper: *Multiple Object Tracking as ID Prediction*
  (arXiv 2403.16848)
- Official URL: https://github.com/MCG-NJU/MOTIP
- Local path: `references/identity_decoding/MOTIP`
- Commit SHA: `ffc0e905ac196a603027eca8d18fb0dff48c8bcc`
- License: Apache-2.0
- Files inspected:
  - `models/motip/id_decoder.py` (IDDecoder: trajectory embeddings +
    one-hot ID labels as in-context prompts, self-attention among current
    objects, cross-attention to trajectory history with **causal
    time mask** (history time <= current time) and relative-position
    embedding, ID classification head)
  - `models/motip/trajectory_modeling.py` (per-trajectory FFN adapter)
  - `models/runtime_tracker.py` (NEW / ID reuse at inference)
  - README, configs
- Actual architecture: current-frame object features are decoded against
  a fixed-length history of trajectory features plus their **sequence
  local ID labels**; prediction is a distribution over
  `num_id_vocabulary + 1` IDs (ID 0 = NEW).  Identity is an in-context
  prompt rather than a universal embedding.  Cross attention is masked so
  future history cannot leak (causal).
- What can be adopted: the **sequence-local identity prompt + causal
  history attention** idea; NEW as an explicit output class; relative-time
  position embedding between current object and history.
- What cannot be adopted: their trajectory features come from a DETR
  decoder trained jointly with detection; we have frozen PBD detections
  and must do online association, not joint detection.
- MOTIP-2 (`references/identity_decoding/MOTIP-2`, commit
  `012856c1`) is the same CVPR 2025 codebase (updated packaging), not a
  distinct second paper.

### 1.4 CO-MOT (ICLR 2025-tracked / CVPR 2023 workshop origin; official repo)

- Paper: *Bridging the Gap Between End-to-end and Non-End-to-end
  Multi-Object Tracking* (arXiv 2305.12724)
- Local path: `references/association_2025_2026/CO-MOT`
- Commit SHA: `1e0618a7bb242a611b24e48b0c5ceab682b8f459`
- License: Apache-2.0
- Files inspected: README (results: DanceTrack HOTA 69.9, BDD100K
  TETA 52.8, MOT17 HOTA 61.1), configs layout.
- Key idea: coopetition label assignment (tracked objects also serve as
  matching targets for detection queries) to fix newborn-query starvation
  in e2e MOT.
- Adopted: the observation that **NEW/birth supervision is a first-class
  training target**; our lifecycle loss supervises NEW and NO-MATCH
  explicitly.  Not adopted: their DETR-specific assignment machinery.

### 1.5 HATReID-MOT (ECCV 2026)

- Paper: *History-Aware Transformation of ReID Features for Multiple
  Object Tracking* (arXiv 2503.12562)
- Local path: `references/association_2025_2026/HATReID-MOT`
- Commit SHA: `3eb440c288bdc5e8548a49c43107f6543c74b264`
- License: to be confirmed per sub-project (read-only)
- Files inspected: README (HAT-MASA, HAT-SORT sub-projects; overview).
- Key idea: transform appearance features into a history-conditioned
  subspace instead of using a fixed universal ReID space.
- Adopted: supports our negative result from L1-B (single-frame universal
  ReID fails) and the design rule that identity representation must be
  history-conditioned.  Not adopted: their ReID+association pipeline.

### 1.6 DecoderTracker (Pattern Recognition 2026; official code in MO-YOLO)

- Paper: *DecoderTracker: Decoder-only end-to-end method for multiple-object
  tracking*
- Local path: `references/l5/MO-YOLO` (AGPL; read-only)
- Commit SHA: `029e23a776ad916d87f27335f804bdb0064d1466`
- Key idea: decoder-only e2e MOT with fixed query memory, weak supervision,
  faster inference.
- Adopted: conceptual only — persistent query/memory reuse across frames.
  No code adopted (AGPL).

### 1.7 Other tracked references (read-only / historical)

- GTR `7138b95b` (global trajectory reasoning, graph transformer for
  offline MOT; confirms graph/trajectory reasoning direction but is
  offline, not causal).
- OC-SORT `8462e7e7` (motion-centric online tracker; confirms motion
  remains a strong cue).
- FARTrack (ICLR 2026, official repo MIV-XJTU/FARTrack): autoregressive
  visual tracking; not MOT association — noted for novelty boundary only.
- NOVA (arXiv 2603.06254, 3D open-vocabulary autoregressive MOT): 3D,
  not our protocol; noted only.
- Dual-Path Temporal Decoder (NeurIPS-track 2025, OpenReview
  T64Fa2hCZn): appearance-adaptive vs identity-preserving dual decoder
  paths; read abstract-level only, no official code audited; noted for
  "identity preservation vs appearance adaptation" separation, which our
  memory update also encodes (identity path keeps anchor, appearance path
  refreshes).

---

## 2. What 2025/2026 methods agree on (synthesis)

1. **Persistent, per-track memory is mandatory.**  Samba carries hidden
   state + conv history; MOTIP carries trajectory feature windows;
   DecoderTracker fixes query memory.  A method that re-encodes 16 frames
   every frame (L5) is not state-of-the-art structure.
2. **Tracks are a set of interacting sequences.**  Samba synchronizes
   state-spaces across tracklets; MOTIP lets current objects compete via
   self-attention before reading history.  Interaction must happen inside
   the temporal model, not only in a post-hoc set encoder.
3. **Identity is sequence-local and in-context, not a universal vector.**
   MOTIP conditions on history + local ID prompts; HATReID transforms
   features by history.  This matches our L1-B negative result.
4. **NEW / NO-MATCH / lifecycle are learned targets, not only thresholds.**
   CO-MOT supervises newborn queries; Samba has explicit uncertainty
   handling (MaskObs) for unmatched tracks.
5. **Losses should target tracking quality (ID switches, continuity),
   not only per-row classification.**  UniTrack's hinge criterion is
   exactly this; it is framework-agnostic and can be attached to our
   learned transition matrix.
6. **Causality is enforced structurally.**  MOTIP masks future history;
   Samba is a causal SSM; our model must likewise use only past states.

---

## 3. Gaps the audited methods leave (our novelty boundary)

- None of the audited methods is a **single shared checkpoint** trained on
  heterogeneous domains (DanceTrack + BDD100K + MOT17 + MOT20 + TAO)
  without dataset-specific heads or dataset routing.
- None models the **causal identity transition matrix** (existing ID →
  candidate / NEW / NO-MATCH) as a differentiable learned dynamics
  process with persistent state, trained with **model-in-the-loop
  rollouts** (Samba/MOTIP are trained teacher-forced within a detector;
  DecoderTracker uses weak supervision but no heterogeneous-domain
  identity-dynamics claim).
- UniTrack is a loss, not a model; Samba is AGPL + DETR-coupled;
  MOTIP is detection-coupled.  A clean, online, detection-agnostic
  **learned causal identity dynamics model** over interacting
  trajectories is therefore a defensible ICLR-level direction.

Detailed novelty collision audit: `reports/l6_novelty_audit.md`.
