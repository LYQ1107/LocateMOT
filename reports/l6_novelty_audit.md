# Stage L6 Novelty / Collision Audit

Date: 2026-08-11

## 1. Claim under audit

**One shared checkpoint learns a causal identity-dynamics process over a
set of interacting trajectories and generalises across heterogeneous MOT
domains (DanceTrack, BDD100K, MOT17, MOT20, TAO), with persistent
model-in-the-loop memory, learned identity transitions (continue / NEW /
NO-MATCH) and a tracking-level training objective.**

## 2. Closest related works and the collision boundary

| Method (venue/year) | What it does | Overlap with UIDM | Boundary |
|---|---|---|---|
| Samba (ICLR 2025) | End-to-end DETR + synchronized SSM over track queries; persistent hidden state; MaskObs | persistent per-track state; set-of-sequences synchronization | UIDM is detection-agnostic (works on frozen PBD candidates), single shared checkpoint across domains, learns explicit NEW/NO-MATCH/alive transitions with a UniTrack-style loss; no SSM kernel, no detector |
| MOTIP (CVPR 2025) | In-context ID prediction: current objects attend to trajectory history + local ID prompts | sequence-local identity; NEW as output; causal history | MOTIP couples identity to DETR queries and per-video ID vocabulary; UIDM is a transition model over an arbitrary candidate set with persistent recurrence and model-in-the-loop training |
| UniTrack (ICLR 2026) | Framework-agnostic hinge loss on predicted boxes/IDs | tracking-level loss philosophy | UniTrack is a loss only; UIDM is a full dynamics architecture that consumes it as one objective |
| HATReID-MOT (ECCV 2026) | History-conditioned ReID subspace + Hungarian | history-conditioned identity evidence | UIDM does not use a universal ReID space and learns lifecycle/competition in the dynamics, not in the feature space |
| DecoderTracker (PR 2026) | Decoder-only e2e MOT with fixed query memory | persistent memory | detector-coupled, single-domain; no heterogeneous-domain identity-dynamics claim |
| GTR (CVPR 2021) | Offline global trajectory transformer | trajectory-level reasoning | offline; UIDM is online causal |
| MeMOTR / MeMOT (ICCV 2023) | Memory-attention e2e tracking | memory bank | detector-coupled; per-domain; no learned lifecycle transitions or heterogeneous-domain claim |
| CO-MOT | Coopetition label assignment | NEW/birth supervision | assignment machinery inside DETR; UIDM uses structured LSA on learned logits |
| NOVA (2026) | 3D open-vocab autoregressive MOT | "autoregressive identity" concept | 3D + detection; not comparable protocol |
| FARTrack (ICLR 2026) | Autoregressive *visual* tracking (single object) | autoregressive temporal reasoning | single-object; not MOT association |

## 3. Novelty statement (careful, no "first")

To the best of the audited evidence, the **specific combination** of:

1. a **persistent recurrent identity memory over interacting track
   sequences** (Samba-style synchronization realized with standard
   attention, license-safe),
2. **learned causal identity transition matrix** over existing
   tracks / NEW / NO-MATCH (MOTIP-style in-context identity, but over
   arbitrary candidate sets),
3. **model-in-the-loop rollout training** (states produced by the model's
   own associations, scheduled sampling, truncated BPTT),
4. **one shared checkpoint** trained jointly on DanceTrack + BDD100K +
   MOT17 + MOT20 + TAO with no dataset ID / router / adapter, and
5. a **UniTrack-style soft tracking-level objective** (soft ID-switch/FP
   margin + lifecycle + motion)

has not been found in the audited 2025/2026 official repositories.  The
closest is Samba (items 1–2 partly) but it is AGPL, detector-coupled, and
not trained/validated as a one-checkpoint heterogeneous-domain method.

## 4. Risks and how we defend them

- "Just a GRU + attention": we stress the identity-transition formulation
  and cross-domain evidence; ablations isolate memory, interaction,
  transition, objective, lifecycle.
- "Dataset-specific calibration": we never pass a dataset ID; the only
  adaptive element is learned cue reliability from evidence features.
- "Appearance baseline": we compare against the strongest historical
  baselines (C1 motion / L1DK / U0 / Route A), not raw PBD cosine.

## 5. Final verdict

The direction is novel relative to audited official code, with a clear
boundary against Samba/MOTIP/UniTrack.  Scientific success is judged by
Macro HOTA/AssA/IDF1 across the four standard domains (+TAO evidence),
not by drift alone.
