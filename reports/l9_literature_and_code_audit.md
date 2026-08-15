# Stage L9 — Literature and Code Audit (2025/2026)

Date: 2026-08-15

All entries below were verified by opening the linked page / cloning and
reading the linked repository.  No method is cited from chat memory alone.
The full L8 audit (`reports/l8_literature_and_code_audit.md`) remains valid
for RMOT/TransRMOT, iKUN, TempRMOT and MOTIP; this document adds the
2025/2026 OVMOT / unified-MOT / RMOT / RL references checked for L9.

## 1. OVTR — End-to-End Open-Vocabulary MOT

- Paper: "OVTR: End-to-End Open-Vocabulary Multiple Object Tracking with
  Transformer", Jinyang Li, En Yu, Sijia Chen, Wenbing Tao; ICLR 2025.
- Paper URL: https://arxiv.org/abs/2503.10616
- Official GitHub: https://github.com/jinyanglii/OVTR
  - Local clone: `LocateMOT_reference_repos/ovtr`
  - Commit: `500e72c19bf5f7f8717546911a5639fdc26bfee5`
  - License: MIT
- Mechanism inspected (files):
  - `ovtr/models/ovtr.py` (OVTR model, forward/track loops)
  - `ovtr/models/transformer.py` (text-conditioned transformer decoder)
  - `ovtr/models/fuse_modules.py` (visual-language fusion)
  - `ovtr_det_bs2_pretrain/` (detection pretrain)
- Observed implementation: query-based DETR tracker whose decoder is
  conditioned on CLIP text embeddings; category information propagation
  (CIP) across frames; attention isolation so open-vocabulary perception
  and tracking coexist.  It is an OVMOT-specific end-to-end model trained
  on TAO/LVIS.
- What we borrow conceptually: the finding that language/CLIP semantics can
  be injected into a transformer tracking decoder, and that OVMOT can be
  end-to-end.  We do **not** copy OVTR code or its DETR-style query
  formulation: our identity core is the causal UIDM with persistent memory
  and lifecycle, and our claim is one shared core across three
  formulations (closed-set MOT / OVMOT / RMOT), which OVTR does not target.

## 2. TRACT — Trajectory-Aware Open-Vocabulary Tracking

- Paper: "Attention to Trajectory: Trajectory-Aware Open-Vocabulary
  Tracking", Yunhao Li, Yifan Jiao, Dan Meng, Heng Fan, Libo Zhang;
  ICCV 2025.
- Paper URL: https://arxiv.org/abs/2503.08145
- Official GitHub: https://github.com/Nathan-Li123/TRACT
  - Local clone: `LocateMOT_reference_repos/tract`
  - Commit: `19f01d72f9f6c212c28fd9cb0171a5432cd41a6a`
  - License: **not detected in the repo root** (no LICENSE file found;
    recorded as unknown — no code reused).
- Mechanism inspected (files):
  - `TraCLIP/` (trajectory-aware CLIP: TFA/TSE training scripts and
    tracklet extraction tools)
  - `masa/` (universal appearance model for association)
- Observed implementation: plug an open-vocabulary detector into a
  trajectory-aware association pipeline (MASA) and a trajectory-aware CLIP
  classifier; trajectory consistency reinforcement (TCR), trajectory-aware
  feature aggregation (TFA), trajectory semantic enhancement (TSE).
- What we borrow conceptually: trajectory-level feature aggregation and
  the idea that identity association should use trajectory memory rather
  than single-frame appearance.  This supports the UIDM's persistent
  track-memory design (already present since L6).  No code copied.

## 3. AED — Associate Everything Detected

- Paper: "Associate Everything Detected: Facilitating Tracking-by-Detection
  to the Unknown", Zimeng Fang et al.; arXiv:2409.09293, IEEE TIP 2025.
- Official GitHub: https://github.com/balabooooo/AED
  - Local clone: `LocateMOT_reference_repos/aed`
  - Commit: `e9c0c7f1884fdcf76c24747d4f3e8245dcfb1064`
  - License: MIT
- Mechanism inspected (files):
  - `models/aed.py`, `models/attention.py`, `models/query_updating.py`,
    `models/deformable_transformer_plus.py`
- Observed implementation: tracks with any off-the-shelf detector; models
  association as similarity decoding with spatial, temporal and cross-clip
  similarities, trained with association-centric learning.  It unifies
  closed-vocabulary MOT and OV-MOT, but relies on a similarity decoder
  rather than a learned identity-dynamics process, and does not support
  referring-expression specifications.
- What we borrow conceptually: association-centric training objectives and
  the spatial/temporal/cross-clip similarity view; already reflected in the
  UIDM's set-level competition and memory design.  No code copied.

## 4. QTrack — Query-Driven Reasoning for Multi-modal MOT

- Paper: "QTrack: Query-Driven Reasoning for Multi-modal MOT", Tajamul
  Ashraf et al.; arXiv:2603.13759 (2026), with the RMOT26 benchmark.
- Paper URL: https://arxiv.org/abs/2603.13759
- Project page: https://gaash-lab.github.io/QTrack/
- Official GitHub: https://github.com/gaash-lab/QTrack
  - Local clone: `LocateMOT_reference_repos/qtrack`
  - Commit: `bc746fe246217a4de0ecac0318ba1cf9be94a604`
  - License: Apache-2.0
- Mechanism inspected (files):
  - `README.md` (method summary, RMOT26 benchmark, TPA-PO RL)
  - `verl/` (RL training stack: TPA-PO with structured rewards)
  - `training_scripts/`, `prepare_dataset/`
- Observed implementation: a 3B VLM that performs query-driven MOT from
  natural-language instructions, trained with Temporal Perception-Aware
  Policy Optimization (TPA-PO) using structured rewards (motion-aware
  reasoning / identity coherence).  It is RMOT-centric (query-driven
  tracking of specified targets) and does not share one identity-dynamics
  core across closed-set MOT, open-vocabulary MOT and RMOT.
- What we borrow conceptually: (a) evidence that language-conditioned
  identity association is an active 2026 research direction; (b) for the
  future-RL record (`docs/future_rl_reference.md`), the existence of
  official RL/GRPO-style tracking code with structured rewards.
- What we do **not** adopt in L9: a large VLM backbone or RL training
  (L9 keeps frozen CLIP + frozen LocateAnything-3B + small trainable UIDM;
  RL remains outside this stage).

## 5. MOTIP / iKUN / TransRMOT (L8 re-verification)

These were re-verified in L8 (same commits and licenses as recorded in
`reports/l8_literature_and_code_audit.md`).  MOTIP (CVPR 2025,
identity-as-ID-prediction) remains the closest published mechanism to our
UIDM identity transitions; iKUN remains the closest RMOT baseline.

## 6. Novelty status after the 2025/2026 audit

After inspecting OVTR, TRACT, AED, QTrack, MOTIP, iKUN and TransRMOT, we
did **not** identify a published, verifiable system that:

1. uses **one trained identity-dynamics core** with persistent memory,
   lifecycle, Existing/NEW/NO-MATCH, and set-level competition;
2. serves **one shared checkpoint** across closed-set MOT, open-vocabulary
   MOT and referring-expression MOT;
3. is driven by a **unified frozen observation space** (PBD identity token
   + CLIP semantic token + specification token).

The nearest neighbours cover only one or two of these: OVTR/TRACT are
OVMOT-only; AED covers CV+OV but no referring language and no learned
identity-dynamics/lifecycle; QTrack is query-driven RMOT with a large VLM,
not a shared identity core; MOTIP is ordinary-MOT ID prediction.
Therefore L9's claim will be phrased as "we did not identify ...", not
"first".

