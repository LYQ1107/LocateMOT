# Stage L9 — Model Design: Specification-Conditioned Identity Dynamics

Date: 2026-08-15

## 1. Research question

Is multi-object identity dynamics specification-agnostic?  Concretely:
can one learned identity-dynamics process (UIDM core) serve closed-set
category, open-vocabulary category and referring-expression target
specifications from one shared checkpoint, when identity evidence (PBD
box-end) and specification semantics (CLIP + spec token) live in one
unified observation space?

## 2. Why a gated residual (and not a new architecture)

L8 evidence:

- B2 (identity-pure core, semantics only in the relevance head):
  cleanest WHAT/HOW decoupling; ordinary Macro AssA 0.5045; OVMOT
  TETA 34.33 / AssocA 30.44; RMOT HOTA 35.20.
- B1 (semantics added to the identity token stream): ordinary Macro
  AssA 0.5087; RMOT HOTA 37.88 / AssA 31.02; OVMOT slightly lower
  (TETA 34.07).

Both are positive, so the question is not whether semantics may enter the
identity stream, but *when and how much*.  A fixed addition (B1) cannot
shut semantics off in easy identity regimes; a fully separated core (B2)
cannot use semantics for identity disambiguation.  The L9 main model adds
a learned per-candidate gate:

    z_id   = PBD_Proj(pbd)                       (identity evidence)
    sem    = CLIP_Proj(clip) + sigmoid(g1) * Spec_Proj(spec)
    gate   = sigmoid(MLP([z_id, sem]))            (per-candidate)
    z      = z_id + gate * W(sem)                 (core input)

The relevance head still sees the ungated `sem`, so WHAT (target
selection) and HOW (identity) remain decoupled in supervision.  The gate
is initialised so the first forward equals L8-B1 (`sigmoid(1) ~ 0.73`,
`W = I`); training can open or close the gate per candidate/regime.

Design evidence from the L9 audit:

- QTrack (2026, Apache-2.0) confirms language-conditioned identity
  association is an active direction, but uses a 3B VLM + RL (RMOT only),
  not a shared identity core.
- TRACT (ICCV 2025) supports trajectory-aware feature aggregation as a
  memory principle; already embodied in UIDM's persistent memory.
- MOTIP (CVPR 2025) supports identity-as-prediction; already the basis of
  UIDM's Existing/NEW/NO-MATCH transitions.
- No published work unifies one identity core + one checkpoint across
  closed/open/referring MOT; the gated residual is our own clean design
  (no external code copied).

## 3. Why not the other L9 options

- FiLM modulation (Option 2): global per-spec scale/bias would couple all
  candidates of a frame and cannot decide per-candidate whether identity
  evidence is sufficient.
- Small cross-attention (Option 3): heavier, needs track state inside the
  adapter and complicates the online tracker; the benefit over a
  per-candidate gate is speculative at this stage.
- Trajectory-conditioned residual (Option 4): could help in principle,
  but requires re-architecting the UIDM rollout and is deferred to L10;
  the per-candidate gate already tests the scientific claim with minimal
  risk.

## 4. Training protocol (L9 main)

- Init: L8-B1 `uidm_l8_final/latest.pt` (large core, 384-d, 6 layers).
- Trainable: full UIDM + adapter + gate (~19.9 M params; total model
  params reported at final run).
- Data: task-balanced 1:1:1 (target) MOT (DanceTrack/BDD/MOT17/MOT20
  CLIP caches), OVMOT (TAO train DLA dets + CLIP + PBD, when available),
  RMOT (Refer-Dance).  No dataset one-hot.
- PBD-dropout 0.15 so the same core also serves the PBD-zero regime.
- Loss: UIDM row/col CE + lifecycle BCE + motion L1 + soft switch +
  relevance BCE (w_rel=0.2), teacher-student rollout as L6/L8.
- Seed 20260806; single main run; resume support.

## 5. Ablation plan (4 groups)

1. identity-only (`mode=identity`)
2. semantic-only (`mode=semantic`)
3. strict decoupled B2 (`sem_in_core=False`, no gate)
4. L9 main (`cond_gated=True`)

Each evaluated on ordinary TrackEval (DanceTrack/BDD/MOT17/MOT20), TAO
official TETA (val, full PBD + CLIP once cache is ready), and Refer-Dance
RMOT TrackEval (40 GT queries).

## 6. Expected cost

- L9 main training: 2-4 GPUs, ~10k-40k steps, ~6-30 h wall-clock.
- TAO val PBD cache: ~36,375 frames x ~44 crops x ~0.25 s/crop;
  parallelised across up to 8 workers under the 4-GPU cap.

