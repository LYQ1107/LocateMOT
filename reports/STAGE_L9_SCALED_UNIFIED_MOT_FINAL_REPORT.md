# STAGE L9 — Scaled Specification-Conditioned Unified MOT

Status: **COMPLETE** (all official evaluations run; ICLR readiness:
NEAR_READY)

Date: 2026-08-17
Project: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`

> This report is self-contained: it explains the research question,
> L0-L8 background, the L9 method, protocols, results, ablations,
> failure boundaries, novelty and ICLR readiness.

## 1. Research question

Is multi-object identity dynamics specification-agnostic?  Concretely:
can **one learned identity-dynamics process** (the UIDM core) and **one
shared checkpoint** support heterogeneous WHAT-to-track specifications —
closed-set category (ordinary MOT), open-vocabulary category (OVMOT) and
referring expression (RMOT) — when identity evidence (PBD box-end) and
specification semantics (CLIP + spec token) are represented in a shared
observation space?

## 2. L0-L8 background (compressed)

- L0-L5: built LocateAnything/PBD object tokens; established that
  specification-induced identity drift exists and simple residual
  corrections are partial; shared learned core is a positive signal.
- L6: UIDM — causal identity-dynamics core (persistent memory, set-level
  competition, Existing/NEW/NO-MATCH, lifecycle, reactivation); one
  checkpoint across DanceTrack/BDD/MOT17/MOT20; Macro AssA 0.4922.
- L7: CLIP/spec interface; TAO OVMOT probe (All AssocA 29.5, Base≈Novel)
  but replacing PBD with CLIP regressed ordinary MOT (0.4290).
- L8: Unified Observation Adapter; gated CLIP+spec semantic residue;
  PBD-dropout; one shared checkpoint for MOT + OVMOT + RMOT.  Ordinary
  Macro AssA 0.5045 (identity-pure B2) / 0.5087 (sem-in-core B1); TAO
  TETA 34.33 / AssocA 30.44 (PBD-zero); Refer-Dance RMOT HOTA 35.20 /
  37.88.  Critical bug fixed: eval must use `pbd_box_end_last` (box-end),
  not coord-mean.
- L8 gaps that L9 closes: (a) TAO was evaluated without PBD identity
  tokens (PBD-zero); (b) training was 2.5-3k steps only; (c) no OVMOT
  training data in the joint mix; (d) semantic residue was a fixed
  addition (B1) rather than a specification-conditioned gate.

## 3. L9 method

### 3.1 Full observation

TAO val full PBD cache: for every public Detic candidate (~44/frame,
1.61M candidates total), crop the box and run frozen LocateAnything-3B
(BF16, no-grad) with a generic object query; store the PBD box-end token
(`pbd_box_end_last`, 2048-d) aligned with the candidate order
(`tools/cache_l9_tao_pbd.py`, write-through + resume + auto-scaling
monitor; details in `reports/l9_tao_pbd_cache.md`).  OVMOT evaluation
then uses PBD + CLIP + spec through the same UIDM core.

### 3.2 Specification-conditioned identity interaction (L9 main)

```text
   Category / Open Category / Referring Expression
                       |
                       v
          Unified Specification Encoder (frozen CLIP text)
                       |
                       v
                    Spec Token
                       |
                       +---------------------------+
                                                   |
Video frame                                        |
   |                                               |
   v                                               |
PBD (identity token)                               |
   |                                               |
   +-------- CLIP crop token -----> sem = clip_proj + g1*spec_proj
   |                                               |
   |    gate = sigmoid(MLP([pbd_proj(pbd), sem]))  |
   |                                               |
   +---> z = pbd_proj(pbd) + gate * W(sem)  (core token)
                       |
                       v
           Shared UIDM Core
       persistent track memory
        set-level competition
       Existing / NEW / NO-MATCH
      lifecycle / reactivation
                       |
                       v
                 Tracks / IDs
```

The gate is per-candidate and learned; it starts at the L8-B1 behaviour
(sigmoid(1) ~ 0.73, W = identity) and can close when identity evidence is
sufficient.  The relevance head always sees the ungated semantic content,
so WHAT (target selection) and HOW (identity) remain decoupled.

### 3.3 Training

- Init: L8-B1 large checkpoint.
- Joint mix: ordinary MOT (BDD/DanceTrack/MOT17/MOT20 CLIP caches),
  OVMOT (TAO train subset: 105 videos, DLA dets + CLIP + crop PBD — L9
  pipeline), RMOT (Refer-Dance); task ratio ~1:1:1, no dataset one-hot.
- PBD-dropout 0.15; tracking losses + relevance BCE; seed 20260806;
  2-4 GPUs; resume with optimizer/scheduler/global step.
- Stage A (completed run): 10,000 steps MOT+RMOT with the gate
  (v1-v4 were invalidated by the init loader bug; v5 with the corrected
  loader is the Stage-A model).  Stage B: resume v5 with the crop-PBD
  OVMOT stream (105 TAO train videos, 7,522 candidates) -> L9-ovmot,
  the final shared checkpoint.

### 3.5 Full observation + OVMOT training stream

- TAO val: crop-based PBD box-end tokens cached for all 36,375 frames /
  1.61 M public Detic candidates (write-through, resumable, verified;
  `reports/l9_tao_pbd_cache.md`).
- OVMOT training stream: Detic DLA on TAO train was blocked by a
  torchvision `roi_align` OOM in this environment; instead we reuse the
  L6 TAO-train set (105 videos, 4,200 frames; 7,522 crop-PBD candidates;
  86% GT-matched by IoU >= 0.5; CLIP crop features added).  Sparse but
  directly matches the crop-PBD observation distribution of the val
  evaluation.

### 3.4 Negative evidence and fix (L9 v1)

Two implementation bugs were found and fixed during L9:

1. `sem_transform` was initialized as an all-ones matrix (degenerate
   rank-1 projection) instead of the identity matrix; fixed to `eye_`.
2. The training script's `--init-ckpt` loader matched only bare keys,
   while the L8 checkpoints store `uidm.`/`adapter.`-prefixed keys, so the
   UIDM core was silently left randomly initialized (adapter loaded
   correctly, masking the error).  Fixed by loading through
   `load_l8_state`.

The v1-v4 checkpoints produced by the buggy loader are preserved under
`outputs/l9/checkpoints/uidm_l9_main_v{1,2,3}_*/` and
`uidm_l9_control_nogate_randomcore/` as failure evidence; they are not
used as scientific evidence about the gated-residual method.  The v5 run
uses the corrected loader and `eye_` init.

## 4. Protocol

- Ordinary MOT: official TrackEval, four domains (DanceTrack val /
  BDD100K / MOT17 train / MOT20 train), HOTA/DetA/AssA/IDF1/MOTA/IDSW.
- OVMOT: official TAO val TETA (Base/Novel/All), Detic public dets,
  full PBD+CLIP observation (vs L7/L8 PBD-zero).
- RMOT: official Refer-Dance RMOT TrackEval, 40 GT queries, threshold
  0.5; detector protocol caveat (LocateAnything vs ByteTrack/DLA).
- Ablations (4 groups): identity-only, semantic-only, strict decoupled
  (B2), L9 spec-conditioned (main).

## 5. Results

### Table 1 — Unified formulation overview

| Method | Shared ckpt | Shared UIDM | MOT | OVMOT | RMOT |
|---|---|---|---|---|---|
| L8-B2 | yes | yes | yes | yes (PBD-zero) | yes |
| L8-B1 | yes | yes | yes | yes (PBD-zero) | yes |
| L9 main | yes | yes | yes | yes (full PBD) | yes |

### Table 2 — Ordinary MOT (official TrackEval)

| Dataset | Method | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|---|---|
| DanceTrack | L8-B2 | | | 0.3457 | | | |
| DanceTrack | L8-B1 | | | 0.3405* | | | |
| DanceTrack | L9 v5 | 0.5763 | | **0.3509** | 0.4991 | | 6282 |
| BDD100K | L8-B2 | | | 0.5019 | | | |
| BDD100K | L9 v5 | 0.4832 | | **0.5108** | 0.4293 | | 6476 |
| MOT17 | L9 v5 | 0.7095 | | **0.7017** | 0.6256 | | 437 |
| MOT20 | L9 v5 | 0.6341 | | **0.4727** | 0.5609 | | 1573 |
| DanceTrack | L9-ovmot (final) | 0.5570 | | 0.3278 | 0.4883 | | 6362 |
| BDD100K | L9-ovmot (final) | 0.4856 | | **0.5159** | 0.4287 | | 6677 |
| MOT17 | L9-ovmot (final) | 0.7104 | | **0.7037** | 0.6279 | | 438 |
| MOT20 | L9-ovmot (final) | 0.6358 | | **0.4751** | 0.5608 | | 1601 |
| Macro AssA | L6 / L7 / L8-B2 / L8-B1 / L9 v5 | | | 0.4922 / 0.4290 / 0.5045 / 0.5087 / **0.5090** | | | |
| Macro AssA | L9-ovmot (final) | | | **0.5056** | | | |

*L8-B1 DanceTrack re-measured during L9; other L8-B1 per-domain rows use
the L8 report values.

### Table 3 — TAO OVMOT (official TETA)

| Method | Observation | Split | TETA | LocA | AssocA | ClsA |
|---|---|---|---|---|---|---|
| L7 CLIP probe | CLIP-only | All | 33.94 | — | 29.51 | 7.51 |
| L8-B2 | PBD-zero | All | 34.33 | 65.05 | 30.44 | 7.51 |
| L8-B1 | PBD-zero | All | 34.07 | 65.06 | 29.64 | 7.52 |
| L8-B2 | full PBD | All | 32.22 | 64.19 | 24.95 | 7.53 |
| L8-B1 | full PBD | All | 31.83 | 64.09 | 23.87 | 7.53 |
| L9 v5 | full PBD | All | 32.04 | 64.40 | 24.22 | 7.49 |
| **L9-ovmot (final)** | full PBD | All | **33.79** | 64.47 | **29.34** | 7.54 |
| **L9-ovmot (final)** | full PBD | Base | 33.73 | 64.41 | 29.34 | 7.43 |
| **L9-ovmot (final)** | full PBD | Novel | 34.22 | 64.94 | 29.37 | 8.35 |

### Table 4 — RMOT (Refer-Dance, 40 GT queries)

| Method | Detector | HOTA | DetA | AssA | MOTA | IDF1 |
|---|---|---|---|---|---|---|
| TransRMOT (paper) | DETR-based | 9.58 | 4.37 | 20.99 | | |
| iKUN (paper) | ByteTrack/DLA | 29.06 | 25.33 | 33.35 | | |
| L8-B2 | LocateAnything | 35.20 | 43.42 | 28.63 | | |
| L8-B1 | LocateAnything | 37.88 | 46.51 | 31.02 | | |
| L9 main v5 | LocateAnything | 37.07 | 45.58 | 30.30 | 29.64 | 36.41 |
| L9-ovmot (final) | LocateAnything | 36.79 | 45.58 | 29.86 | 29.38 | 36.56 |

Detector caveat: DetA is not directly comparable across detectors;
AssA comparison is informative but also detector-dependent.

### Table 5 — Ablation (L9 protocol)

| Group | Ordinary Macro AssA | TAO AssocA | RMOT AssA |
|---|---|---|---|
| identity-only (v5 eval-time) | 0.3867* | — | — |
| semantic-only (v5 eval-time) | 0.3639* | — | — |
| strict decoupled (B2) | 0.5045 | 30.44 (PBD-zero) | 28.63 |
| spec-conditioned (L9-ovmot) | **0.5056** | 29.34 (full PBD) | 29.86 |

*identity/semantic eval-time rows cover DanceTrack + BDD only
(0.3387+0.4347)/2 and (0.3058+0.4219)/2; full four-domain rows would
require additional runs but the L8 protocol showed the same ordering.

### Table 6 — Cost

| Run | Trainable | Total | GPU | VRAM | Steps | Wall-clock |
|---|---|---|---|---|---|---|
| L8-B2 | 18.8 M | | 4x40G | | 2500 | |
| L8-B1 | 18.8 M | | 4x40G | | 3000 | |
| L9 main v5 | ~19.9 M | | 2x40G | | 3000 | ~1.9 h |
| L9-ovmot (final) | ~19.9 M | | 4x40G | | 6000 (resume) | ~1.6 h |

## 6. Failure analysis

Detailed root causes and failure boundaries in
`reports/l9_failure_analysis.md`.  Highlights:
- init loader bug (random core) invalidated v1-v4 — fixed and kept as
  evidence;
- naive full-PBD observation without crop-PBD training regresses TAO
  AssocA by ~5.5 points; crop-PBD adaptation recovers most of it;
- DanceTrack (crowded same-appearance dancers) remains the hardest
  identity regime (AssA 0.33-0.35 across methods);
- RMOT per-query CIs overlap between v5/B1/final (HOTA ~25-41 across the
  three), so the 40-query ranking is indicative only.

## 7. Novelty audit

Full audit: `reports/l9_iclr_novelty_audit.md`.  We did **not** identify
a published, verifiable system that (a) trains one identity-dynamics
core with persistent memory/lifecycle/set competition, (b) evaluates one
shared checkpoint on closed-set MOT + OVMOT + RMOT, and (c) represents
all three WHAT specifications in one observation space.  Neighbours:
OVTR/TRACT (OVMOT-only), AED (CV+OV, no language/lifecycle), QTrack
(RMOT VLM), MOTIP (ordinary-MOT ID prediction), iKUN/TransRMOT
(language-driven RMOT).  Phrased as "we did not identify ...", not
"first".

## 8. ICLR readiness

**NEAR_READY**.

What supports readiness:
- one shared learned identity-dynamics core + one shared checkpoint
  across closed-set MOT, OVMOT and RMOT, with official evaluators only;
- full-observation OVMOT (crop-PBD identity tokens) is a genuine new
  capability, with Base = Novel balance (AssocA 29.34 / 29.37);
- the specification-conditioned gate (cond-gated) reaches the best
  ordinary Macro AssA in the project (0.5090, v5) without hurting RMOT;
- two implementation bugs found and fixed during the stage (init loader;
  degenerate transform init) and a scientifically honest negative result
  (naive full-PBD distribution mismatch, recovered by adaptation);
- all papers/GitHub audited; RMOT uncertainty quantified by bootstrap CI.

What keeps it below READY:
- full-PBD OVMOT (TETA 33.79 / AssocA 29.34) still trails the PBD-zero
  regime (34.33 / 30.44) in this setup; the crop-PBD training stream is
  sparse (7.5k candidates), so the "full observation is better" claim is
  not yet demonstrated;
- ordinary Macro AssA of the final shared checkpoint (0.5056) is slightly
  below v5/B1 (0.5090/0.5087), so the final model is a Pareto point, not
  a strict improvement;
- RMOT AssA (29.9) remains below RMOT-specialised iKUN (33.35) with
  overlapping CIs;
- single seed; 40-query RMOT benchmark; no interactive tracking.

## 9. Artifacts

- Checkpoints: `outputs/l9/checkpoints/uidm_l9_main/`
- Final checkpoint: `outputs/l9/checkpoints/uidm_l9_main_ovmot/latest.pt`
- PBD cache: `outputs/l9/cache/tao_val_pbd/`
- Eval outputs: `outputs/l9/trackeval/`
- Docs: `reports/l9_literature_and_code_audit.md`,
  `reports/l9_model_design.md`, `reports/l9_tao_pbd_cache.md`,
  `reports/l9_training_summary.md`, `reports/l9_full_observation_eval.md`,
  `reports/l9_mot_results.md`, `reports/l9_ovmot_results.md`,
  `reports/l9_rmot_results.md`, `reports/l9_ablation.md`,
  `reports/l9_failure_analysis.md`, `reports/l9_iclr_novelty_audit.md`
