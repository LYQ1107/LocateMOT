# STAGE L9 — Scaled Specification-Conditioned Unified MOT

Status: **IN PROGRESS** (results being collected; final numbers to be
filled from official evaluators)

Date: 2026-08-15
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
  (v1 regressed due to an init bug and was kept as a failed-run evidence;
  v2 rerun with the corrected identity init is in progress).
  Stage B (planned): resume with OVMOT stream.

### 3.4 Negative evidence and fix (L9 v1)

The first L9 main run (10k steps, cond-gated, init L8-B1) regressed
ordinary MOT (DanceTrack AssA 0.3457 -> 0.1135, Macro AssA 0.5087 ->
~0.42) and RMOT (AssA 31.02 -> 10.58), while an L8-B1 control eval was
healthy (Dance AssA 0.3405).  Root cause: `sem_transform` was initialized
as an all-ones matrix (degenerate rank-1 projection) instead of the
identity matrix.  The v1 checkpoint is preserved under
`outputs/l9/checkpoints/uidm_l9_main_v1_failed/`; the corrected v2 run
uses `eye_` initialization.

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
| DanceTrack | L8-B1 | | | | | | |
| DanceTrack | L9 | | | | | | |
| BDD100K | L8-B2 | | | 0.5019 | | | |
| ... | ... | | | | | | |
| Macro AssA | L6 / L7 / L8-B2 / L8-B1 / L9 | | | 0.4922 / 0.4290 / 0.5045 / 0.5087 / TBD | | | |

### Table 3 — TAO OVMOT (official TETA)

| Method | Observation | Split | TETA | LocA | AssocA | ClsA |
|---|---|---|---|---|---|---|
| L7 CLIP probe | CLIP-only | All | 33.94 | — | 29.51 | 7.51 |
| L8-B2 | PBD-zero | All | 34.33 | 65.05 | 30.44 | 7.51 |
| L8-B1 | PBD-zero | All | 34.07 | 65.06 | 29.64 | 7.52 |
| L8-B2 | full PBD | Base / Novel / All | TBD | TBD | TBD | TBD |
| L8-B1 | full PBD | Base / Novel / All | TBD | TBD | TBD | TBD |
| L9 main | full PBD | Base / Novel / All | TBD | TBD | TBD | TBD |

### Table 4 — RMOT (Refer-Dance, 40 GT queries)

| Method | Detector | HOTA | DetA | AssA | MOTA | IDF1 |
|---|---|---|---|---|---|---|
| TransRMOT (paper) | DETR-based | 9.58 | 4.37 | 20.99 | | |
| iKUN (paper) | ByteTrack/DLA | 29.06 | 25.33 | 33.35 | | |
| L8-B2 | LocateAnything | 35.20 | 43.42 | 28.63 | | |
| L8-B1 | LocateAnything | 37.88 | 46.51 | 31.02 | | |
| L9 main | LocateAnything | TBD | TBD | TBD | | |

Detector caveat: DetA is not directly comparable across detectors;
AssA comparison is informative but also detector-dependent.

### Table 5 — Ablation (L9 protocol)

| Group | Ordinary Macro AssA | TAO AssocA | RMOT AssA |
|---|---|---|---|
| identity-only | TBD | TBD | TBD |
| semantic-only | TBD | TBD | TBD |
| strict decoupled (B2) | TBD | TBD | TBD |
| spec-conditioned (L9 main) | TBD | TBD | TBD |

### Table 6 — Cost

| Run | Trainable | Total | GPU | VRAM | Steps | Wall-clock |
|---|---|---|---|---|---|---|
| L8-B2 | 18.8 M | | 4x40G | | 2500 | |
| L8-B1 | 18.8 M | | 4x40G | | 3000 | |
| L9 main | ~19.9 M | | 2x40G | | 10000 | TBD |

## 6. Failure analysis

To be completed: worst ordinary-MOT sequences (high IDSW), worst RMOT
queries (low AssA), TAO novel failures; classify same-class crowd, motion
crossing, language ambiguity, detector miss, semantic FP, long occlusion,
NEW/NO-MATCH and reactivation errors.

## 7. Novelty audit

To be completed after results; expected statement based on
`reports/l9_literature_and_code_audit.md`: we did not identify a
published system that uses one trained identity-dynamics core + one
shared checkpoint across closed-set MOT, OVMOT and RMOT.

## 8. ICLR readiness

**TBD** (one of ICLR_READY / NEAR_READY / NOT_READY), based on novelty,
method depth, unification evidence, benchmark breadth, competitiveness,
ablation quality, protocol fairness and reproducibility.

## 9. Artifacts

- Checkpoints: `outputs/l9/checkpoints/uidm_l9_main/`
- PBD cache: `outputs/l9/cache/tao_val_pbd/`
- Eval outputs: `outputs/l9/trackeval/`
- Docs: `reports/l9_literature_and_code_audit.md`,
  `reports/l9_model_design.md`, `reports/l9_tao_pbd_cache.md`,
  `reports/l9_training_summary.md`, `reports/l9_full_observation_eval.md`,
  `reports/l9_mot_results.md`, `reports/l9_ovmot_results.md`,
  `reports/l9_rmot_results.md`, `reports/l9_ablation.md`,
  `reports/l9_failure_analysis.md`, `reports/l9_iclr_novelty_audit.md`
