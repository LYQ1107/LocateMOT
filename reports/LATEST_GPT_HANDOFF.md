# LocateMOT — Stage L9 GPT Handoff (final)

Date: 2026-08-17

## In one sentence

Stage L9 adds full-observation OVMOT (crop-based PBD identity tokens for
every TAO val candidate), a specification-conditioned identity gate, and
a crop-PBD-adapted OVMOT training stream to the L8 shared UIDM.  The
final shared checkpoint (L9-ovmot) reaches ordinary Macro AssA 0.5056,
RMOT HOTA 36.79 / AssA 29.86, and full-PBD TAO TETA 33.79 / AssocA 29.34
with Base = Novel.

## Where things live

- Root: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
- Conda: `/home/lwr/anaconda3/envs/locatemot`
- Final report: `reports/STAGE_L9_SCALED_UNIFIED_MOT_FINAL_REPORT.md`
- This handoff: `reports/LATEST_GPT_HANDOFF.md`

## Final state

- **Final checkpoint**: `outputs/l9/checkpoints/uidm_l9_main_ovmot/latest.pt`
  (L9-v5 + crop-PBD OVMOT resume, 6,000 steps, 4 GPUs).
- **TAO val PBD cache**: COMPLETE (36,375 frames, verified).
- **Evals (official)**:
  - Ordinary: Macro AssA 0.5056 (Dance 0.3278 / BDD 0.5159 / MOT17
    0.7037 / MOT20 0.4751).  Best-project ordinary model remains L9-v5
    (0.5090).
  - RMOT: HOTA 36.79 / DetA 45.58 / AssA 29.86 / MOTA 29.38 / IDF1 36.56;
    bootstrap CI HOTA [27.9, 40.6], AssA [22.7, 37.0] (36 queries).
  - OVMOT full-PBD: TETA 33.79 / LocA 64.47 / AssocA 29.34 / ClsA 7.54;
    Base 29.34, Novel 29.37.
- **Key findings**:
  1. Naive full-PBD on PBD-zero-trained checkpoints regresses TAO
     (AssocA 30.44 -> 24.95); crop-PBD adaptation recovers to 29.34.
  2. Full-PBD still trails PBD-zero by ~1.1 AssocA / 0.5 TETA — sparse
     OVMOT training stream (7.5k candidates) is the likely limit.
  3. v1-v4 were invalidated by an init-loader bug (random core); the
     corrected v5 reaches Macro AssA 0.5090.
  4. No published system unifies one identity core + one checkpoint
     across closed-set MOT + OVMOT + RMOT (novelty: "we did not
     identify", not "first").
- **ICLR readiness**: NEAR_READY (evidence solid; headline gain modest;
  full-PBD not yet beating PBD-zero; single seed; 40-query RMOT).

## Key files

- Model: `locatemot/models/l8_unified.py` (`cond_gated`)
- Cache: `tools/cache_l9_tao_pbd.py`, `tools/check_l9_pbd_cache.py`,
  `tools/monitor_l9.py`
- Train: `tools/train_l9_uidm.py`
- Eval: `tools/eval_l8_{ordinary,ovmot,rmot}.py`,
  `tools/eval_l9_three_tasks.py`, `tools/eval_l9_ovmot_full.py`,
  `tools/finalize_l9_evals.py`, `tools/bootstrap_rmot_ci.py`
- OVMOT train stream: `tools/build_l9_tao_train_from_l6.py`,
  `tools/merge_l9_train_pbd.py`, `configs/l9/tao_train_videos.json`
- Docs: `reports/l9_*.md`, `docs/future_rl_reference.md`

## Known caveats

- RMOT baselines use different detectors (LocateAnything vs
  ByteTrack/DLA); DetA is not comparable.
- 40-query RMOT evaluation has wide confidence intervals.
- Full-PBD OVMOT does not yet beat the PBD-zero regime; more OVMOT
  training data (e.g., full TAO train with working DLA dets) is the
  clearest next step.
- DLA dets on TAO train are blocked by a torchvision roi_align OOM in
  this environment (documented; the L6-based stream was used instead).

