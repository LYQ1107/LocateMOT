# LocateMOT — Stage L9 GPT Handoff (living draft)

Date: 2026-08-15 (updated as the stage progresses)

## In one sentence

Stage L9 adds full-observation OVMOT (crop-based PBD identity tokens for
every TAO val candidate) and a specification-conditioned identity gate to
the L8 shared UIDM; the L9 main checkpoint (v5) reaches ordinary Macro
AssA 0.5090 (vs L8-B1 0.5087) and RMOT HOTA 37.07 / AssA 30.30, with the
full-PBD TAO evaluation pending the val cache.

## Where things live

- Root: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
- Conda: `/home/lwr/anaconda3/envs/locatemot`
- Final report (in progress): `reports/STAGE_L9_SCALED_UNIFIED_MOT_FINAL_REPORT.md`
- This handoff: `reports/LATEST_GPT_HANDOFF.md`

## State

- **Training**: L9 main v5 complete (`outputs/l9/checkpoints/uidm_l9_main/`,
  3000 steps from `uidm_l8_joint`, cond-gated, `eye_` semantic transform,
  corrected `load_l8_state` init).  v1-v4 were invalidated by an init
  loader bug (random core) and kept as failure evidence.
- **TAO val PBD cache**: running with 4 workers (GPUs 1-4), write-through
  + resume; ~6.9k/36.4k frames at last check; auto-scaled/paused by
  `tools/monitor_l9.py`; regression-checked
  (`tools/check_l9_pbd_cache.py`).
- **Evals done (official)**:
  - Ordinary: Macro AssA 0.5090 (Dance 0.3509 / BDD 0.5108 / MOT17
    0.7017 / MOT20 0.4727).
  - RMOT: HOTA 37.07 / DetA 45.58 / AssA 30.30 / MOTA 29.64 / IDF1 36.41
    (threshold 0.45 calibrated on Refer-Dance train, F1 0.8905).
  - OVMOT full-PBD: pending cache.
- **Next**: full-PBD TETA on L8-B2/L8-B1/L9-v5; eval-time ablation
  (identity/semantic); then TAO train subset (105 videos: DLA dets +
  CLIP + crop PBD) and a resumed OVMOT-joint run; final report +
  ICLR-readiness verdict.

## Key files

- Model: `locatemot/models/l8_unified.py` (`cond_gated`)
- Cache: `tools/cache_l9_tao_pbd.py`, `tools/check_l9_pbd_cache.py`,
  `tools/monitor_l9.py`
- Train: `tools/train_l9_uidm.py`
- Eval: `tools/eval_l8_{ordinary,ovmot,rmot}.py`,
  `tools/eval_l9_three_tasks.py`, `tools/eval_l9_ovmot_full.py`
- Data prep: `tools/generate_l9_tao_train_dets_subset.py`,
  `tools/prepare_l9_tao_train.py`, `configs/l9/tao_train_videos.json`
- Docs: `reports/l9_*.md`, `docs/future_rl_reference.md`

## Known caveats

- RMOT baselines use different detectors (LocateAnything vs
  ByteTrack/DLA); DetA is not comparable.
- 40-query RMOT evaluation has wide confidence intervals.
- The val cache rate is ~21 frames/min with 4 workers; full TAO val is
  expected within ~1 day (write-through, resumable).
- Another user's SAM3_InterMOT jobs intermittently consume host RAM;
  LocateMOT never exceeds 4 physical GPUs and auto-pauses cache workers
  under memory pressure.

