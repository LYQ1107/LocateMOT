# Stage L9 — Training Summary (UIDM-L9 main)

Status: in progress (2026-08-15, to be finalised after the full run)

## Run 1 — L9 main (MOT + RMOT, cond-gated)

- Script: `tools/train_l9_uidm.py`
- Checkpoint: `outputs/l9/checkpoints/uidm_l9_main/`
- Model: L8 `large` UIDM core (d_model=384, 6 layers) + Unified
  Observation Adapter + cond-gated residual.
- Trainable params: ~19.9 M.
- Init: L8-B1 `outputs/l8/checkpoints/uidm_l8_final/latest.pt`.
- GPU: 2 x 40 GB (physical 4,6; DDP, batch 4/GPU).
- Steps: target 10,000; lr 1.2e-4 (adapter) / 5e-5 (core), OneCycle,
  pct_start 0.05.
- Data: ordinary (BDD 200 vids / DanceTrack 32+cal / MOT17 / MOT20 CLIP
  caches) + RMOT (Refer-Dance 142 clips); task ratio RMOT 0.35;
  PBD-dropout 0.15; seed 20260806.
- Save: every 1000 steps (model + optimizer + scheduler + step/epoch);
  resume verified.

Observed convergence (preliminary): total loss ~100 at step 10 -> ~10-30
by step 1500; row/col losses decreasing.  Final metrics to be added after
the run completes.

## Run 2 — L9 main + OVMOT (planned)

- After `outputs/l9/data/tao_train` (105 TAO videos: DLA dets + CLIP +
  crop PBD) is ready, resume Run 1 with `--p-ovmot 0.3` for additional
  steps so the shared core sees crop-based PBD identity evidence from
  open-vocabulary candidates.
- Will be reported with its own commit/steps/loss curve.

