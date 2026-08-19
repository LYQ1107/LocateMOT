# Stage L10 — Training Summary

Date: 2026-08-18 (final)

## Configuration

- Model: `large` UIDM (d_model 384, 4 layers, 8 heads), `cond_gated`,
  unified mode; trainable ~19.9M params.
- Init/resume: `outputs/l9/checkpoints/uidm_l9_main_ovmot/latest.pt`
  (L9 final shared checkpoint, 6,000 steps).
- Data: task-balanced MOT : OVMOT : RMOT = 0.4 : 0.3 : 0.3;
  OVMOT stream = L10 expanded TAO-train (500 videos / 18,274 frames /
  322,843 candidates, C-TAO GT); RMOT = Refer-Dance.
- Optimizer: AdamW (uidm lr_core 5e-5, adapter lr 1.2e-4), OneCycle
  cosine, weight decay 1e-4; PBD dropout 0.15; grad clip 5.0.
- Run: **15,000 steps, batch 8/GPU (effective 32), 4 GPUs (1-4)**,
  ~3.07 s/step, wall-clock 27,014 s (~7.5 h); LR adapter 2.4e-4 /
  core 1e-4 (linear scaling for 2x effective batch), OneCycle cosine;
  save every 1,000.

## Loss curve (end)

Final epoch averages: loss_row 1.49, loss_col 0.67, loss_nm 0.99,
loss_new 0.52, loss_motion 0.48, loss_switch 0.13, loss_relevance 0.30;
LR annealed to ~0.  The curve was not clearly improving over the last
~3k steps, so the run was not extended beyond 15k.

## Checkpoint

`outputs/l10/checkpoints/uidm_l10_main/latest.pt` (step 15,000, epoch 32)
is the single shared checkpoint used for all L10 evals.

## Cost

~7.5 h x 4 x 40 GB A100 GPUs; feature caches: DLA ~44 MB, CLIP/pkls
~4 GB, TAO train PBD 5.5 GB, TAO val PBD 27 GB (L9).
