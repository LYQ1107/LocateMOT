# Stage L10 — Training Speed Audit

Date: 2026-08-18 (filled from 200-step profiles)

## Protocol

- Same trainer as L9 (`tools/train_l9_uidm.py`, model=large, cond_gated),
  resumed from `outputs/l9/checkpoints/uidm_l9_main_ovmot/latest.pt`,
  with the expanded L10 OVMOT stream.
- 4 GPUs DDP, 120-step profile at start; per-step data/forward/backward/
  optimizer timings plus VRAM allocated/reserved.
- Before/after rows report micro-batch per GPU, effective batch, GPU util,
  samples/sec, steps/sec, time per 1k steps.

## Before (L9 config baseline)

From the L9 training log (`outputs/l9/logs/uidm_l9_main_ovmot_train.log`):
6,000 steps in 5,719 s (~0.95 s/step, 4 GPUs, batch 4, effective 16
clips/step).  L9 OVMOT clips had ~1.8 candidates/frame, so this is an
upper bound on speed for the L10 stream (which has ~50 candidates/frame).

## Profiling results (200 steps each, 4 GPUs, resume from L9 final)

| config | per-step (wall) | clips/s | VRAM reserved/GPU | decision |
| --- | ---: | ---: | ---: | --- |
| batch 4/GPU (eff. 16) | ~2.03 s | 7.9 | 5.1 GB | baseline |
| batch 8/GPU (eff. 32) | ~3.2 s | 10.0 | 11.6 GB | **chosen** |
| batch 16/GPU (eff. 64) | ~10+ s | <6.4 | 23.1 GB | rejected (superlinear blowup) |

Data loading was negligible (0.001-0.005 s/step); forward/rollout
dominates.  Batch 8 gives +27% clip throughput; batch 16 blows up
superlinearly in the set-competition forward and is not used.

## Chosen L10 configuration

- micro-batch 8/GPU, effective batch 32 (was 16), max-steps 15,000
  (= same 480k clip updates as 30k steps at batch 4), LR scaled 2x
  (adapter 2.4e-4, core 1e-4); single batch/LR adjustment, recorded.
- GPUs 1,2,3,4 (0 was taken by another user's OVTR job; DDP all-reduce
  made every rank ~5-10x slower, so the run moved off GPU 0).

## Changes applied

1. `--ovmot-dir`, `--rmot-dir` selectors in the trainer (L9 path remains
   the default).
2. `--workers`, `--prefetch`, `--no-pin-memory` dataloader options.
3. `--profile-steps` / `--profile-out` timing instrumentation.
4. Candidate/GT pipeline: DLA dets (top-50, score>=0.05) instead of
   LocateAnything boxes; C-TAO continuous GT instead of sparse TAO GT.

Speedup vs the L9 config at the same total clip budget: ~27% higher
clips/s; 15k steps instead of 30k for the same number of samples.
