# YesTrack native v2 real training started

After the permission repair and GPU7 resource wait, the preserved retry owner
entered `REAL_YESTRACK_TRAINING_RUNNING` at `2026-10-21 19:07:28 +0800`.

- Physical GPU: 7
- Base checkpoint: `onejev_unified600_native_v2/ckpt-step600`
- Training rows: 726 native Yes/No rows, SHA-256
  `065337d2fedca2377d02b94e57c9eb3778aeca09aeb1a8b9463429fa15bbee66`
- Validation rows: 390, SHA-256
  `d63a9fb0d54b8a1dc5d12b0960a48cb75c56ab786671f955fb317e196b3033d8`
- Planned schedule: 273 optimizer steps / 2178 examples / 3 epochs
- Observed step 10: loss `1.3658664280`, slot accuracy `0.725`, peak memory
  `7.265 GiB`

The process loaded the real model and LoRA configuration and is writing the
training log under `/data3/user/LocateMOT-checkpoints/yestrack_native_v2_lora`.
No YesTrack metric is claimed until the terminal adapter, frozen-base
evaluation, fine-tuned evaluation and receipt are all present.

The protected G2 Clean run remains independent and active at step 16991/18227;
the formal MOT/RMOT/OVMOT OneJev joint optimizer remains at zero updates behind
its native OVMOT/G2 gates.
