# G2 Clean recovery-checkpoint audit

- Checked: `2026-10-21T06:27:15.356555+08:00` on CPU with `CUDA_VISIBLE_DEVICES` empty.
- Checkpoint: `latest_training_state.pt`, step `13000`, world size `2`, `469247597` bytes.
- Durable adapter checkpoint: `adapter-step012500.pt`, `469243990` bytes.
- Required recovery components present: model trainable state, optimizer state, scheduler state, sampler and position, Python/NumPy/Torch/CUDA RNG state, training contract and trainable audit.
- Manifest SHA recorded in checkpoint: `f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71`.
- Missing required components: `none`.

This validates recoverability of the latest durable checkpoint while G2 continues. It is not a benchmark or training-quality result.
