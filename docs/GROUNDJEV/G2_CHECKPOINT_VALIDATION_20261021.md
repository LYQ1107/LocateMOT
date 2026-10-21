# G2 durable checkpoint validation — step 12000

Generated `2026-10-21T03:18:41.435695+08:00` from the protected G2 Clean run directory. Both the full recovery state and periodic adapter artifact load successfully with the registered local trainer under a trusted local read.

- format: `groundanything-g2-projector-qwen3-lora-v2-strict`
- step: `12000`
- manifest SHA: `f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71`
- trainable tensors: `293`; signature entries: `293`
- optimizer state: present; scheduler state: present; RNG state: present; rank RNG paths: `['0', '1']`
- sampler position: `96000`; world size: `2`
- terminal G2 receipt: absent; this is a recovery/checkpoint validation, not a final performance result.

PyTorch 2.6's default `weights_only=True` reader rejects the checkpoint's recorded NumPy RNG object. The audit therefore used `weights_only=False` only for these trusted local artifacts, matching the trainer's own restore path; no checkpoint was modified.
