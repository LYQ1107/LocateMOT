# GroundJEV G2 Clean checkpoint audit — step 13500

- The protected two-rank exact-once run reached step **13,500 / 18,227** on physical GPUs 3/5.
- Durable adapter checkpoint: `checkpoints/adapter-step013500.pt`, 469,243,990 bytes, SHA-256 `84a7e84a391d375cfeb3c74dea59bdb36f9a608fdf523132f0c6c9f6e1273c70`.
- Durable recovery state: `checkpoints/latest_training_state.pt`, 469,247,597 bytes, SHA-256 `3a3a9a51d6f1e6b8f3e4a7d1ad399bd4864805c50dd0eab2eeb19c6def85626c`.
- Both files loaded successfully on CPU with explicit trusted-checkpoint setting `torch.load(..., weights_only=False)`. The first ad-hoc validation attempt used PyTorch 2.6's default `weights_only=True` and failed before reading the state; this was a validation-tool issue, not a checkpoint or training failure.
- Both states report step 13,500, format `groundanything-g2-projector-qwen3-lora-v2-strict`, world size 2, exact-once coverage, and the registered Clean TRAIN manifest SHA-256 `f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71`.
- The parent process remained alive; `final_receipt.json` and `adapter_projector_final.pt` are not yet present. Downstream G1, native TAO/OVMOT, three-task OneJev and held-out evaluation remain gated.
- This is checkpoint/recovery evidence only and makes no MOT/RMOT/OVMOT performance claim.

Machine-readable record: [G2_CHECKPOINT_PROGRESS_20261021_STEP13500.json](./G2_CHECKPOINT_PROGRESS_20261021_STEP13500.json).
