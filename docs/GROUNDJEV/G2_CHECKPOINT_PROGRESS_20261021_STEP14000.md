# GroundJEV G2 Clean checkpoint audit — step 14000

- The protected exact-once two-rank run wrote the durable step **14,000 / 18,227** adapter checkpoint on physical GPUs 3/5. The event stream had already advanced to step 14,021 at the audit.
- Adapter checkpoint: `checkpoints/adapter-step014000.pt`, 469,243,990 bytes, SHA-256 `0cbf160b57a4feccb11a77f602cd8ebd39e79d93d79453cb78316a1d3ef6371e`.
- Recovery state: `checkpoints/latest_training_state.pt`, 469,247,597 bytes, SHA-256 `d7bbdfdecfeba789ff022d31800e1cd7d1fedb465e34e325ed738f33f66bae78`.
- Both files loaded successfully on CPU with explicit trusted-checkpoint `torch.load(..., weights_only=False)`. Both report step 14,000, format `groundanything-g2-projector-qwen3-lora-v2-strict`, world size 2, exact-once coverage, effective global batch 32, seed 20261019, optimizer/scheduler/RNG state, and the registered Clean TRAIN manifest SHA-256 `f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71`.
- Latest event at the audit: loss `0.0572921158`, gradient norm `6.24900246`, peak event memory `12.828891 GiB`; all finite.
- The parent process remained alive; `final_receipt.json` and `adapter_projector_final.pt` are not yet present. Downstream G1, native TAO/OVMOT, three-task OneJev and held-out evaluation remain gated.
- This is checkpoint/recovery and training-health evidence only; it is not a MOT/RMOT/OVMOT performance claim.

Machine-readable record: [G2_CHECKPOINT_PROGRESS_20261021_STEP14000.json](./G2_CHECKPOINT_PROGRESS_20261021_STEP14000.json).
