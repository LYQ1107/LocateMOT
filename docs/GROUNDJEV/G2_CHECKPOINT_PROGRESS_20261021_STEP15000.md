# GroundJEV G2 Clean checkpoint — step 15000

- CPU recovery load succeeded for `adapter-step015000.pt` and `latest_training_state.pt` with explicit `weights_only=False`.
- Adapter SHA-256: `f1f71140ec3367e099307ffbde40805ce41a7d1b41f33c8613d58aff7e49720c`; full training-state SHA-256: `49497d6cade73067ac03fabbf581a436a808bad45bcc1e88bd75e369d00a3aea`. Both report step `15000`, world size `2`, exact-once coverage, the registered Clean manifest SHA, and optimizer/scheduler/RNG state.
- Latest live event is step `15002` at `2026-10-21 12:47:01 +0800`, with loss `0.125136467512`, gradient norm `10.066356658936`, and peak event memory `12.828786 GiB`.
- `final_receipt.json` is absent; downstream owners remain gated and report no coupled optimizer update.

This is checkpoint/recovery and training-health evidence only. It does not claim MOT/RMOT/OVMOT performance or a coupled JEV result.
