# GroundJEV G2 Clean checkpoint — step 16500

- CPU recovery load succeeded for `adapter-step016500.pt` and `latest_training_state.pt` with explicit `weights_only=False`.
- Adapter SHA-256: `29724ad8f16eca5defe5d918cdbd3bb41d7680cf1736277d9a0a1b88f3a73ca4`; full training-state SHA-256: `6dd00a22fa768d9d8ad80060d2da6aba54ee1f55958efb737c05515e45fc659e`. Both report step `16500`, world size `2`, exact-once coverage, the registered Clean manifest SHA, and optimizer/scheduler/RNG state.
- Latest live event is step `16501` at `2026-10-21 17:35:04 +0800`, with loss `0.1464215029263869`, gradient norm `9.189213752746582`, and peak event memory `12.828853 GiB`.
- `final_receipt.json` is absent; downstream owners remain gated and report no coupled optimizer update.

This is checkpoint/recovery and training-health evidence only. It does not claim MOT/RMOT/OVMOT performance or a coupled JEV result.
