# GroundJEV G2 Clean checkpoint — step 16000

- CPU recovery load succeeded for `adapter-step016000.pt` and `latest_training_state.pt` with explicit `weights_only=False`.
- Adapter SHA-256: `f473d5dc0d108576264f944cca26a3cbc64834774025646695183ae13bc97ed5`; full training-state SHA-256: `560c314569d8e087cd7d53443ea2d886d56ee77381bf3c2719c874f88ac28e71`. Both report step `16000`, world size `2`, exact-once coverage, the registered Clean manifest SHA, and optimizer/scheduler/RNG state.
- Latest live event is step `16014` at `2026-10-21 16:00:52 +0800`, with loss `0.14327701507136226`, gradient norm `10.117892265319824`, and peak event memory `12.828872 GiB`.
- `final_receipt.json` is absent; downstream owners remain gated and report no coupled optimizer update.

This is checkpoint/recovery and training-health evidence only. It does not claim MOT/RMOT/OVMOT performance or a coupled JEV result.
