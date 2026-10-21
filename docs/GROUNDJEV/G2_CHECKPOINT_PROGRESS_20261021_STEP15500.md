# GroundJEV G2 Clean checkpoint — step 15500

- CPU recovery load succeeded for `adapter-step015500.pt` and `latest_training_state.pt` with explicit `weights_only=False`.
- Adapter SHA-256: `54bf3472d9e7d811dd5f08d445a3f1352a430effb4af7b1df73beeb553feb326`; full training-state SHA-256: `7ef2bf4cb0a1494b2c0c07debad206a49df908fcea32b2f40fad782f1fcc7191`. Both report step `15500`, world size `2`, exact-once coverage, the registered Clean manifest SHA, and optimizer/scheduler/RNG state.
- Latest live event is step `15737` at `2026-10-21 15:07:17 +0800`, with loss `0.216046137153171`, gradient norm `11.128178596496582`, and peak event memory `12.828757 GiB`.
- `final_receipt.json` is absent; downstream owners remain gated and report no coupled optimizer update.

This is checkpoint/recovery and training-health evidence only. It does not claim MOT/RMOT/OVMOT performance or a coupled JEV result.
