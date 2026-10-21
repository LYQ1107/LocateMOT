# GroundJEV G2 Clean checkpoint — step 14500

- CPU recovery load succeeded for both the adapter and full training state with explicit `weights_only=False`.
- Adapter SHA-256: `2f79894019ba565849b4adb34cc209f0959fb494d7ceeb2b42d7c33da2f26fde`; full training-state SHA-256: `b98fc2721c49e8c2e359b51a4d3367e25475edd0430eb71df833ca27a550eea6`. Both report step `14500`, world size `2`, exact-once coverage, registered Clean manifest SHA, and optimizer/scheduler/RNG state.
- Latest live event is step `14512` at `2026-10-21 11:14:01 +0800`, with loss `0.225528766343`, gradient norm `9.290996551514`, and peak event memory `12.828843 GiB`.
- `final_receipt.json` is still absent; downstream owners remain gated and report no coupled optimizer update.

This is checkpoint/recovery and training-health evidence only. It does not claim MOT/RMOT/OVMOT performance or a coupled JEV result.
