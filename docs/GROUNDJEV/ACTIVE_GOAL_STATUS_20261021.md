# GroundJEV active status — 2026-10-21

Generated 2026-10-21T02:22:04+08:00 from the live owner states and the formal G2 event log.

- **G2 Clean:** event step `11735/18227`; latest durable checkpoint `step11500`; two ranks on physical GPUs 3/5; recent 500-step rate `11.617` s/step; linear remainder estimate `20.9` hours. Final receipt is not present.
- **G0 MOT20-05:** the registered 1,657-frame validation-half export is complete. Its fresh native-source control reports HOTA `0.073347`, IDF1 `0.059598`, IDSW `1933`; this is a control scope, not a full MOT20 TEST claim.
- **G1 GPU5:** `WAITING_G2_CLEAN_SUCCESS`; it has not started real optimizer updates.
- **G2 deployment bundle:** `WAITING_G2_AND_G1_TERMINAL`; no bundle has been exported.
- **Independent held-out:** `WAITING_G2_CLEAN_BUNDLE_AND_GPU7` for MOT20-05, RMOT single/multi, and TAO OVMOT; no official held-out metrics yet.
- **Old independent OneJev four-GPU resume:** explicitly paused before GPU launch at durable step `3000`. The pause marker and code guard prevent accidental auto-start; historical checkpoint/config/logs remain intact.
- **MOT-only long training:** no live independent MOT-only optimizer process was found.
- **GroundJEV coupled Phase A/B:** engineering and bounded learnability receipts are preserved in the isolated `research/groundjev-coupled-single-backbone-20261021` worktree; they do not constitute full MOT/RMOT/OVMOT performance claims.

The overall research goal remains incomplete until G2 terminates, G1/bundle/held-out receipts are terminal, and independent metrics are produced under the registered protocols.
