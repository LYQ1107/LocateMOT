# GroundJEV status snapshot — G2 step 10862

- Observed event: `2026-10-20 23:28:21 +0800`; optimizer step `10862/18227`.
- Loss `0.199614327`, gradient norm `6.536875725`, peak memory `12.828872 GiB`; all finite.
- Final receipt: `False`. Latest durable checkpoint remains step `10500`.
- Scope: GroundJEV/TrackJEV only; external TTA/shift-RMOT processes excluded.

G0 MOT20-05 control has `1657` frames, HOTA `0.073347`, DetA `0.090270`, AssA `0.059674`, IDF1 `0.059598`, IDSW `1933`. It is not a G2 held-out claim.

## Downstream queues

| Queue | State |
|---|---|
| g1_gpu5_adaptation | `WAITING_G2_CLEAN_SUCCESS` |\n| g2_bundle_export | `WAITING_G2_AND_G1_TERMINAL` |\n| independent_heldout | `WAITING_G2_CLEAN_BUNDLE_AND_GPU7` |\n| onejev_joint | `WAITING_G2_G1_AND_EXCLUSIVE_GPU_SET` |\n
Independent held-out preflight and TrackJEV plumbing preflight remain passed; no final independent metrics are promoted. Overall goal remains incomplete.
