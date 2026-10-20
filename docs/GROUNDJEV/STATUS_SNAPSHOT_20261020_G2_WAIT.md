# GroundJEV status snapshot — G2 wait

- Observed: `2026-10-20T15:27:19.900919+00:00`
- Scope: GroundJEV/TrackJEV only. External TTA/shift-RMOT processes are excluded.
- Project root: `/data2/user/LocateMOT`
- G2 Clean: `G2_FORMAL_TRAINING_RUNNING`, target `18227` optimizer steps; final receipt is absent. Durable checkpoint is step `10500`.
- G0 MOT20-05 control: `1657` frames, HOTA `0.073347`, DetA `0.090270`, AssA `0.059674`, IDF1 `0.059598`, IDSW `1933`, end-to-end FPS `0.061370`. This is a G0 control, not a G2 held-out claim.

## Downstream queue states

| Queue | State |
|---|---|
| g1_gpu5_adaptation | `WAITING_G2_CLEAN_SUCCESS` |\n| g2_bundle_export | `WAITING_G2_AND_G1_TERMINAL` |\n| independent_heldout | `WAITING_G2_CLEAN_BUNDLE_AND_GPU7` |\n| onejev_joint | `WAITING_G2_G1_AND_EXCLUSIVE_GPU_SET` |\n
Held-out manifest preflight passed; no independent held-out metrics are promoted yet. TrackJEV runtime preflight passed, but it is plumbing evidence only. Overall goal remains incomplete.
