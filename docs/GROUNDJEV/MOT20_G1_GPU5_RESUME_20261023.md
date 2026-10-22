# G1 MOT20-05 GPU5 causal resume

The registered G1 control on GPU4 was interrupted only after an immutable 18-frame prefix (physical frames 1659–1676) had been written. It had no `result.json` or official metric receipt, so it is not a result claim.

- Original owner: `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/heldout/g1_mot20_05_gpu4_v1`
- Original terminal status: `G1_MOT20_05_PARTIAL_SUPERSEDED_FOR_GPU5_RESUME_FAILED`
- Resumed owner: `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/heldout/g1_mot20_05_gpu5_resume_v1`
- Physical GPU: 5; service port: 8240
- Resume source: the original `G1_GPU5_V2/mot20_05_validation1657` output
- Legal scope: MOT20-05 validation half, 1,657 registered frames, `training_on_this_scope=false`
- Manifest SHA-256: `f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71`
- G1 model SHA-256: `0efd9d7ddb241d588be85cb6b08853de5dfd18011aa477932c0580bab3265a25`

The resumed command uses the unchanged official parser, `tracking_dense` decoder, Simple tracker, and official MOT scorer. `run_video.py --resume-from` restores the exact completed prefix and continues the suffix in a fresh directory; no partial frame is scored until the resumed owner writes all 1,657 rows and a terminal official receipt. Resumed timing is explicitly marked as incomplete for end-to-end FPS because the original interruption interval is not measurable.
