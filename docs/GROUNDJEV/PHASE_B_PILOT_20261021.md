# GroundJEV Phase B choice + Yes/No pilot — 2026-10-21

This bounded pilot used the G2 base checkpoint with model SHA `3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2` and 32 MOT17-02-FRCNN TRAIN frames. It reused 992 proposal boxes from a prior native GroundAnything candidate export, then pooled them with the G2 visual module once per frame (`g2_forward_count=32`). The 24-frame prefix was used for 192 readout optimizer steps; the final 8 frames are a temporal development holdout from the same video.

| Measure | Before | After |
|---|---:|---:|
| Train identity accuracy | 0.0618 | 0.6720 |
| Temporal-dev identity accuracy | 0.0363 | 0.6492 |
| Train admit accuracy | 0.1022 | 0.9247 |
| Temporal-dev admit accuracy | 0.1331 | 0.9153 |

The saved readout checkpoint is `adapter-step000192.pt` (SHA256 `c96f27c9c8de5a01a36036c78c4e4a36c467bb96a0ca5f7d0eee27d93e32c904`). The real replay used the trained readout without teacher memory on the eight development frames, made one-to-one assignments and retained three predicted memory tracks. The replay is recorded in the JSON receipt. It is a causal engineering replay only: there is no HOTA/IDF1 claim, no independent-video claim, and no claim that language hidden-state coupling or identity-conditioned Grounding has been solved.

Training labels used legal MOT GT only for causal supervision; previous-frame region tokens formed teacher-forced memory slots. This teacher-forced condition is explicit and must be removed or compared against learned memory in the next experiment. The proposal source was native GroundAnything output, so this pilot did not replace GroundAnything candidates with all-GT boxes.

The first run exposed and fixed a real field-name mismatch between legacy `track_id/lost_age` and the new `persistent_track_id/lost_duration` memory schema during replay.
