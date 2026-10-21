# G2 Clean training dataset scope — 2026-10-21

G2 Clean is the GroundAnything perception adaptation run (projector + Qwen3 LoRA). It is not J1/OneJev identity training. The run consumes the TRAIN-only manifest with 583,251 rows; manifest SHA is `f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71`.

| Dataset | Actual training rows | Role |
|---|---:|---|
| AerialMind | 69,358 | dense + referring grounding |
| AnimalTrack | 11,543 | dense animal tracking frames |
| DanceTrack | 41,796 | dense person frames |
| MOT17 | 15,948 | dense person frames |
| MOT20 | 5,616 | dense person frames; MOT20-05 is excluded/held out |
| RMOT26 | 294 | referring rows with initial target boxes |
| Refer-KITTI-V2 | 379,341 | legal fit-video referring rows |
| SportsMOT | 28,574 | dense person frames |
| TAO-Amodal | 30,781 | Base-category rows only |

Excluded or missing sources are explicit: BURST has no legal TRAIN image root, and MOT20-05 is the frozen held-out long-export sequence. TAO novel annotations are excluded from this training manifest. The split policy is TRAIN only with no evaluation labels.
