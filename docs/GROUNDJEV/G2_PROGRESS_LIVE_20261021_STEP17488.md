# G2 Clean live progress — step 17488

The protected two-rank G2 Clean run is still live at `17488/18227`. The last
event at `2026-10-21 20:45:17 +0800` has finite loss `0.1523213747`, finite
gradient norm `8.3949747`, and peak `12.828824 GiB`.

The measured recent rates are `310.18`, `307.92`, and `311.03` steps/hour over
the 100, 500, and 1000-step windows. The 500-step linear estimate is about
`2.40` hours to the target. The durable recovery file is
`checkpoints/latest_training_state.pt`; the latest periodic adapter is
`checkpoints/adapter-step017000.pt`. No terminal receipt exists yet.

All downstream owners remain dependency-gated: G1 GPU5 and the G2 bundle wait
for the terminal Clean receipt, native OVMOT joining waits for a complete TAO
stream, and the three-task queue has zero optimizer updates. The independent
YesTrack training remains paused so the single-backbone GroundJEV coupling work
has priority. This is a training-health/recoverability record only; it makes no
tracking or scientific gain claim.
