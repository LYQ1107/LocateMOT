# G2 Clean progress snapshot — step 12873

- Run: `/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/formal_clean_coverage_v2_final`
- Event time: `2026-10-21 06:01:24 +0800`
- Progress: `12873/18227`; remaining `5354` updates.
- Latest loss: `0.272433674`; gradient norm `11.859652`; peak memory `12.828881` GiB.
- Measured throughput: `310.88`, `312.23`, `311.53` steps/hour over the last 100/500/1000 updates.
- Linear ETA from the 1,000-step window: approximately `2026-10-21T23:12:34.824000+08:00` (`17.19` hours).
- Durable recovery pair: `adapter-step012500.pt` and `latest_training_state.pt`; `final_receipt.json` is still absent.
- Downstream G1, bundle export, held-out evaluation, native OVMOT join, three-task OneJev training and YesTrack remain gated; no downstream optimizer was launched by this snapshot.

This is a progress and recoverability record only. It does not claim a detector or tracking metric.
