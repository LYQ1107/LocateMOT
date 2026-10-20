# G2 Clean live progress — step 13065

- Observed: `2026-10-21T06:38:27.069757+08:00`
- Run: `/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/formal_clean_coverage_v2_final`
- Status: **RUNNING**, `13065/18227`
- Latest event time: `2026-10-21 06:38:25 +0800`
- Loss: `0.2548696502344683`; gradient norm: `9.990986824035645`
- Peak event memory: `12.828814506530762` GiB
- Latest durable checkpoint: `/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/formal_clean_coverage_v2_final/checkpoints/latest_training_state.pt` (SHA-256 `4a79cbf3e64e69ffc56aaf964754d0209382b87c2124d46f13506325ca52fd22`)
- Final receipt: absent at observation time
- GPU snapshot: `3, 45, 19293, 81920; 5, 54, 19321, 81920`

Both distributed ranks were live and the event stream had advanced beyond the previous step-13014 audit.
This is a training-health receipt only; it does not claim MOT/RMOT/OVMOT performance. G1, TAO/OVMOT joining, native three-task OneJev training and independent held-out evaluation remain gated on the terminal G2 receipt.
