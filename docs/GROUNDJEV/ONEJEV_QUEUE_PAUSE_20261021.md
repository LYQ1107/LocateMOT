# Historical OneJev queue pause — 2026-10-21

The independent four-GPU OneJev source-resume owner was waiting before optimizer launch. It is paused for the GroundJEV single-backbone mainline.

- Previous state: `WAITING_G2_G1_AND_EXCLUSIVE_GPU_SET`
- Durable resume step: `3000`
- Historical non-durable observation: step `3525`
- Checkpoint: `/data3/user/LocateMOT-checkpoints/onejev_full_joint_source_epoch1_v1/latest_training_state.pt`
- Expected historical course: `28690` steps
- Owner PID `3994216` was terminated before any new GPU claim.
- The checkpoint, optimizer/scheduler/RNG state, manifest and logs are retained.
- A queue marker and a code guard prevent accidental automatic restart; only `--resume-paused` explicitly re-registers this historical comparison.

This pause does not claim a OneJev metric result and does not modify the protected G2 Clean owner.
