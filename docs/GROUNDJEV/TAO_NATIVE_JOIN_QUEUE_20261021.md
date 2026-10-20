# GroundJEV native TAO TRAIN join owner — 2026-10-21

The CPU-only owner `scripts/groundjev/queue_native_ovmot_join.py` is live at
`WAITING_COMPLETE_NATIVE_TAO_STREAM`. It waits for the strict
`COMPLETE_FULL_NATIVE_TAO_TRAIN_CANDIDATE_STREAM` receipt from the eight-shard
export, then runs the checked-out repository joiners:

- `tools/groundjev/prepare_native_ovmot_train_join.py` assigns official TAO
  TRAIN Base category targets only by deterministic one-to-one IoU matching;
  candidate boxes and native provenance are preserved.
- `tools/groundjev/prepare_real_three_task_joint_manifest.py` replaces the
  teacher-control OVMOT component only after the complete native join passes.

The owner launches no model service, GPU process, or optimizer update. The
join entrypoints resolve from `/data2/user/LocateMOT`; the long-lived runtime
data and outputs remain under `/data2/user/LocateMOT_GroundJEV`. Partial or
existing output directories are never overwritten.

Current owner state: `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/training_data/native_ovmot_join_queue_v1/owner_state.json`.
