# GroundJEV real native MOT/RMOT/OVMOT OneJev owner — 2026-10-21

`scripts/groundjev/queue_real_three_task_native.py` is the separate resumable
entrypoint for the real native three-task run. Its persistent owner is live at
`WAITING_NATIVE_OVMOT_MANIFEST` and has not launched an optimizer. It first requires a joined manifest with
`native_ovmot_training_rows=true`, `ovmot_training_mode=native`, a positive
row count, and a CPU-only manifest receipt. It then waits for exact terminal
G2 Clean, the independent deployment-bundle and held-out receipts, and an
exclusive four-GPU window on physical GPUs 4/5/6/7. The G1 GPU5 adaptation is
recorded as an optional comparison arm and is not a logical prerequisite; the
four-GPU resource gate still prevents physical overlap while G1 owns GPU5.

The prior compact teacher-control queue remains paused at zero optimizer
updates. This owner writes a distinct runtime config and checkpoint root, so
it cannot overwrite the historical course or the G2 Clean run.

The owner was started as a dependency waiter before the native join exists;
this creates no GPU process. It will continue to wait, then write a distinct
runtime config and apply the remaining gates once the native manifest is
produced.

Current owner state: `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/training/real_three_task_joint_native_v1_queue/owner_state.json`.
