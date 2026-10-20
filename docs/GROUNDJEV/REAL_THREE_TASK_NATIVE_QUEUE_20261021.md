# GroundJEV real native MOT/RMOT/OVMOT OneJev owner — 2026-10-21

`scripts/groundjev/queue_real_three_task_native.py` is the separate resumable
entrypoint for the real native three-task run. It has not launched an
optimizer. It first requires a joined manifest with
`native_ovmot_training_rows=true`, `ovmot_training_mode=native`, a positive
row count, and a CPU-only manifest receipt. It then waits for exact terminal
G2 Clean, G1, deployment-bundle and independent-held-out receipts and an
exclusive four-GPU window on physical GPUs 4/5/6/7.

The prior compact teacher-control queue remains paused at zero optimizer
updates. This owner writes a distinct runtime config and checkpoint root, so
it cannot overwrite the historical course or the G2 Clean run.

The owner is intentionally registered only after the native join owner has
produced the manifest; until then no owner state or GPU process is started.
