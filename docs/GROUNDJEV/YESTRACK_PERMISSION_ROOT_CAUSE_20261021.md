# YesTrack native v2 permission failure and preserved retry

## Scope

This record covers the first real YesTrack GPU attempt and the repaired retry
queue. It does not claim a YesTrack metric or a coupled OneJev optimizer update.

## Observed failure

The original queue
`/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/yestrack_native_v2_queue`
passed input validation and reached the GPU7-ready gate. It then exited before
training with `PermissionError(13, 'Permission denied')`. Its `TRAINING.log` has
no training record, and its terminal state is retained as
`REAL_YESTRACK_TRAINING_FAILED`.

## Root cause and fix

`scripts/groundjev/queue_yestrack_native_v2.py` executed
`tools/groundjev/run_with_resources.py` directly. On this host that source file
has mode `0666`, so the operating system rejected the exec before the wrapper
could create the training process. The queue now invokes the wrapper through
the pinned `/data2/user/LocateMOT_GroundJEV/.venv-onejev-py310/bin/python`.
The queue script and wrapper both pass compilation and the queue `--help`
entrypoint check.

## Retry state

The failed queue was not overwritten. A new owner is running at
`/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/yestrack_native_v2_queue_retry_permission_fix`.
At launch it entered `WAITING_GPU7_EXCLUSIVE`. GPU7 currently has an unrelated
`szgx` compute process (PID `1839747`, about `10196 MiB`), so the owner waits
under its explicit no-compute-client/32-GiB-free rule. No unrelated process was
stopped or modified.

The retry still requires a real GPU training terminal record (`2178` examples,
`273` steps), adapter, frozen-base validation, fine-tuned validation and a
terminal receipt before any YesTrack result is reported.

## Coupling boundary

The real three-task OneJev/G2 coupled optimizer remains gated by the missing
G2 Clean terminal receipt. It is still at zero optimizer updates; the current
G2 Clean run is the prerequisite and remains protected.
