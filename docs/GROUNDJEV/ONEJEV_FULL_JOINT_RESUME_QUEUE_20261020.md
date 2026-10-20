# OneJev full-source joint resume queue — 2026-10-20

The registered full-source manifest contains **459,027 immutable rows**:
252,258 MOT, 100,813 RMOT, and 105,956 OVMOT rows. The native CPU plan
validated zero dropped rows and 28,690 optimizer updates at global 16 rows per
update.

The historical four-GPU run `onejev_full_joint_source_epoch1_v1` did execute
the real shared source mixture. Its log reached step 3,525, but it has no
terminal `done` record and no live owner process. The last atomic full
FP32/Adam/scheduler/all-rank-RNG state is step 3,000 at:

`/data3/user/LocateMOT-checkpoints/onejev_full_joint_source_epoch1_v1/latest_training_state.pt`

The recovery point is therefore step 3,000. The step-3,525 observations are
kept as non-durable diagnostics and are not presented as completed training.

`scripts/groundjev/queue_onejev_full_joint_resume.py` is the new resumable
owner. It never overwrites the historical output. It waits for:

1. the protected G2 Clean receipt (`18,227` steps and the registered manifest
   hash);
2. the GPU5 G1 adaptation and its isolated controls to finish successfully;
3. all selected physical GPUs 4, 5, 6, and 7 to have no compute process and at
   least 32 GiB free each.

Only then does it resume the exact four-rank protocol from the step-3,000 full
state. It records a nonzero exit, a missing terminal receipt, or an incomplete
step count as failure and leaves all checkpoints in place. No feature cache is
created.

This queue is a scheduling and recovery artifact, not a performance claim.
The source mixture still requires the final training receipt and the fixed
held-out MOT/RMOT/OVMOT evaluations before it can support a model claim.
