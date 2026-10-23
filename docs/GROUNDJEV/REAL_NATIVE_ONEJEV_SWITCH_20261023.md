# Real native OneJev switch — 2026-10-23

The historical `full_joint_source_epoch1_resume_after_heldout_v1` queue is
preserved as an immutable teacher-control/step-3000 course. The held-out
orchestrator no longer invokes it.

After all 27 registered held-out owners reach terminal receipts and the three
receipt finalizers complete, it invokes
`scripts/groundjev/queue_real_three_task_native.py` with a fresh queue and
checkpoint root. The queue reads
`real_three_task_joint_native_v1/train.sources.json`, which has
`native_ovmot_training_rows=true`, `ovmot_training_mode=native`, and 131,618
audited rows:

| task family | rows |
|---|---:|
| MOT identity | 131,214 |
| RMOT membership/presence | 346 |
| native TAO OVMOT category group/leaf | 58 |

The native OVMOT rows are a registered TAO TRAIN subset derived from the
preserved native candidate/annotation join. They are valid native TRAIN rows,
but they are not a full-TAO benchmark. The manifest receipt records zero
optimizer updates and excludes MOT20-05 from the source scan.

The new launch defaults to physical GPUs 5 and 7 (two distributed ranks),
requires at least 32 GiB free per selected device, records co-resident foreign
PIDs, and never terminates another user's process. The historical four-GPU
teacher-control queue and all existing G2/G1/G200 checkpoints remain
unchanged.
