# Research log append — corrected three-task source registration

The historical full-source OneJev manifest was rejected for the independent
comparison because its inventory included MOT20-05 rows. The replacement
manifest was streamed and audited at
`/data2/user/LocateMOT_GroundJEV/outputs/groundjev/training_data/real_three_task_joint_v1/`:
131,214 native MOT identity rows, 346 native RMOT membership/presence rows,
and 430 registered OVMOT Base teacher-control rows. Duplicate IDs, train
split, family filters, row counts, media paths, forbidden held-out markers,
and source SHA-256 values passed. The registered TAO native archive contains
40 frames and 35,228 candidates but no category supervision, so native OVMOT
training rows remain pending. No optimizer update or GPU launch was performed;
the old independent four-GPU OneJev queue remains paused.

The registered `.venv-onejev-py310` loader subsequently loaded all 131,990
manifest rows and reproduced the five task-family counts exactly. System
Python was rejected as a training runtime because it lacks the OneJev
`pydantic` dependency.
