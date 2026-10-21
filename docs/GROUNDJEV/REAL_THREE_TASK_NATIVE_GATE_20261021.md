# Native OVMOT gate for the real three-task OneJev course — 2026-10-21

The registered v2 source manifest has 131,990 rows: 131,214 native MOT identity rows, 346 native RMOT rows, and 430 OVMOT Base category teacher-control rows. Its native TAO archive is a 40-frame validation/native replay archive with no category labels. It is not legal evidence for native OVMOT training.

The waiting four-GPU OneJev owner was therefore stopped before any optimizer update and its state was preserved as `teacher_control_snapshot_20261021.json`. The owner state is now `PAUSED_FOR_NATIVE_OVMOT_ROWS` with `optimizer_updates=0`.

The queue guard now requires both `native_ovmot_training_rows=true` and `ovmot_training_mode=native` in the manifest. Until a legal TAO TRAIN candidate export is joined to TRAIN annotations, the queue records `WAITING_NATIVE_OVMOT_ROWS` and cannot launch. This prevents a teacher-only OVMOT course from being reported as the requested real MOT/RMOT/OVMOT joint optimization.

This is a protocol correction, not a training result. The historical four-GPU source course remains paused and untouched.
