# GroundJEV native three-task queue

The previous `real_three_task_joint_native_v1` owner was paused before it could
launch because its terminal command was the independent `run_source_manifest_sft.py`
OneJev trainer.  That route is incompatible with the single-backbone GroundJEV
direction.  Its owner state is preserved with status
`PAUSED_FOR_GROUNDJEV_SINGLE_BACKBONE` and `optimizer_updates=0`.

The replacement owner is `queue_groundjev_native_three_task.py`.  It waits for
the exact G2 Clean terminal receipt and the native TAO OVMOT three-task manifest,
then reserves GPU4 and invokes `train_native_three_task.py`.  The trainer uses
one G2 Clean DLM forward per unique input, freezes the G2 backbone for this
bounded phase, and updates only the shared typed JEV readout, causal memory
projector, and small OVMOT category heads.  GPU5 remains reserved for the G1
comparison and GPU7 for independent held-out evaluation.

The trainer retains source-row provenance and masks identity targets whose track
slot is not present in causal history.  It records per-task updates, selected
row IDs, manifest/model/adapter hashes, visual forward count, memory-history
audit, and the before/after task readout accuracy.  This receipt is a real
three-task shared-G2 readout experiment; it is not an independent OneJev or
YesTrack model and does not claim benchmark generalization by itself.

Static validation completed before queue registration:

- `py_compile`/`compileall` for the new trainer, task heads, and queue.
- CPU loader over MOT, RMOT, and OVMOT sources retained all three task families.
- Synthetic forward/loss/backward regression reached identity, RMOT Yes/No,
  and OVMOT group/leaf paths with finite gradients.
- Existing 11 coupling tests remained green.

The new owner is registered under
`/data2/user/LocateMOT_GroundJEV/outputs/groundjev/training/groundjev_native_three_task_v1_queue`.
