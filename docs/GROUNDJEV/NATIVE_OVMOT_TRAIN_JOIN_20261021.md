# Native OVMOT TRAIN candidate/annotation join — 2026-10-21

The new [`prepare_native_ovmot_train_join.py`](../../tools/groundjev/prepare_native_ovmot_train_join.py) is ready to convert a complete native GroundAnything TAO TRAIN candidate stream into OneJev category rows. It keeps every detector box unchanged, performs a deterministic one-to-one IoU match against official TAO TRAIN annotations offline, and writes gzip JSONL rows with real-image crop recipes. The model's query label is retained as provenance and is never used as the category target.

The registered TAO TRAIN source has 18,274 images, 500 videos and 54,639 annotations. It contains 765 Base categories that map to the 1,203-class input vocabulary and 52,978 eligible Base annotations; 11 low-frequency Base category IDs have no exact vocabulary entry and are reported rather than guessed.

The only existing native TAO archive is validation-only (40 frames and 35,228 candidates), so the script rejects its `/frames/val/` paths. A complete `/frames/train/` archive over all 18,274 images is still required. Until that archive and its join manifest exist, `native_ovmot_training_rows` remains false and the three-task queue stays `PAUSED_FOR_NATIVE_OVMOT_ROWS`.

Once the full join exists, `prepare_real_three_task_joint_manifest.py --ovmot-native-manifest <manifest.json>` can replace the teacher-control component and emits the explicit `native_ovmot_training_rows=true`, `ovmot_training_mode=native` gate fields required by the queue.

`py_compile`, a one-frame synthetic preflight (one matched candidate and two category rows), and explicit rejection of the validation archive passed. No GPU call, optimizer update, or OVMOT performance claim was made.
