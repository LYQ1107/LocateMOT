# TAO TRAIN native OVMOT scope — 2026-10-21

Registered all 18,274 TAO TRAIN images across 500 videos at `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/training_data/tao_native_ovmot_train_scope_v1/manifest.json`. The manifest contains only image paths and the fixed 1,203-category query vocabulary. It records `training_on_this_scope=true` for the exporter, while `runtime_GT_fields=false` and `official_test_labels_read=false`; labels are kept for the separate offline candidate/annotation join.

The native exporter now requires an explicit `--allow-training-scope` flag. It rejects validation manifests when the flag is present and rejects TRAIN manifests without it. The exporter was not started because the current GPUs are occupied by protected/foreign jobs; no native candidate rows or optimizer updates exist yet.

After an exclusive resource window, run the exporter into a new directory, then pass its complete `/frames/train/` archive to `prepare_native_ovmot_train_join.py`. Only that join can set `native_ovmot_training_rows=true`; the three-task queue remains paused until then.
