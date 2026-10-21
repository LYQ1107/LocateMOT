# Native OVMOT TRAIN archive audit — 2026-10-21

A read-only search across the registered data roots `/data0/user`, `/data1/user` and `/data3/user` found no complete TAO TRAIN `frame_candidates.jsonl` or equivalent native OVMOT candidate archive. The available TAO files are annotations, not model-generated candidates.

The existing native archive at `outputs/groundjev/multitask_runs/tao_base_novel63_full_v2_expression_none/native_candidates/frame_candidates.jsonl` contains 40 frames whose image paths are under `TAO-Amodal/frames/val`. It is therefore validation scope and cannot be joined into the TRAIN manifest. The corrected three-task OneJev queue remains `PAUSED_FOR_NATIVE_OVMOT_ROWS` with zero optimizer updates. No validation data was relabeled, and no teacher-only fallback was launched.

The next legal unlock is a native candidate export over TAO TRAIN images followed by the offline Base-category join. This audit does not claim that export has started.
