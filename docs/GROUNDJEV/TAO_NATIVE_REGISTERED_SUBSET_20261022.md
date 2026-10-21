# Registered native TAO TRAIN subset

The immutable TAO TRAIN parent scope contains 18,274 frames from 500 videos.
Running all 1,203 vocabulary queries on every frame was stopped before any
terminal candidate receipt because it was not a practical prerequisite for the
coupled training gate and would create an unnecessary full cache.

The source manifest is preserved. A separate registered scope selects one
eligible annotated frame from 64 evenly spaced TAO TRAIN videos, for 64 frames
and 64 videos, split into four 16-frame shards. Each shard still requests all
1,203 vocabulary classes and keeps the GT-free native inference contract. The
scope is explicitly marked `registered_train_subset=true` and
`full_train_image_scope=false`.

The resumable owner is
`scripts/groundjev/queue_tao_native_train_subset.py`. It runs the private G2
native bundle on legal physical GPUs 4, 5, 6, and 7, records every shard
receipt, joins TAO TRAIN labels offline, and then builds the shared native
MOT/RMOT/OVMOT manifest. It cannot be interpreted as a full-TAO benchmark.
