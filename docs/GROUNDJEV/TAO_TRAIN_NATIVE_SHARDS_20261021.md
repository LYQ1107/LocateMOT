# TAO TRAIN native export shards — 2026-10-21

The registered TAO TRAIN scope is now split into eight independent, contiguous runtime manifests. Each manifest contains only image paths, frame/video IDs and the fixed 1,203-class vocabulary; it carries no annotation boxes or category labels. All shards point to the same parent manifest SHA `835adaadff282088599987de4c5bc906e7d8aca826deb7426cda05598e9db3a4`.

| Shard | Parent indices | Frames | Videos |
|---:|---:|---:|---:|
| 0 | `[0, 2284)` | 2,284 | 60 |
| 1 | `[2284, 4568)` | 2,284 | 59 |
| 2 | `[4568, 6852)` | 2,284 | 59 |
| 3 | `[6852, 9137)` | 2,285 | 69 |
| 4 | `[9137, 11421)` | 2,284 | 59 |
| 5 | `[11421, 13705)` | 2,284 | 60 |
| 6 | `[13705, 15989)` | 2,284 | 64 |
| 7 | `[15989, 18274)` | 2,285 | 77 |

The CPU registration proves exact coverage of all 18,274 parent frames with zero missing or duplicate frames. No native candidate export or optimizer update has started; every shard still reports `native_ovmot_training_rows=false`. When an exclusive GPU window is available, each shard can be streamed independently with `stream_tao_native_candidates.py --allow-training-scope`. The streams must all be receipt-validated and concatenated in shard order before `prepare_native_ovmot_train_join.py` can open the native TRAIN gate.

The first preflight caught and repaired a local slice-length assertion bug before any valid shard was accepted. The failed partial directory was retained under the external output root with a `failed_preflight` suffix; the valid registration is `tao_native_ovmot_train_shards_v2`.
