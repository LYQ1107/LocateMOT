# U1 formal expression grounding gate

- Execution code commit: `75264a762824469725ff963f99d0fef42691af9d`
- Frozen legal split SHA-256: `8ee806048ccbb227919f27edf68cf0bebc1fa50f824a8108878fc5f39fc6e1c0`
- Foundation checkpoint SHA-256: `55949c9c0f46339a73b415334765615d491ee6ed739ed3f568142b7fc5581143`
- Preprocessing: LoadImageFromFile -> FixScaleResize(800,1333,keep_ratio=True) -> PackDetInputs -> DetDataPreprocessor(bgr_to_rgb=True); boxes restored to original coordinates.
- Unit: `dataset/video/query_id/representative_frame/target_id`.
- Query universe: every legal calibration/validation query with one deterministic middle non-empty labeled frame; every query also contributes its first deterministic empty-target frame when available.
- Source-missing target references remain in the source denominator and are excluded only from valid-positive metric denominators.

## Gate result

- Status: **FORMAL_COMPLETE_REPRESENTATIVE_FRAME_PER_QUERY**
- Gate pass: **False** (required V1 and V2 Top1 Recall@IoU0.50 >= 0.75)
- Query frames: `7160` (positive and empty rows)
- Target references: `7199`
- Valid target units: `7199`
- Source-missing target units: `0`

| scope | Top1 @.25/.50/.75 | Top5 @.25/.50/.75 | Top10 @.25/.50/.75 | Top100 @.25/.50/.75 | Top300 @.25/.50/.75 |
|---|---|---|---|---|---|
| overall | 0.0104/0.0028/0.0000 | 0.0721/0.0057/0.0000 | 0.1227/0.0082/0.0000 | 0.4434/0.0389/0.0046 | 0.6959/0.1163/0.0111 |
| refer_kitti_v1 | 0.0089/0.0030/0.0000 | 0.0503/0.0089/0.0000 | 0.1065/0.0148/0.0000 | 0.4379/0.0503/0.0074 | 0.6538/0.1464/0.0178 |
| refer_kitti_v2 | 0.0106/0.0028/0.0000 | 0.0744/0.0054/0.0000 | 0.1243/0.0075/0.0000 | 0.4440/0.0377/0.0043 | 0.7003/0.1131/0.0104 |

The three numbers in each cell are target Recall@IoU0.25, Recall@IoU0.50, and Recall@IoU0.75.

## Additional query-level metrics

| scope | precision@Top10 IoU.50 | AP50 | single-target recall@Top10 | multi-target recall@Top10 | multi-target exact set | empty FP/query | predictions/query score>=.05 |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 0.0016 | 0.0056 | 0.0083 | 0.0081 | 0.0000 | 12.4705 | 11.4049 |
| refer_kitti_v1 | 0.0029 | 0.0080 | 0.0208 | 0.0124 | 0.0000 | 8.8555 | 9.4335 |
| refer_kitti_v2 | 0.0015 | 0.0054 | 0.0072 | 0.0077 | 0.0000 | 12.8754 | 11.6158 |

Top10 is the registered precision/set-coverage operating point; score `0.05` is fixed before reading formal results for empty-frame false-positive counts. The IoU0.50 Top1 recalls are far below the 0.75 gate. No expression adaptation checkpoint has been selected and no U2 work has started. Raw worker shards remain outside the repository.
