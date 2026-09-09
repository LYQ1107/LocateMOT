# R0 dense data contract report

## Result

`R0_DATA_CONTRACT_INVALID`. The R0 dense training index was intentionally not
constructed. The failure is a data-contract violation, not evidence that the
R0 visual representation or model is weak.

The authoritative machine evidence is
`outputs/r0/data/dense_contract_attempt1/status.json`, with the complete first
trace in `INCOMPLETE.md`. The audit loaded exactly 5,314 L49 fit rows:

| dataset | fit rows | `(video, query_id)` groups | contract violations |
| --- | ---: | ---: | ---: |
| Refer-KITTI V1 | 2,670 | 532 | 422 |
| Refer-KITTI V2 | 2,644 | 2,062 | 229 |
| total | 5,314 | 2,594 | 651 |

The sentence was stable for the first offending group, but its target tuple was
not. For `refer_kitti_v1/0001/query_id=0`, the same expression
`black cars in right` has frame-row target sets `['45']`, `['57']`,
`['57','58']`, `['84']`, and `['9']`. This is consistent with frame-level
visibility/annotation semantics, but it violates the R0 preregistered rule
that repeated fit rows agree on `sentence + target_ids`.

The implementation does not take a target union, choose a majority target, or
silently reinterpret a frame label. It stops at the first invalid contract as
required. The diagnostic records all 651 disagreement groups while preserving
the first actionable error.

Native L69 frame-pointer reading and the R0 label-attachment boundary are
implemented in `locatemot/rmot/r0_dense_data.py`, but no visual feature cache or
training index containing a repaired label interpretation was produced.

## Scope and safety

Only the permitted V1/V2 fit metadata was read. No screening or official-test
labels, detector cache, raw image, dense feature cache, TrackEval, HOTA,
ordinary MOT, or OVMOT asset was touched. The fixed manifest SHA remained
unchanged. The only valid next action is a separately approved repair or
explicit re-registration of frame-specific target visibility semantics before
R0 training.
