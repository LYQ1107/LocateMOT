# L89C fixed semantic replay

## Frozen evaluation

The authoritative output is
`outputs/l89c/eval/fixed_semantic_corrected_attempt2/`. It has 40 ordered
records: the first 16 are calibration and the last 24 are validation. The
preselection audit reports all nine forbidden label fields absent and
`selection_frozen_before_fixed_labels=true`. Calibration-only fitting used the
corrected candidate-vs-NULL rule; validation was read once after the selection
and thresholds were frozen. The first attempt at the same replay is retained
as `fixed_semantic_corrected_attempt1/INCOMPLETE.md` because GPU0 was out of
memory.

Frozen selection: epoch 2, Rule R, checkpoint SHA
`f8b175597ece8aad1f0ec3ae9d05c70e0759a0f7480dff9af6ea2619a2e3f08b`. Frozen
thresholds are candidate `-0.75`, presence `-1.0`, and NULL margin `0.0`.
The actual emission is the shared corrected equation:

```text
(candidate_energy >= candidate_threshold)
and (candidate_energy - null_logit >= null_margin)
and (presence_logit >= presence_threshold)
```

No top-k, NMS, candidate deletion, or post-hoc NULL suppression was used.

## Metrics

| split | units | candidate rows | top1 | top5 | recall | precision | FP/frame | pred/positive | hard violation | multi-positive recall | empty | inactive FA |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| calibration | 16 | 967 | 0.555556 | 0.555556 | 0.717949 | 0.136585 | 11.0625 | 5.2564 | 0.777778 | 0.715128 | 0 | 1.0 |
| validation | 24 | 1468 | 0.615385 | 0.846154 | 0.838710 | 0.086957 | 11.375 | 9.6452 | 0.692308 | 0.833333 | 0 | 1.0 |

Validation V1/V2 rows were both retained and reported separately:

| split/domain | units | candidate rows | recall | precision | FP/frame | pred/positive | hard violation | multi-positive recall |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation/V1 | 12 | 655 | 0.875000 | 0.097902 | 10.750 | 8.9375 | 0.714286 | 0.833333 |
| validation/V2 | 12 | 813 | 0.800000 | 0.076923 | 12.000 | 10.4000 | 0.666667 | 0.833333 |

## Gate decision

The machine decision is `semantic_gate_fail` in
`outputs/l89c/eval/fixed_semantic_corrected_attempt2/gate_decision.json`.
Hard violation improves from the immutable L29 `0.916667` to `0.692308`, and
recall, precision, and multi-positive floors pass. The deployment gate still
fails three required conditions: FP/frame `11.375 > 11.125`,
predictions/positive `9.6452 > 4.069`, and inactive false acceptance remains
`1.0`. The candidate rows and scores are complete and finite, so this is not
an implementation or coverage-key failure. It is a deployable volume and
no-match failure after corrected candidate-vs-NULL accounting.

This result is not a screening result, official-test result, HOTA result, or
ordinary MOT/OVMOT result. `zero_training=true`.
