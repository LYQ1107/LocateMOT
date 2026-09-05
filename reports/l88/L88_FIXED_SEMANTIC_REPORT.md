# L88 Fixed Semantic Evaluation

## Evidence boundary

This is one fixed 16-calibration/24-validation semantic evaluation after the
dev selection was frozen. It is not screening, official-test evaluation,
HOTA, or a TrackEval result. The evaluator used the L88 candidate rows in
their native order and separately read the immutable L29 records. It retained
all candidate rows, with no top-k, NMS, deletion, or truncation. The
preselection label-isolation record contains no target fields before the
calibration attach point.

The fixed manifest SHA256 remained:
`06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.

## Validation results

The L29 values below are the accepted immutable control. L88 candidate-only
means the candidate threshold is applied without the presence/NULL part of
Rule B. The final frozen Rule B column is the registered candidate plus
presence/NULL emission rule.

| method | recall | precision | FP/frame | predictions/positive | hard violation | multi-positive recall | empty rate | inactive false acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| immutable L29 | 0.7333333 | 0.0830189 | 10.1250 | 8.8333 | 0.9166667 | 0.8194444 | — | — |
| L88 candidate-only | 0.4193548 | 0.1830986 | 2.4167 | 2.2903 | 0.8461538 | 0.3333333 | 0.1250 | 0.8333333 |
| L88 frozen Rule B candidate+presence/NULL | 0.1935484 | 0.1363636 | 1.5833 | 1.4194 | 0.8461538 | 0.1666667 | 0.5417 | 0.5000 |

Rule B used candidate threshold `0.75`, presence threshold `0`, and NULL
margin `0` in the fixed semantic evaluator. The candidate-only result already
fails the recall and multi-positive floors, so the final failure is not being
hidden by a NULL-only choice. The presence/NULL component further lowers
emitted recall and increases empty output.

The fixed gate checks were:

- hard violation: pass (`0.8461538 <= 0.8666667`);
- precision: pass (`0.1363636 >= 0.0830188679`);
- FP/frame: pass (`1.5833 <= 11.125`);
- predictions/positive: pass (`1.4194 <= 4.069`);
- recall: **fail** (`0.1935484 < 0.7233333`);
- multi-positive: **fail** (`0.1666667 < 0.7894444`);
- finite complete keys and no candidate deletion/truncation: pass.

Thus the registered decision is `semantic_gate_fail`. The lower output volume
is not a correspondence success because it is accompanied by severe recall
and multi-positive collapse.

## Domain slices under the final frozen rule

| dataset | recall | precision | FP/frame | predictions/positive | hard violation | multi-positive recall | empty rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Refer-KITTI V1 | 0.2500 | 0.1538462 | 1.8333 | 1.6250 | 0.7143 | 0.2222 | 0.6667 |
| Refer-KITTI V2 | 0.1333333 | 0.1111111 | 1.3333 | 1.2000 | 1.0000 | 0.1111 | 0.4167 |

V2 retains a complete but weak candidate path and has hard violation 1.0 in
this slice. The low aggregate output count therefore cannot be interpreted
as a stable V1/V2 correspondence improvement.

## Machine-readable sources

- `outputs/l88/eval/fixed_semantic_attempt3/semantic.json`;
- `outputs/l88/eval/fixed_semantic_attempt3/gate_decision.json`;
- `outputs/l88/eval/fixed_semantic_attempt3/score_records.jsonl`;
  SHA256 `e3ddcdeb20a5ebcf3dc085e1f806e5723c09c7485748be5fcc0d614c92cb7e86`;
- `outputs/l88/eval/fixed_semantic_attempt3/preselection_label_isolation.json`.

The score record count is 40, split 16/24, and all finite/key/deletion checks
are recorded. No screening or official-test labels were read.

