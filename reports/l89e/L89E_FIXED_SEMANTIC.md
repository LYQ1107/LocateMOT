# L89E fixed semantic replay

This is a fixed 16-calibration/24-validation semantic diagnostic, not
screening, official-test, HOTA or a production RMOT result.  The selected
epoch/rule/threshold was frozen before validation.  The unchanged corrected
candidate-vs-NULL emission contract was used.

| metric | validation |
|---|---:|
| recall | 0.4516129 |
| precision | 0.1458333 |
| FP/frame | 3.4166667 |
| predictions/positive | 3.0967742 |
| hard violation | 0.7692308 |
| multi-positive recall | 0.3611111 |
| inactive false acceptance | 0.8333333 |
| empty rate | 0.2083333 |

Decision: `semantic_gate_fail`.  The registered recall and multi-positive
floors are not met, so this remains a semantic-gate failure despite lower
volume and improved precision.  No threshold rescue was performed.
