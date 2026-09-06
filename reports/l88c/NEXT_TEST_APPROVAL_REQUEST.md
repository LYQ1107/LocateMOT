# L88C next-test approval request

## Evidence requiring a new test

The corrected replay passes the hard-negative and volume checks but fails
recall and multi-positive preservation. Existing training is finite and the
L76 audit found candidate coverage adequate, so another L88C epoch, threshold
grid, top-k rule, or NULL filter is not an evidence-based continuation.

## One proposed next structural test

Approve exactly one RMOT-only correspondence head that keeps a separate score
for every positive candidate in a target bag and uses an explicit calibrated
bag/multi-positive objective, with the frozen L69/L88 candidate bank and no
new proposal generator, tracker, production entrypoint, or screening labels.
The test would first have a label-free/data contract audit, then a bounded
fit-only smoke and the same frozen 16-calibration/24-validation gate. It would
not reuse L88 scores as teacher inputs and would not add top-k/NMS/NULL
suppression.

## Frozen controls and success criteria

Use the immutable L29 control and the L88C corrected-final records as
controls. The same simultaneous floors remain: recall `.7233333`, precision
`.0830188679`, FP/frame `11.125`, predictions/positive `4.069`, hard violation
at most `.8666667`, multi-positive recall `.7894444`, finite complete rows,
and no candidate deletion. No longer training or screening is authorized by
this request.

## Approval boundary

This is a request only. L88C is stopped pending supervisor approval; no new
architecture or experiment was started in this stage.
