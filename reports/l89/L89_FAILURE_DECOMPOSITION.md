# L89 failure decomposition

## Decision

L89 is `STOPPED_PENDING_SUPERVISOR_REVIEW`. The contract smoke, complete
40-epoch fit, legal dev scoring/selection, fixed semantic replay, and final
internal V1/V2 TrackEval all completed. The semantic gate failed and both
internal HOTA values are below L87-A. No L89 continuation, screening, official
test, or production MOT/OVMOT action is authorized by this result.

## First actionable evidence

The first actionable root cause is not a finite/reload failure or a missing
expression-level supervision path. It is a failure of deployable
query-to-candidate emission/no-match calibration:

1. On the fixed 24 validation units, QSC-D reduced row hard violation from
   `0.9167` to `0.6923` and preserved recall (`0.8710`) and multi-positive
   recall (`0.9028`), so it exposed some local ranking signal.
2. The same frozen Rule R failed precision (`0.0794 < 0.0830`), FP/frame
   (`13.0417 > 11.125`), predictions/positive (`10.9677 > 4.069`), and
   inactive false acceptance (`1.0`). The empty rate was `0`, so this was not
   an empty-output recall collapse.
3. In final full-video internal inference, the selected rule emitted 17,591
   rows from 76,266 scored rows. Post-hoc target-row recall was only `0.0131`
   on V1 and `0.0026` on V2, while inactive acceptance remained `1.0`.
   Thus the fixed-slice ranking improvement did not become dense,
   persistent frame membership/track emission.
4. TrackEval confirmed the failure: HOTA was `2.2906` on V1 and `0.5894` on
   V2 versus L87-A `28.5752` and `22.1300`. DetA and AssA were also very low.

The evidence supports a combined emission-volume/absence-calibration and
sequence realization bottleneck in this L89 design. It does not justify
claiming that the frozen Z1 representation has no information, because the
fixed-slice hard-negative and multi-positive diagnostics did move. It also
does not justify claiming a successful correspondence method: the complete
deployable and sequence metrics fail.

## What was not the cause

- The QSC-D contract and loss smoke were finite and had nonzero gradients.
- All L69 candidate rows were retained; no top-k, NMS, deletion, or
  truncation was used.
- Candidate bank, L85 Z1 cache, language cache, fixed manifest, UIDM,
  ordinary MOT/OVMOT, and TrackEval code were not modified.
- L76 coverage evidence remains separate; L89 did not repair or remeasure
  candidate acquisition.
- Token/span-to-region and static/motion alignment remain `UNALIGNED`.

## Required next action

Do not extend L89, change its threshold, add NULL suppression, or select a
more favorable checkpoint. The unique next action is supervisor review of
the complete L89 evidence and authorization of one new, single-factor RMOT
design if warranted. Until that prompt arrives, keep L89 frozen and leave
screening/official-test labels and ordinary MOT/OVMOT untouched.
