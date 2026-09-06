# L88C failure decomposition

## Status

`STOPPED_PENDING_SUPERVISOR_REVIEW`.

The corrected candidate-vs-NULL replay is complete and reproducible. It is a
valid zero-training semantic failure, not a training or runtime failure.

## What passed

- Frozen manifest and L88 cache contracts remained valid.
- All five corrected dev candidates completed on V1 and V2 full video.
- Internal TrackEval matrix has 30 complete rows (5 epochs × 3 rules × 2
  domains), with selection frozen before fixed validation.
- Fixed semantic replay has 40 complete ordered records, finite scores, and no
  candidate deletion/truncation.
- The corrected gate reduces hard violation from the immutable L29 `.916667`
  to `.846154`, and precision/FP/frame/predictions-per-positive are within
  the registered ceilings.

## What failed

- Recall `.354839` is far below `.7233333`.
- Multi-positive recall `.305556` is far below `.7894444`.
- V2 is especially weak: recall `.266667`, hard violation `1.0`,
  multi-positive recall `.166667`.
- The candidate-only control has higher recall `.419355` but still fails the
  gate; the NULL comparison removes additional positives.

## Attribution

The pure corrected-gate control changes hard-negative accounting but not hard
violation on the historical L88 final records (`.909091` before and after),
while the full corrected final strategy improves it to `.846154`. The latter
also changes checkpoint/threshold, so the gain cannot be attributed to the
formula alone. The decisive failure is insufficient recall-preserving
candidate correspondence and multi-positive bag emission, not universal
NULL acceptance, missing proposal coverage, or finite/reload failure.

## Boundaries

No screening or official-test labels were read. No official HOTA, fast
screening, production RMOT, ordinary MOT, OVMOT or TAO result is claimed.
The internal dev TrackEval numbers remain clearly scoped to dev only.
