# L88C offline diagnosis

Machine-readable source: `outputs/l88c/diagnosis/root_cause_attempt2/root_cause.json`.
This diagnosis reads only existing L88/L88C replay, dev, TrackEval and
checkpoint/training artifacts. It performs no training, threshold selection,
screening read, or new model forward.

## Findings A–I

- **A — candidate representation ranking:** on the 13 fixed validation units
  with both positives and negatives, the existing candidate score has mean
  unit pairwise positive-over-negative accuracy `0.895934`, but the best
  positive rank has mean `5.38` and reaches rank 25. A high average pairwise
  number does not preserve the lowest-scoring positive in a multi-target bag.
- **B — precision/recall frontier:** the fixed diagnostic frontier shows that
  the accepted final threshold emits a small subset. The candidate-only
  recall is `.419355`; adding the corrected candidate-vs-NULL rule reduces it
  to `.354839`. The precision/volume improvement is therefore accompanied by
  a recall loss, not a correspondence solution.
- **C — presence/inactive behavior:** inactive false acceptance is `.666667`,
  not universal, while the learned presence logit remains positive on the
  inactive slice. NULL/presence calibration contributes to errors but is not
  the sole bottleneck.
- **D — candidate-vs-NULL margins:** on validation, candidate-score minus
  NULL has mean `-1.7908` for positive rows and `-8.7421` for negative rows.
  The separation is real but the positive distribution is broad; a single
  emission rule rejects many valid positives.
- **E — multi-target collapse:** all six fixed multi-positive validation units
  are retained in the analysis, but their minimum-positive scores are low
  (mean `-6.976`) and the registered final multi-positive recall is only
  `.305556`. The head is not preserving all positives as a bag.
- **F — V2 generalization:** fixed V2 recall is `.266667`, hard violation is
  `1.0`, and multi-positive recall is `.166667`; selected internal dev V2
  HOTA/DetA/AssA is `.265682/.156622/.451938`, below V1
  `.296615/.216472/.408302` on HOTA/DetA. V2 remains the harder domain.
- **G — LoRA deltas:** saved scaled LoRA delta Frobenius norm grows from
  `1.78999` (epoch 2) to `20.53365` (epoch 40), with no missing checkpoint
  or non-finite package. The growing adaptation is not evidence of held-out
  semantic success.
- **H — L84:** no direct L84 artifact was used in this diagnosis; no
  inference is made from it.
- **I — training trajectory:** loss decreases from `4.064745` at epoch 1 to
  `2.197868` at epoch 40, and saved gradient entries are finite/nonzero.
  Thus the stop is not an implementation-smoke or optimizer-finite failure.

## Root cause

The primary evidence-supported root cause is
`correspondence_and_multi_positive_recall_insufficient_after_corrected_emission`.
The candidate score can rank many positives above negatives, but it does not
keep every positive in a multi-target bag above the fixed candidate-vs-NULL
emission boundary. The failure is strongest on V2. Candidate coverage was
already separately found adequate by L76; this replay adds no evidence for a
proposal ceiling. The output is also not a universal NULL-acceptance failure.

No further L88C training, threshold retuning, top-k/NMS, or NULL suppression
is justified by these artifacts. One new structural experiment requires
supervisor approval; see `NEXT_TEST_APPROVAL_REQUEST.md`.
