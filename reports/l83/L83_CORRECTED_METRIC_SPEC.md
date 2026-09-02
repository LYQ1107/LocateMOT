# L83 corrected metric specification

## Scope

This file freezes the metric contract used by the L83 corrected baseline,
faithful probe, and decoder-sharpness diagnostic.  It is not a replacement for
any L82 report.  L82 assets and metrics remain immutable historical evidence.

## Corrections to the L82 protocol

1. Rows sharing one non-null `candidate_gt` are one target bag.  The primary
   score for that bag is the maximum row score.  `candidate_index` duplication
   is legal and rows are never deleted.
2. A `candidate_gt=None` row is a singleton background negative bag.
3. Target-bag hit@1, recall@5, minimum-positive margin, hard-negative
   violation, and multi-target exact top-T are computed over unique target bags;
   row AP/margins remain diagnostics.
4. Query-swap accuracy is computed only from exact same-frame, real-label
   flips.  The reported query-swap ROC-AUC is an independent rank-based AUC
   over those positive/negative scores, not pair concordance relabeled as AUC.
5. `present_uncovered` is masked from correspondence loss and is not changed
   into an inactive all-negative example.  Inactive rows are separately
   counted as explicit no-match cases.

The L82 AST/source audit at
`outputs/l83/audit/l82_protocol_mismatch_retry1/l82_protocol_mismatch.json`
records these five issues and their corrective evidence.

## Primary fields and denominators

- `target_bag_hard_violation`: hard negative bag comparisons with
  `max(negative) >= min(positive)` divided by all eligible comparisons.
- `target_bag_hit_at1`: eligible positive target queries whose top unique bag
  is positive divided by eligible target queries.
- `target_bag_recall_at5`: positive target bags in the first five unique bags
  divided by all positive target bags.
- `multi_target_exact_topT`: queries for which all target IDs appear in the
  top number-of-targets unique bags, divided by multi-target queries.
- `query_swap_pair_accuracy`: exact label-flip comparisons with positive score
  greater than negative score divided by exact label-flip pairs.
- `query_swap_roc_auc`: rank-based AUC with average ranks for ties over the
  exact label-flip score list.

All metrics are finite checks over the complete candidate row set.  They are
fit/dev diagnostics, not HOTA, TrackEval, screening, or official-test
metrics.

## Frozen gate values

The faithful probe gate is the pre-registered G1--G6 tuple in
`reports/l83/L83_PREREGISTERED_PLAN.md`.  No fixed historical 16/24 labels are
used by this metric contract; the faithful probe uses 524 fit groups for
training and 138 video-disjoint fit-derived dev groups.

`token_span_region_alignment=UNALIGNED` and
`static_motion_alignment=UNALIGNED` remain explicit.
