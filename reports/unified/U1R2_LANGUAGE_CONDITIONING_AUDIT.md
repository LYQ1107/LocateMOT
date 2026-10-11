# LocateMOT-U U1R2 language-conditioning audit

The audit uses exactly 500 deterministic legal validation query/frame pairs.
For every pair, the same image is run with the true sentence and with a
sentence from the same video whose target set is different. Both
query-conditioned forwards finish before the source label is opened. AP50 is
computed over the top 300 predictions at IoU 0.50; target recall is computed at
Top10.

| metric | true expression | shuffled expression | margin |
|---|---:|---:|---:|
| Top10 target Recall@IoU0.50 | 0.959333 | 0.871133 | +0.088200 |
| AP50 (Top300) | 0.736901 | 0.491159 | +0.245743 |

Both strict comparisons pass. This demonstrates that the expression branch is
not merely emitting the same generic proposals independent of language. The
pair construction records the true and shuffled target IDs and asserts that
the sets differ.

Evidence: `tools/unified/u1_expression_counterfactual_audit.py` and
`outputs/unified/u1r2/expression_counterfactual_epoch1/result.json`.
