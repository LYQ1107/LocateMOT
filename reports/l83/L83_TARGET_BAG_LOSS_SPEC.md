# L83 faithful target-bag loss specification

The only trainable probe is `L83FaithfulRankProbe`, a 66,561-parameter
LN/Linear/GELU/Dropout/Linear head with input dimension 256.  It consumes the
existing representation state only; candidate IDs, target IDs, source/pool
IDs, query IDs, scores from another method, and GT identity are not model
inputs.

## Bag construction

For each `(dataset, video, frame)` group, rows with the same non-null
`candidate_gt` form one positive or negative target bag, and the primary bag
score is `max(row interaction)`.  A null `candidate_gt` row is retained as a
singleton background bag.  Duplicate candidate indices do not cause
deduplication.  The query target IDs determine which bags are positive only
after the complete representation rows have been constructed.

## Loss terms

- candidate-axis balanced target-bag classification;
- minimum-positive term so every positive target bag, including multi-target
  queries, receives gradient;
- query-axis exact same-frame label-flip term;
- explicit inactive/no-match term;
- `present_uncovered` correspondence masking;
- all-negative fallback for hard negatives because verified same-class
  metadata is unavailable.

The contract audit at
`outputs/l83/audit/loss_contract_final/contract.json` verified finite forward,
finite loss, nonzero gradients, multi-positive handling, present-uncovered
masking, independent AUC, and complete candidate rows.  Its parameter list is
`score.0/1/4.{weight,bias}` and its total parameter count is 66,561.

The faithful run uses seed `20260829`, AdamW (`lr=2e-4`, `weight_decay=1e-4`),
10 epochs, and the fixed 524-group train / 138-group dev split.  No fixed
calibration/validation labels are read.  Token/span-to-region and
static/motion alignment remain `UNALIGNED`.
