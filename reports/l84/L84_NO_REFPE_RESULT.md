# L84 no-refPE result

The original paired run selected R1 because its implementation reversed the
registered lexicographic tie-break.  That output is retained as historical
evidence.  A selection-correction audit applied the exact tuple and found a
Z1/R1 tie in saved checkpoint state and dev records; the earliest/simple
correct selection is Z1.

The corrected single structural test therefore rebuilt the same process-local
states with reference positional encoding removed from decoder content while
retaining reference points/query positional/deformable paths.  It used the
same three seeds, schedule, loss and ten epochs.  Output:
`outputs/l84/train/selection_correction_attempt2/`.

| quantity | original Z1 mean | Z1 no-refPE mean |
|---|---:|---:|
| bag hard violation | 0.705128 | 0.722222 |
| hit@1 | 0.407051 | 0.409188 |
| V2 bag hard violation | 0.766143 | 0.771379 |

The no-refPE rule required hard improvement of at least `.02`, hit@1
improvement of at least `.02`, and no V2-hard worsening.  It failed (`pass=false`):
hard worsened by `0.017094`, hit improved only `0.002137`, and V2 hard
worsened by `0.005236`.  The final representation remains original-content
Z1.

The failed schedule-metadata attempt at
`outputs/l84/train/selection_correction_attempt1/` is preserved.  Its first
error was using a one-process schedule against the registered world-size-4
schedule; it produced no model result.
