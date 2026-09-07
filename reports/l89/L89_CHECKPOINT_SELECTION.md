# L89 checkpoint and rule selection

## Frozen selection protocol

All 20 even checkpoints (epochs 2 through 40) were scored on the legal
video-disjoint fit/dev split before fixed validation labels were read. The
dev score job covered 138 groups and emitted 498 complete candidate-set
records per checkpoint (9,960 JSONL records total). Every candidate row was
scored; no top-k, NMS, candidate deletion, or validation read occurred.

The registered shortlist was at most five entries: fixed epochs 8, 20, and
40, the best target-bag-F1 checkpoint, and the best distinct-target-recall
checkpoint satisfying the target-bag precision floor. This produced epochs
8, 20, 40, 32, and 4. Full-video dev TrackEval was run for B/R/P rules on
that frozen shortlist. The final tie order was higher HOTA, higher DetA,
higher AssA, higher distinct-target recall, lower inactive acceptance, and
earlier epoch, with deterministic rule order last.

## Dev shortlist matrix

HOTA, DetA and AssA are percentages converted from the local TrackEval
`[0,1]` values. The distinct-recall and inactive columns are post-hoc dev
descriptors used only by the registered selector.

| epoch | rule | HOTA | DetA | AssA | distinct target recall | inactive acceptance |
|---:|:---:|---:|---:|---:|---:|---:|
| 4 | B | 0.9623 | 0.2443 | 3.8423 | 0.3586 | 1.0000 |
| 4 | R | 1.2774 | 0.4119 | 4.0666 | 0.4669 | 1.0000 |
| 4 | P | 1.1683 | 0.3500 | 3.9787 | 0.4139 | 1.0000 |
| 8 | B | 0.8561 | 0.2144 | 3.4778 | 0.3524 | 1.0000 |
| 8 | R | 1.1943 | 0.3668 | 4.0094 | 0.4479 | 1.0000 |
| 8 | P | 1.0034 | 0.2766 | 3.7249 | 0.4097 | 1.0000 |
| 20 | B | 0.7907 | 0.2048 | 3.1354 | 0.3398 | 1.0000 |
| 20 | R | 0.9518 | 0.2800 | 3.3328 | 0.3971 | 1.0000 |
| 20 | P | 0.9518 | 0.2800 | 3.3328 | 0.3971 | 1.0000 |
| 32 | B | 0.8657 | 0.2206 | 3.4565 | 0.3226 | 1.0000 |
| 32 | R | 0.9200 | 0.2531 | 3.4203 | 0.3652 | 1.0000 |
| 32 | P | 0.7257 | 0.1525 | 3.5634 | 0.2843 | 1.0000 |
| 40 | B | 0.8021 | 0.1980 | 3.3213 | 0.3013 | 1.0000 |
| 40 | R | 0.8937 | 0.2417 | 3.3884 | 0.3568 | 1.0000 |
| 40 | P | 0.6655 | 0.1349 | 3.4166 | 0.2589 | 1.0000 |

The frozen result is **epoch 4, Rule R**, checkpoint SHA256
`5ab3cb344b73b320b34de7b4bb41622ce665ecb17c4d90c1999640a318e69aa8`.
Its fixed rule values are candidate threshold `-1`, presence threshold `-1`,
and null margin `0`. These values are inherited from the registered dev
selection; they were not chosen from validation.

Machine-readable authoritative paths:

- `outputs/l89/eval/dev_scores_joint40/dev_scores.json`;
- `outputs/l89/eval/dev_selection_joint40_retry1/checkpoint_selection.json`;
- `outputs/l89/eval/final_selection/final_selection.json`;
- `outputs/l89/eval/trackeval_dev_shortlist_joint40/trackeval_matrix.json`.

The internal validation replay used the frozen selection and did not use its
validation results to change the epoch or rule.
