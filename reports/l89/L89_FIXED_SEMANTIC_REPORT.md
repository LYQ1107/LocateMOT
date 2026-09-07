# L89 fixed semantic report

## Scope and isolation

This is one fixed 16-calibration/24-validation diagnostic after the L89
checkpoint and rule had been frozen on legal fit/dev groups and full-video dev
TrackEval. It is not screening, official-test evaluation, or a production
RMOT result. The evaluator built and scored all 40 L69 rows before attaching
labels; calibration labels were attached first and validation labels only
after the frozen selection was loaded. The preselection audit records the
forbidden label fields as absent. Candidate rows remained complete and in
native order; there was no top-k, NMS, deletion, or truncation.

Authoritative output:

```text
outputs/l89/eval/fixed_semantic_retry4/
```

The selected checkpoint is epoch 4 and the selected rule is R. The frozen
rule is candidate threshold `-1`, presence threshold `-1`, and null margin
`0`. This was not reselected on validation.

## Overall validation gate

| metric | immutable L29 | L89 frozen epoch4/R | gate |
|---|---:|---:|:---:|
| top-1 | — | 0.6154 | — |
| top-5 | — | 0.8462 | — |
| candidate recall | 0.7333 | 0.8710 | pass |
| precision | 0.0830 | 0.0794 | **fail** |
| FP/frame | 10.1250 | 13.0417 | **fail** |
| predictions/positive | 8.8333 | 10.9677 | **fail** |
| hard violation | 0.9167 | 0.6923 | pass (decrease 0.2244) |
| multi-positive recall | 0.8194 | 0.9028 | pass |
| empty rate | historical control | 0.0000 | — |
| inactive false acceptance | — | 1.0000 | **fail** |

The registered simultaneous gate is therefore `semantic_gate_fail`. The model
does produce lower fixed-slice hard violation and preserves recall and
multi-positive recall, but it accepts too much volume and does not produce a
usable no-match decision. The lower hard violation cannot be called a
correspondence success while precision, false-positive volume, and inactive
acceptance fail together.

## Domain and category slices

| slice | units | rows | recall | precision | FP/frame | pred/positive | hard | multi | inactive FA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | 12 | 655 | 0.9375 | 0.0955 | 11.8333 | 9.8125 | 0.7143 | 1.0000 | 1.0000 |
| V2 | 12 | 813 | 0.8000 | 0.0656 | 14.2500 | 12.2000 | 0.6667 | 0.8056 | 1.0000 |
| inactive | 6 | 347 | 0.0000 | 0.0000 | 12.8333 | 77.0000 | — | — | 1.0000 |
| multi-positive | 6 | 382 | 0.8750 | 0.2283 | 11.8333 | 3.8333 | 1.0000 | 0.9028 | 0.0000 |
| positive | 7 | 399 | 0.8571 | 0.0612 | 13.1429 | 14.0000 | 0.4286 | — | 0.0000 |
| present-uncovered | 5 | 340 | 0.0000 | 0.0000 | 14.6000 | 73.0000 | — | — | 0.0000 |

The V2 slice is the weaker domain in volume and hard-negative behavior. The
present-uncovered rows are retained as a separate coverage-masked category;
they are not treated as ordinary all-negative semantic examples.

## Machine evidence and flags

- `record_count=40`, split `16/24`;
- finite scores and complete row arrays for every record;
- `candidate_rows_retained=true`, `candidate_deletion=false`,
  `candidate_truncation=false`;
- fixed L29 constants were reported from the accepted immutable control, not
  re-fitted from current rows;
- `screening_gt_used=false`, `official_test_labels_read=false`,
  `ordinary_mot_ovmot_touched=false`, `hota_trackeval_run=false`.

No HOTA claim is made here. The required internal TrackEval evidence is a
separate report.
