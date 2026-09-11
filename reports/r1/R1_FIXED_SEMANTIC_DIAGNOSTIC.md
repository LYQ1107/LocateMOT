# R1 fixed 16-calibration/24-validation semantic diagnostic

## Scope

This is an internal fixed-slice diagnostic after legal-development checkpoint
selection. It is not a preregistered final semantic gate and is not a
screening, official-test, HOTA, or final RMOT result.

The evaluator used the immutable L62 40-unit order: 16 calibration units
followed by 24 validation units. It reconstructed L69 rows from each bank's
own frame pointers and retained all 2,435 candidate rows (V1 1,237; V2
1,198). The L69 candidate set is different from the old L29 candidate set;
therefore no row-paired equality with L29 is claimed. The fixed slice contained
11 inactive, 8 multi-positive, 14 positive and 7 present-uncovered units.

Features and scores were constructed before labels were attached. Target IDs
and candidate sidecars were then attached for the diagnostic only. All 40
native unit keys, candidate counts, score lengths and finite checks passed;
candidate deletion and truncation were false.

## Frozen controls and outputs

Authoritative output directory:
`outputs/r1/eval/fixed_semantic_attempt1/`.

The frozen legal-development selections used the checkpoint SHAs:

- V1 `aec170e1c7a792c0b8ae47dab783232b17baf028beb78de3ad7d35333e9f271c`.
- V2 `ec234aca4f2fa5821b12d3c5c840d80ea156e87e550aedf19580fef032ae3439`.

The evaluator kept the L89E Rule-B values unchanged: candidate threshold
`1.0`, presence threshold `0.5`, null margin `0.0`. No validation-based
threshold or checkpoint choice was made.

## Aggregate diagnostic metrics

| Method/output | Recall | Precision | FP/frame | Pred/positive | Hard violation | Multi-positive recall | Empty | Inactive false acceptance | Top-1 | Top-5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Immutable L29 control | 0.7333 | 0.0830 | 10.1250 | 8.8333 | 0.9167 | 0.8194 | — | — | — | — |
| Frozen anchor, candidate-only | 0.4714 | 0.1844 | 3.6500 | 2.5571 | 0.7727 | 0.4304 | 0.1750 | 0.9091 | 0.5455 | 0.8636 |
| R1, candidate-only | 0.1000 | 0.1842 | 0.7750 | 0.5429 | 0.7727 | 0.1818 | 0.7250 | 0.2727 | 0.4545 | 0.7727 |
| Frozen anchor, Rule B | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7727 | 0.0000 | 1.0000 | 0.0000 | ranking-only | ranking-only |
| R1, Rule B | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7727 | 0.0000 | 1.0000 | 0.0000 | ranking-only | ranking-only |

The R1 candidate-only output selects only 38 rows at the fixed threshold and
has a 0.725 empty rate. Under the frozen Rule-B emission contract, neither
anchor nor R1 emits a row on this slice because the internal score scale does
not reach the fixed decision boundary. This is recorded as a diagnostic scale
and volume mismatch, not repaired by changing the threshold or adding NULL
filtering.

R1's candidate-only hard violation is identical to the anchor diagnostic
(`0.7727`) and its multi-positive recall is substantially lower. The apparent
precision increase over L29 is therefore a volume/recall collapse, not a
correspondence fix. The fixed Rule-B zero-emission result cannot be used as a
semantic success.

## Provenance boundaries

- `prediction_before_label=true` and labels were attached after feature/score
  construction for all 40 units.
- `screening_gt_used=false`, `official_test_labels_read=false`,
  `ordinary_mot_ovmot_touched=false`.
- `hota_trackeval_run=false` for this diagnostic; legal-development TrackEval
  is reported separately in `R1_LEGAL_DEV_REPLAY_REPORT.md`.
- `token_span_region_alignment=UNALIGNED` and
  `static_motion_alignment=UNALIGNED`.

The diagnostic is retained as evidence that R1 did not produce a stable,
deployable semantic improvement on the fixed slice.
