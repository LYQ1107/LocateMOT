# L88C final internal validation

## 1. Purpose and boundary

This report closes the L88C final-internal repair. The repair corrected the
historical fixed-control scope and completed the internal full-video replay
that the original L88C plan registered but had not run. It did not train,
select a new model, refit a threshold, alter the bank, or modify any
production MOT/OVMOT path.

The internal TrackEval result is valid full-video internal validation evidence
on the same V1/V2 scope used by L86, L87-A, and L88. It is not screening,
official-test, or ordinary-MOT evidence.

## 2. Source, code, and frozen strategy

- Branch: `codex/l88c-final-internal-repair-20260906`
- Base commit: `364435610f2c42c3a3ae6bdc404a7419ff1f6cf0`
- Scientific code commit: `e0e6a6bb49c49ae25519e9844c18e2380b092427`
- Current branch head after resource-report commit:
  `ece9fc5549a9210424eb127fb10430c3fed7b7ba`
- GitHub branch: [codex/l88c-final-internal-repair-20260906](https://github.com/LYQ1107/LocateMOT/tree/codex/l88c-final-internal-repair-20260906)
- Fixed selection source: `outputs/l88c/dev/final_selection_attempt2/checkpoint_selection.json`
- Frozen checkpoint: epoch 30, SHA256
  `30f3936bcb60029ceb8fd954af8815b8393656ca50f29f6f04a8e8f0a8d970de`
- Frozen rule: `B`
- Candidate/presence/NULL values: `1.0 / -1.0 / 0.0`
- Fixed manifest SHA256:
  `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`

The selection JSON was checked for `status=complete` and
`selection_frozen_before_fixed_validation=true`; epoch, rule, thresholds, and
checkpoint SHA were read from the JSON rather than guessed.

## 3. Repair A — fixed-control scope

The old evaluator computed the historical original Rule B and pure corrected
gate over all 40 score records, while the L88C corrected-final row used only
the last 24 validation records. That mixed calibration and validation in the
causal controls.

The repair added `_split_fixed_records()`, which sorts and validates
`fixed_eval_order=0..39`, requires the first 16 rows to be calibration and the
last 24 to be validation, and computes historical controls separately for
both partitions. It also asserts six original L88 validation metrics against
the authoritative `validation_frozen_rule` in the original L88 JSON.

The new fixed output is
`outputs/l88c/eval/fixed_semantic_corrected_attempt2/`. It contains 40 ordered
rows, finite scores, complete candidate rows, no deletion/truncation, and a
preselection audit with all forbidden label fields absent. The corrected-final
metrics are identical to attempt1:

| method / scope | recall | precision | FP/frame | pred/positive | hard violation | multi-positive |
|---|---:|---:|---:|---:|---:|---:|
| Original L88 Rule B, validation-only historical control | .193548 | .136364 | 1.5833 | 1.4194 | .846154 | .166667 |
| Pure corrected gate, validation-only historical control | .290323 | .157895 | 2.0000 | 1.8387 | .846154 | .250000 |
| L88C corrected final, candidate + NULL, validation | .354839 | .207547 | 1.7500 | 1.7097 | .846154 | .305556 |

The L88C fixed semantic decision remains `semantic_gate_fail`: recall and
multi-positive recall are below the registered floors. The corrected formula
and checkpoint selection are not a deployable correspondence result.

## 4. Repair B — frozen final internal inference

`tools/l88c_infer_fullvideo.py` now accepts mutually exclusive `--shortlist`
and `--selection` inputs. Selection mode validates the frozen JSON, enables
only Rule B, disallows candidate filtering arguments, records the strategy
source and SHA, and asserts the internal video scope:

- V1: videos `0004`, `0018`, 86 query sequences;
- V2: videos `0016`, `0017`, `0020`, 537 query sequences.

The completed inference is
`outputs/l88c/internal/fullvideo_corrected_final_attempt1/`. Its summary
reports `scope_key=internal`, `full_video=true`, `frozen_selection_mode=true`,
`selected_rule_names=["B"]`, epoch 30, complete candidate rows, and no
screening or official-test labels.

## 5. Final internal TrackEval

The scope-aware wrapper uses the existing `l88_trackeval_matrix.run_dataset()`
only; it does not copy the TrackEval implementation. It evaluates exactly two
datasets and asserts the expected sequence counts. Output:
`outputs/l88c/internal/trackeval_corrected_final_attempt1/`.

| dataset | sequences | HOTA | DetA | AssA | LocA | DetRe | DetPr | AssRe | AssPr | IDF1 | IDR | IDP | IDSW | CLR_FP | CLR_FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Refer-KITTI V1 internal | 86 | 26.0715 | 19.2723 | 35.6429 | 90.9287 | 58.0484 | 22.1956 | 40.0018 | 77.5090 | 21.5878 | 39.0234 | 14.9211 | 1,975 | 60,230 | 11,336 |
| Refer-KITTI V2 internal | 537 | 20.2144 | 12.0672 | 34.1349 | 89.8870 | 29.7221 | 16.7767 | 39.0959 | 75.8496 | 16.4805 | 22.8390 | 12.8915 | 8,877 | 437,725 | 205,854 |

These are internal validation TrackEval metrics, not official Refer-KITTI
test metrics and not HOTA evidence for ordinary MOT/OVMOT.

## 6. Same-scope comparison

The comparison below uses the existing internal full-video artifacts. L87-A
values come from its authoritative retry3 TrackEval artifact; no missing
values were guessed.

| method | V1 HOTA | V1 DetA | V1 AssA | V1 DetRe | V1 DetPr | V2 HOTA | V2 DetA | V2 AssA | V2 DetRe | V2 DetPr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L86 | 29.1663 | 20.5210 | 41.8054 | 69.5047 | 22.3567 | 21.6467 | 13.3584 | 35.2978 | 42.4989 | 16.1970 |
| L87-A | 28.5752 | 19.0948 | 43.1006 | 76.9083 | 20.0996 | 22.1300 | 13.3668 | 36.8421 | 48.9007 | 15.4430 |
| L88 original | 26.0914 | 19.2573 | 35.7070 | 56.5138 | 22.4023 | 20.2386 | 12.0139 | 34.3170 | 33.1874 | 15.7444 |
| L88C corrected final | 26.0715 | 19.2723 | 35.6429 | 58.0484 | 22.1956 | 20.2144 | 12.0672 | 34.1349 | 29.7221 | 16.7767 |

HOTA deltas for L88C corrected final are:

- versus L88 original: V1 `-0.0199` pp, V2 `-0.0243` pp;
- versus L87-A: V1 `-2.5037` pp, V2 `-1.9157` pp;
- versus L86: V1 `-3.0948` pp, V2 `-1.4324` pp.

Both domains remain below L87-A. This is the registered Case C: the
candidate-vs-NULL deployment correction is a real protocol repair, but it does
not explain the broad LoRA failure or recover sequence-level performance.

## 7. Updated interpretation and remaining gaps

The corrected gate improves the accounting and gives lower hard-negative
violation in the fixed semantic result, but its recall-preserving
multi-positive correspondence remains inadequate. The same-scope full-video
metrics show no V1/V2 recovery over L87-A; V2 remains the stronger failure
domain. The earlier A–I diagnosis therefore stands: broad RMOT-only fusion and
decoder LoRA can learn finite score margins, but not reliable recall of every
positive in a target bag. Candidate coverage was separately adequate in L76;
this stage adds no proposal-ceiling evidence.

Still unproven: official screening/test performance, public-benchmark RMOT
comparability, and any ordinary MOT/OVMOT impact. Token/span-to-region and
static/motion alignment remain `UNALIGNED`.

No new structural experiment was launched by this repair. The next structural
choice remains supervisor-controlled and must not be inferred from these
internal HOTA values.

## 8. Boundary and provenance flags

`zero_training=true`; `backward_called=false`; `optimizer_step_called=false`;
`new_checkpoint_written=false`; `lora_update=false`; `sidecar_update=false`;
`screening_gt_used=false`; `official_test_labels_read=false`;
`ordinary_mot_ovmot_touched=false`. TrackEval ran only on the declared internal
V1/V2 validation scope. No screening or official-test labels were read.
