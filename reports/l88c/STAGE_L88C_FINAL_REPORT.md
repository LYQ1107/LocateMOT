# Stage L88C final report

## Decision

**`STOPPED_PENDING_SUPERVISOR_REVIEW` — corrected candidate-vs-NULL replay
completed; semantic gate failed.**

L88C did not train or update a model. It corrected and replayed the existing
L88 strategy using the exact frozen L88 checkpoints and registered dev/fixed
unit protocols.

## Frozen result

The dev TrackEval selection froze epoch 30, Rule B, with candidate threshold
`1.0`, presence threshold `-1.0`, and NULL margin `0.0` before fixed validation.
Internal dev full-video TrackEval was V1 HOTA `.296615`, DetA `.216472`, AssA
`.408302`; V2 HOTA `.265682`, DetA `.156622`, AssA `.451938`. These are
internal dev values only.

The fixed semantic result is:

| control | recall | precision | FP/frame | pred/positive | hard | multi |
|---|---:|---:|---:|---:|---:|---:|
| immutable L29 | .733333 | .083019 | 10.125 | 8.8333 | .916667 | .819444 |
| L88C corrected final | .354839 | .207547 | 1.750 | 1.7097 | .846154 | .305556 |

The hard-negative reduction is not enough: recall and multi-positive recall
collapse relative to the registered gate. Candidate-only recall is higher
(`.419355`) but still fails; the final corrected candidate-vs-NULL rule is
not a deployable output.

## Root cause and next action

The primary root cause is insufficient recall-preserving query–candidate
correspondence for multi-positive target bags, especially on V2. The offline
A–I analysis is in `L88C_DIAGNOSIS.md` and its JSON output. The only proposed
next action is the single supervisor approval request in
`NEXT_TEST_APPROVAL_REQUEST.md`; no new test has been launched.

## Evidence classification and isolation

- Fit/training: historical L88 fit only; L88C added no training.
- Calibration/selection: corrected dev refit and internal dev TrackEval,
  frozen before fixed validation.
- Validation: one corrected 16-calibration/24-validation replay.
- Oracle: none added in L88C.
- Screening/official test: not run; labels not read.
- TrackEval: internal dev only, not official screening.
- Ordinary MOT/OVMOT/TAO: unchanged and not evaluated.
- Token/span-to-region and static/motion alignment: `UNALIGNED`.

## Reproducibility paths

- Corrected dev refit: `outputs/l88c/dev/corrected_reselect_attempt2/`
- Full-video replay: `outputs/l88c/dev/fullvideo_corrected_attempt1/`
- Corrected internal TrackEval: `outputs/l88c/dev/trackeval_matrix_attempt2/`
- Frozen selection: `outputs/l88c/dev/final_selection_attempt2/`
- Fixed semantic gate: `outputs/l88c/eval/fixed_semantic_corrected_attempt1/`
- Root-cause JSON: `outputs/l88c/diagnosis/root_cause_attempt2/`
- Code branch commit: `e8f5f62`
