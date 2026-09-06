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

## Final-internal repair addendum — 2026-09-06

The earlier version of this report ended after corrected dev TrackEval and
used a mixed 40-unit historical control in its causal table. The final
internal repair corrected that scope to 16 calibration / 24 validation and
asserted the six original L88 validation metrics against the authoritative
L88 JSON. The corrected fixed output is
`outputs/l88c/eval/fixed_semantic_corrected_attempt2/`; its semantic decision
is still `semantic_gate_fail`, with corrected-final validation
recall `.354839`, precision `.207547`, FP/frame `1.750`, pred/positive
`1.7097`, hard `.846154`, and multi-positive recall `.305556`.

The registered final internal validation was then completed for the frozen
epoch30 / Rule B strategy (`1.0 / -1.0 / 0.0`) on exactly V1 86 and V2 537
sequences. The TrackEval output is
`outputs/l88c/internal/trackeval_corrected_final_attempt1/`. HOTA is
`26.0715` V1 and `20.2144` V2, below L87-A `28.5752/22.1300` in both domains.
These are internal validation HOTA values, not screening or official-test
results. The detailed same-scope table and provenance are in
`reports/l88c/L88C_FINAL_INTERNAL_VALIDATION.md`.

Code repair commit: `e0e6a6b`; current branch head after the separate
recoverable storage report is `ece9fc5`. No training, checkpoint update,
screening/official-test read, or ordinary MOT/OVMOT change occurred.
