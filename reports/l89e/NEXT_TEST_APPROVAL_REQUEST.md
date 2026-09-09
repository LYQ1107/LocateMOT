# L89E — next-test approval request

L89E is complete as a zero-training phase/history replay and its fixed
semantic gate failed. No new experiment is authorized by this file.

## Hypothesis

A separately designed absence/volume-calibration study may reduce the
remaining false-positive/no-match volume while preserving the recall and
multi-positive behavior that L89E lost. This is a hypothesis for supervisor
review, not a result.

## Why L89E evidence is insufficient

The phase/history mismatch was corrected, but the frozen epoch-4/S Rule-B
replay still achieved validation recall `0.4516129` and multi-positive recall
`0.3611111`, below the registered floors, even though precision and
FP/frame improved. Inactive false acceptance remained `0.8333333`. Thus the
failure is not resolved by the replay correction and cannot justify a longer
L89E run.

## Exact proposed change

Only if separately approved: define one RMOT-only absence/volume calibration
control around the existing candidate-vs-NULL emission contract, with the
candidate correspondence model, bank, tracker, checkpoint, and phase/history
policy frozen. No new head, temporal retraining, top-k/NMS, or production
integration is proposed here.

## Data and protocol

Use only the already registered fit/dev and fixed 16-calibration/24-validation
units, with calibration-only rule fitting and complete candidate rows. Do not
read screening or official-test labels until a later approved gate.

## Resources and expected outcomes

- GPU: 1 GPU, only after approval.
- Estimated wall-clock: approximately 1–3 hours for a bounded replay, subject
  to the approved implementation contract.
- Expected outcomes: either lower inactive/false-positive acceptance with
  recall retained, or evidence that calibration alone cannot recover
  correspondence.
- Confounds: candidate correspondence and multi-positive ranking are already
  weak; any volume gain that lowers recall is not a fix.
- Why minimum useful: it isolates the remaining emission/absence factor after
  phase consistency, without changing the model, bank, tracker, or ordinary
  MOT/OVMOT paths.

Status: `pending_supervisor_approval`; do not start this study in L89E.
