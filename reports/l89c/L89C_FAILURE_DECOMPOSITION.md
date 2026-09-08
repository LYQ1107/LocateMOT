# L89C failure decomposition

## Validity

The L89C protocol correction is valid. The original L89 score/deployment path
trained and audited `candidate_energy` against `null_logit`, while old
deployment used `presence_logit - null_logit`. L89C uses the registered shared
`corrected_emission_mask` and shared metric implementation. The fixed replay
has complete 40-unit order, calibration-before-validation label boundaries,
finite scores, and no candidate deletion/truncation. The first GPU attempt
failed only because GPU0 was out of memory; its traceback is preserved and an
identical CPU retry completed.

## Evidence failure

The corrected fixed validation still fails the simultaneous gate. The first
actionable scientific failure is **absence/volume calibration of the frozen
correspondence output**: inactive false acceptance is `1.0`, FP/frame is
`11.375`, and predictions/positive is `9.6452`, despite hard violation
improving to `0.692308` and recall/multi-positive floors remaining above the
gate. This is not a claim that candidate coverage is missing, nor a claim
that the corrected equation failed numerically. It is also not evidence that
the historical buggy result was deployable.

The corrected internal TrackEval confirms the operational consequence:
HOTA is only `2.244718%` on V1 and `0.583059%` on V2, versus the historical
L87-A reference `28.5752%` and `22.1300%`. No screening or official-test
evidence exists.

## Single next action

Stop L89/L89C and request supervisor review for **one new RMOT branch that
explicitly addresses learned absence/volume calibration together with
candidate correspondence**. Do not extend L89 training, alter its threshold,
add top-k/NMS/NULL filtering, or choose another checkpoint. Until that branch
is separately authorized and passes a fixed semantic gate, no screening or
production RMOT claim is permitted.

Status: `STOPPED_PENDING_SUPERVISOR_REVIEW`.
