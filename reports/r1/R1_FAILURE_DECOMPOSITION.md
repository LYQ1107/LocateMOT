# R1 failure decomposition

## Final status

`R1_NO_BREAKTHROUGH` / `STOPPED_PENDING_SUPERVISOR_REVIEW`.

R1 passed its implementation, anchor, training, reload, key-preservation and
legal-development replay contracts. The failure is scientific and
attributional, not an unexecuted-smoke or missing-gradient failure.

## First actionable root cause

The aligned residual/history sidecar does not provide a stable held-out
expression-to-track correspondence improvement under the frozen emission
contract. The selected legal-development checkpoint regressed V1 HOTA from
the L89E anchor's 28.7628% to 24.2345%, while its V2 increase was only 1.8157
points. On the independent fixed 40-unit diagnostic, R1 and the frozen anchor
have the same candidate-only hard violation (`0.7727`), but R1 falls to
recall `0.1000`, multi-positive recall `0.1818`, and empty rate `0.7250`.
The frozen Rule-B output emits zero rows for both methods on this slice.

Thus the actionable bottleneck is not bank key integrity, missing optimizer
steps, or a recoverable checkpoint-load error. It is the combination of
correspondence residuals with an incompatible score/volume behavior and no
stable cross-domain identity-semantic gain. The lower candidate-only false
acceptance is accompanied by output collapse, so it is not a fix.

## Evidence that rules out alternative interpretations

- `2,250/2,250` formal steps were finite and had nonzero gradients.
- Eight checkpoints strictly reloaded; the anchor was frozen and no forbidden
  production source changed.
- Six legal-dev prediction-only replays completed before GT was opened, and
  all 8 TrackEval results were produced from the legal scope.
- The fixed diagnostic retained all 2,435 L69 rows and all 40 unit keys; no
  top-k, NMS, candidate deletion, or validation threshold repair was used.
- V1 and V2 were evaluated separately; the V2 result was not used to hide the
  V1 regression.

## Decision and single next action

Do not extend R1, sweep thresholds/layers/learning rates, add a NULL filter,
alter the bank, or merge the sidecar into production. Do not read screening or
official-test labels. The single next action is supervisor review and explicit
authorization of one separately defined follow-up branch; no automatic R1
continuation is launched.

The ordinary MOT/OVMOT/TAO paths, UIDM, frozen L69 bank, old checkpoints and
TrackEval source remain untouched. R1 has no claim of final ordinary-RMOT
success.
