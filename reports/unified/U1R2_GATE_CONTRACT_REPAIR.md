# LocateMOT-U U1R2 gate and contract repair

U1R2 starts from branch `codex/u0r-u1r-foundation-repair-20261001` at base
SHA `9f9782874183807b6faa034d4b4645fb3847a2a3`. The prior U1-A-v1 JSONs and
reports remain unchanged. U1R2 is the repair stage and does not start U2.

## Contract fixes

- The generic hard gate is now V1 and V2 Top150 Recall@IoU0.50 >= 0.90,
  with at least 500 unique physical objects. IoU0.25 and IoU0.75 remain
  diagnostics; they do not block the gate.
- The expression hard gate is now V1 and V2 Top100 target Recall@IoU0.50 >=
  0.75. Top1/5/10/100/300, AP50, precision@Top10, single/multi-target
  coverage, empty-frame FP, and score-threshold prediction counts remain
  diagnostics.
- Refer-KITTI `labels_with_ids` fields are normalized top-left `x,y,w,h`.
  Both formal gates now retain the source extent and do not clip GT boxes.
  The old U1-A-v1 formal files are preserved as historical artifacts; the
  corrected R2 evaluation is the readiness evidence.
- Queries and images are restricted by the existing legal split. Official
  reserved/test labels were not opened, and no screening GT was used.

## External evaluator contract

`audit_trackeval_contract.py` passed the TempRMOT TrackEval fork and the
registered LocateMOT wrapper on the same arrays. Synthetic perfect gives
HOTA/DetA/AssA = 1.0; empty gives DetRe/HOTA = 0; the deliberate ID switch
keeps DetA = 1.0 while AssA = 0.333333 and HOTA = 0.577350. The cross-evaluator
maximum absolute difference is 0.0. This is an evaluator/format sanity check,
not a model benchmark.

The released TempRMOT checkpoint's declared `refer-kitti-v2.train` pool
contains 0012 (78 frame rows) and 0014 (106 frame rows), the two evaluated
sequences. Its HOTA 94.679 is therefore recorded as
`PIPELINE_SANITY_ONLY_TRAINING_OVERLAP`, not held-out generalization evidence.

## Status semantics

`formal_measurement_is_training=false`, `u1a_training_started=true`,
`u1a_training_completed=true`, `u1b_training_started=true`, and
`u1b_training_completed=true`. U2/U3/U4 were not launched. The final state is
set only after the generic regression, expression coverage, and language
counterfactual gates all pass.

Evidence: `outputs/unified/u1r2/trackeval_contract.json`,
`outputs/unified/u1r2/temprmot_overlap_audit.json`,
`outputs/unified/u1r2/formal_generic_expression_epoch1/result.json`, and
`outputs/unified/u1r2/formal_expression_adaptation_epoch1/result.json`.
