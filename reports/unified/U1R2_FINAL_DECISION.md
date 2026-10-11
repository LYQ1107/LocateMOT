# LocateMOT-U U1R2 final decision

Branch: `codex/u0r-u1r-foundation-repair-20261001`.
Base SHA before U1R2 changes: `9f9782874183807b6faa034d4b4645fb3847a2a3`.
The final shared checkpoint is
`outputs/unified/u1r2/expression_adaptation_epoch1/epoch_1.pth`, SHA-256
`4efc35ce70be5cddf2c4d9a0f91bc4edef3e9ec5277a819211ba4ac42e0d7d26`.

## Hard gates

- Generic proposal gate: PASS. On the final shared checkpoint, V1/V2
  Top150 Recall@IoU0.50 are `0.986592/0.958739`; unique physical objects are
  5,992. IoU0.25/0.75 remain reported diagnostics.
- Expression candidate coverage gate: PASS. V1/V2 Top100 target
  Recall@IoU0.50 are `1.000000/0.961367`.
- Language conditioning: PASS. In 500 legal counterfactual pairs, true
  versus shuffled Top10 recall is `0.959333/0.871133` and AP50 is
  `0.736901/0.491159`.
- FP32 gradient audit: PASS with zero non-finite and zero no-gradient
  trainable parameters.
- TrackEval contract: PASS; cross-evaluator maximum difference `0.0`.

The selected shared checkpoint preserves generic capability after expression
fine-tuning. V2's 157 source-target-ID-missing references remain explicit in
the manifests and were not silently dropped. `official_test_labels_read=false`
and `screening_gt_used=false` for all U1R2 outputs. The external TempRMOT
94.679 HOTA result is marked pipeline-sanity-only because its evaluated 0012
and 0014 sequences are in its declared training pool.

Decision:

```text
LOCATEMOT_U_READY_FOR_U2
STOPPED_PENDING_SUPERVISOR_REVIEW
```

No U2 Universal Track Core, U3 RMOT, or U4 prompt work was launched.
