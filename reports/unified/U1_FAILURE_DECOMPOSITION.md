# U1 failure decomposition and stop record

The registered generic adaptation schedule was exhausted at epochs 1, 2, 4,
and 6 using the valid BF16, frozen-stage, eight-GPU DDP run. The epoch-6
formal result is V1 Top150 Recall@IoU0.50 `0.8938` and V2 `0.8815`, below the
required `0.90` for both datasets. The foundation expression result is also
below its `0.75` target, but expression adaptation is downstream of the
generic gate and was not run.

The valid checkpoints and complete diagnostic metrics are recorded in
`outputs/unified/u1/formal_generic_adaptation/epoch{1,2,4,6}.json`; the
interpretation and size/domain decomposition are in
`reports/unified/U1_GENERIC_ADAPTATION.md`.

The first non-DDP launch and the exploratory FP32/stage0-only run were
excluded from evidence. A later batch-2/two-step-accumulation attempt hit a
PyTorch reducer assertion; the final batch-2/no-accumulation run passed a
multi-step DDP smoke and completed all six epochs. Its logs show occasional
`grad_norm: nan` telemetry with finite losses, which is retained as a reported
anomaly.

The stop is therefore:

```text
LOCATEMOT_U_BLOCKED_GENERIC_GATE_AFTER_MAX_ADAPTATION
STOPPED_PENDING_SUPERVISOR_REVIEW
```

`official_test_labels_read=false`; `next architecture stage launched=false`.
