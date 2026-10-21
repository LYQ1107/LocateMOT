# Downstream launch-gate audit — 2026-10-21

The registered downstream entrypoints were read after the formal G2 run had
reached step 12642. The audit is static and does not launch a model.

## Exact G2 dependency

The following owners require all three exact fields from `final_receipt.json`:

- `tools/groundjev/queue_g1_gpu5_after_g2.py`
- `tools/groundjev/queue_g2_clean_bundle_export.py`
- `scripts/groundjev/queue_g2_independent_heldout_eval.py`
- `scripts/groundjev/queue_tao_native_train_shards.py`
- `scripts/groundjev/queue_real_three_task_native.py`

Their contract is:

1. `status == G2_EXACT_COVERAGE_TRAINING_COMPLETE`;
2. `steps == 18_227`;
3. manifest SHA equals
   `f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71`.

The G2 bundle owner additionally requires `adapter_projector_final.pt`; the
held-out and native owners also require their own bundle/manifest/resource
gates. The exact-step check in the bundle owner is represented by
`G2_STEPS = 18_227` rather than the unformatted decimal spelling.

## Independent YesTrack boundary

`scripts/groundjev/queue_yestrack_native_v2.py` intentionally has no G2
training dependency. It is gated by real GPU7 exclusivity and the registered
726-row/390-row native split; it must still refuse CPU mock training. Its
current resource gate remains closed because a foreign compute PID occupies
GPU7.

## Result

No launch-gate defect was found. Since the G2 terminal receipt is absent, no
downstream optimizer or held-out evaluation was started. This audit preserves
the protected G2 run and does not change any checkpoint or owner state.
