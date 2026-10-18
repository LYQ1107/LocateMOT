# G2 clean full-coverage V2 report

## Contract

`G2_CLEAN_FULL_COVERAGE_V2` starts from the original
`GroundAnything-DLM` weights. It does not inherit the contaminated G2 adapter.
The native objective remains the released `0.5 MDM + 0.5 causal` loss, with
the projector and Qwen3 q/k/v/o LoRA trainable and the vision tower and
original language weights frozen.

The clean manifest has 583,251 rows. The exact-once source-interleaved sampler
shuffles each source, emits every row once, and adds only the minimum tail for
distributed rectangular batches. For global batch 32, the first coverage epoch
has 583,264 scheduled slots: 583,251 unique rows and 13 explicit padding rows.
The startup audit covers all 9 sources, 1,033 video/sequence keys, and 168,065
image keys.

The formal launch uses physical GPU3/GPU5, world size 2, micro-batch 2,
gradient accumulation 8, and effective global batch 32. The token envelope is
16,384 padded input tokens per device; every batch is measured and rejected if
it exceeds the envelope. DDP uses `no_sync()` on non-final accumulation
micro-batches. The planned one-pass budget is
`ceil(583251 / 32) = 18,227` optimizer updates, with checkpoint interval 500.

## Verification gates

The 2-update BS2 smoke passed on GPU3/GPU5 with finite loss and gradients and a
17.43 GiB peak. Its step-1 checkpoint was resumed in a fresh two-rank process
under the identical scheduler/batch contract; the replayed step-2 loss and
effective token count matched the uninterrupted run exactly. Checkpoints carry
base-model and manifest SHA, LoRA names/rank/alpha/dropout, exact trainable
key/shape/dtype signatures, optimizer/scheduler contract, per-rank RNG
sidecars, sampler cursor, and coverage statistics. Legacy checkpoints are
refused by strict resume.

The BS8 probe is retained as a failed resource gate, not silently retried. The
old contaminated 1,200-step G2 checkpoint and all prior receipts remain
untouched.

## Live run

The current formal owner is:

`/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/formal_clean_coverage_v2_final/`

Its `launch_contract.json`, `trainer_contract.json`, `launch.log`,
`training_events.jsonl`, and atomic checkpoints are the authoritative runtime
record. At publication time the owner has passed initialization and is running;
no detector, MOT, RMOT, or OVMOT quality result is claimed until its terminal
receipt and independent evaluation are complete.

## Deployment handoff registered 2026-10-19

The CPU-only owner tools/groundjev/queue_g2_clean_bundle_export.py is registered against the exact G2 terminal receipt and the GPU5 G1 queue receipt. It waits internally, then invokes g2/export_g2_checkpoint.py to create the immutable models/GroundAnything-DLM-G2-CLEAN-V2 bundle with the 569 native tensors, the clean manifest SHA, and an export receipt. It refuses to overwrite an existing bundle and records explicit dependency failure states. This owner is running under outputs/groundjev/ground_adaptation/g2_clean_bundle_export_v1/owner_state.json; the bundle and any G2 quality claim remain pending until the terminal receipts exist.
