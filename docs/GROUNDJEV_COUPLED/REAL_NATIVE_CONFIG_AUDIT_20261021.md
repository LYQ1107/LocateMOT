# Native joint-training launch configuration audit — 2026-10-21

The post-G2 native three-task queue had a launch-blocking configuration bug.
`queue_real_three_task_native.py` added `native_manifest_rows` to the generated
OneJev config as an audit field, while the pinned `train.sft` parser rejects
unknown keys. The queue would therefore fail before model loading when the
native manifest gate opened.

The audit-only field was removed from the generated config. The row count is
already recorded in `owner_state.json` and is checked against the terminal
`done.examples` value, so removing it does not weaken the manifest contract.

The audit also fixes the interpretation of `max_rows`: it is the maximum rows
in one length-bucket micro-batch. It is not an epoch or training-file limit.
With `max_rows=1` and `accum_rows=16`, the trainer accumulates at least sixteen
rows across micro-batches before an optimizer update. The conservative value is
kept because it bounds multimodal per-forward memory; no throughput claim is
made until a real native launch supplies matched workload measurements.

Validation passed with the pinned OneJev Python 3.10 environment:

- queue `py_compile`;
- generated-config keys normalized through `common.CONFIG_ALIASES` and checked
  against `train.sft.build_parser()` (zero unknown keys);
- coupled package/tools/tests `compileall`;
- coupled package import smoke.

No G2 process, checkpoint, optimizer, scheduler, sampler, or RNG state was
modified. The native joint owner remains gated on the exact G2 terminal
receipt, native TAO join, bundle/held-out receipts, and the four-GPU resource
probe.
