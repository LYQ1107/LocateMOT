# U2 exact joint execution source snapshot

Original local scientific commit: `355c152634e3cef7f3fcd511d982892692514034`.
Source tree: `a22172dda939b81f27a0d2996927c6ed780daba6`.
Parent scientific commit: `3d2a9ec790aea938570fc176476be9c19234ad70`.

The incremental `u2_source_history_355c152.bundle` preserves the original
commit and requires the parent above. Earlier published source-history bundles
provide the parent chain back to the verified U1R2 base.

This stage implements the exact gated 4,000-step joint MOT/OVMOT trainer,
all-five-dataset ordinary and complete TAO/TETA evaluation, and a blocking
owner that waits for recomputed eligible MOT-only selection. Actual eight-GPU
BF16 synthetic proofs cover strict initialization, optimizer groups, alternating
tasks, scheduled-teacher RNG and optimizer restoration with a bit-exact next
update, all official readers, and completed-video evaluation resume. Inclusive
1/2/3-point gates compare recorded decimal scores without an added tolerance.

These proofs are integration fixtures, not benchmark tracking measurements.
Full perception audit, selected candidate caches, simple and learned tracking
measurements remain pending. Joint optimization is held by original eligibility
gates. U3 readiness is unproven. The shared U1 foundation and all references,
seeds, training budgets, candidate limits and source scopes are preserved.

No private cloud/browser/administration files or model weights are published.
