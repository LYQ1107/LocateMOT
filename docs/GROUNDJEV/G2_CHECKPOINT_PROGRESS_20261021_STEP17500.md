# G2 Clean checkpoint recovery audit — step 17500

The latest periodic G2 Clean adapter and the full recovery state were loaded
with the registered CPU environment while the protected two-rank trainer
continued running. Both report step `17500`, strict format
`groundanything-g2-projector-qwen3-lora-v2-strict`, world size `2`, and the
registered Clean manifest SHA
`f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71`.

The adapter and recovery state each contain 293 trainable tensors. Optimizer,
scheduler, sampler and RNG state are present; all floating trainable tensors in
the adapter are finite. SHA-256 values are recorded in the paired JSON receipt.
The terminal receipt is not present yet, and the formal training process was
not modified. This is checkpoint/recoverability evidence only and makes no
tracking or scientific gain claim.
