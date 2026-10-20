# G2 Clean live event sanity — 2026-10-20

The protected G2 Clean event log was checked read-only through the latest
observed event. Event steps are strictly increasing, all recorded loss,
gradient norm, peak-memory, and learning-rate values are finite, and every
event preserves the registered 8 micro-batches / 16 global samples contract.

- Events checked: 10746
- Latest event: step 10746 / 18227
- Latest loss: 0.253603151
- Latest peak memory: 12.828891 GiB
- Final receipt: absent (training is still live)

This is a training-integrity check only; it makes no tracking or detector
metric claim.
