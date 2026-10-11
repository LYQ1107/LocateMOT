# U2 native batch admissibility measurement

Original scientific commit: `24be26a0b2f72843d26df4530b021c8ba3b9ec1d`.
Source tree: `b2af534a13a540c842e3f77c6750073b1c7d00ac`.
Parent scientific commit: `41027403a67b0aefb050c4747fead05b4d555c31`.

The incremental `u2_source_history_24be26a.bundle` retains the original
scientific commit and requires that parent; earlier bundles preserve the
chain back to verified U1R2.

Native FP32 batch2 on two actual fixed MOT17 frames changes Top150 boxes and
scores relative to singleton inference. Sorted proposal alignment changes;
same-rank coordinate differences are not matched-box localization errors.
The actual fusion scalar-max flags are false, and second-image perturbation
leaves the first batched output bit-exact in this limited probe.

Reject native image batching for the current cache. Retain native singleton
inference plus previously validated fixed-text reuse. No full-vocabulary
batching, source/precision changes or live audit restarts are performed.
Full tracking experiments and selected-artifact/U3 evidence remain pending.

No model weights, datasets or private cloud/browser/administration artifacts
are published.
