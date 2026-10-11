# LocateMOT-U U1R2 gradient audit

The old valid BF16 log contains three `grad_norm: nan` records, all with finite
loss and learning rate `1.0000e-04`:

| epoch | iter | loss |
|---:|---:|---:|
| 1 | 50 | 21.4998 |
| 3 | 100 | 7.2971 |
| 6 | 100 | 5.9542 |

There are zero non-finite-loss log lines. The anomaly is retained as evidence;
it is not filtered or reclassified as a successful finite-gradient run.

A fresh one-step FP32 backward audit from the frozen epoch-4 checkpoint with
the class-aware manifest loaded all checkpoint keys consistently. It found 981
trainable parameters, total loss `40.27210998535156`, zero non-finite
gradients, and zero parameters without gradients on CUDA. The R2 refinement and
expression runs therefore use FP32 with TF32 allowed, no AMP/BF16, Swin stages
0/1 frozen, and BERT frozen. Their training logs and checkpoints completed
with finite logged losses and gradients.

Evidence: `outputs/unified/u1r2/bf16_nan_log_audit.json` and
`outputs/unified/u1r2/gradient_audit.json`.
