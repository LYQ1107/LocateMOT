# U1-A controlled generic detection adaptation

Status: **BLOCKED after the registered six-epoch maximum**. The final valid
run did not reach the generic Top150 Recall@IoU0.50 requirement of `0.90` on
both legal datasets, so U1-B expression adaptation and U2 were not started.

The legal split remained frozen at
`8ee806048ccbb227919f27edf68cf0bebc1fa50f824a8108878fc5f39fc6e1c0`. The
training stream contains 3,606 legal fit image rows from V1/V2, with the
fixed mapping `label 0 -> car` in `outputs/unified/u1/category_mapping.json`.
Calibration, validation, reserved, and official-test labels were not used for
training. The query-independent vocabulary and official MMDetection
inference contract were unchanged.

The final valid run started from the foundation checkpoint and used
`frozen_stages=2`, which freezes the Swin patch embedding and stages 0 and 1
in the MMDetection implementation. BERT was frozen (`lr_mult=0`); the
multimodal/decoder/head path used AdamW at `1e-4`, and trainable Swin stages 2
and 3 used the `0.1` backbone multiplier (`1e-5`). Weight decay was `1e-4`
and gradient clipping was `0.1`. Training used BF16 AMP, the base model's
gradient checkpointing, explicit PyTorch DDP (`--launcher pytorch`) on eight
A800 GPUs, batch size 2 per GPU, and no accumulation (effective global batch
16). The batch-2 run used about 5,945 MB per GPU at the logged peak. A small
BF16-only cast around MMCV's unsupported BF16 deformable-attention kernel was
kept in the launcher; all other model/data computation remained under the
registered BF16 AMP path.

## Formal progression

The first row is the unadapted foundation result. The remaining rows are the
formal full legal calibration/validation evaluations with 4,201 images,
33,526 physical GT units, and 5,992 unique physical objects. Each cell is
`Recall@IoU0.25 / Recall@IoU0.50 / Recall@IoU0.75` at Top150.

| checkpoint | overall | V1 | V2 |
|---|---:|---:|---:|
| foundation | 0.5113 / 0.0732 / 0.0054 | 0.5309 / 0.0803 / 0.0045 | 0.4940 / 0.0669 / 0.0062 |
| valid adaptation epoch 1 | 0.9903 / 0.8615 / 0.4827 | 0.9923 / 0.8729 / 0.5305 | 0.9886 / 0.8515 / 0.4404 |
| valid adaptation epoch 2 | 0.9922 / 0.8816 / 0.6183 | 0.9935 / 0.8873 / 0.6611 | 0.9911 / 0.8766 / 0.5805 |
| valid adaptation epoch 4 | 0.9934 / 0.8891 / 0.6714 | 0.9943 / 0.8945 / 0.7110 | 0.9926 / 0.8843 / 0.6364 |
| valid adaptation epoch 6 | 0.9923 / 0.8873 / 0.6634 | 0.9929 / 0.8938 / 0.7215 | 0.9917 / 0.8815 / 0.6119 |

The final valid epoch-6 checkpoint is
`a1ff39f9c0bd59a5fd2b14968a528fc11e882b8fa6a3e85a9b64253a1cdc5ccc`.
The four complete formal outputs are
`outputs/unified/u1/formal_generic_adaptation/epoch{1,2,4,6}.json`.

The epoch-6 size strata at Top150 are:

| size | IoU0.25 | IoU0.50 | IoU0.75 |
|---|---:|---:|---:|
| small | 0.9766 | 0.9136 | 0.6001 |
| medium | 0.9978 | 0.9445 | 0.7438 |
| large | 0.9997 | 0.7085 | 0.5479 |

The first eight-process command was stopped because it omitted
`--launcher pytorch` and therefore did not run DDP; it is not evidence. A
subsequent exploratory FP32 run used `frozen_stages=1`, which freezes only
stage 0, so its metrics are also excluded from the table. A batch-2 attempt
with two-step accumulation passed a one-step smoke but hit PyTorch's
`expect_autograd_hooks_` reducer assertion in the full run; it produced no
checkpoint used here. The final run uses batch 2 with accumulation 1 after a
two-step DDP regression smoke passed.

Training logs contain occasional `grad_norm: nan` telemetry while the loss
values stayed finite; the run completed all six epochs, checkpoints load, and
the formal evaluator produced finite predictions for every shard. This is
recorded as a training telemetry anomaly rather than hidden or repaired
post-hoc.

## Failure decomposition

- **Object discovery versus localization:** IoU0.25 recall reached about
  `0.993` in both domains, while epoch-6 IoU0.50 stopped at `0.8938/0.8815`
  (V1/V2) and IoU0.75 at `0.7215/0.6119`. The residual is primarily overlap
  precision/localization rather than failure to discover road objects.
- **V1 versus V2:** V2 trails V1 by 1.23 percentage points at IoU0.50 and
  10.97 points at IoU0.75 on the final valid checkpoint. The weaker V2 side
  remains the main held-out limitation.
- **Object size:** epoch-6 IoU0.50 recall is `0.9136` small, `0.9445`
  medium, and `0.7085` large; large-object localization is the clearest
  failure. At IoU0.75 the values are `0.6001/0.7438/0.5479`.
- **Classification versus box regression:** the fixed `label 0 -> car`
  stream and fixed multi-class prompt were unchanged. Near-0.99 IoU0.25
  recall does not support a broad class-discovery failure; the current
  measurements do not isolate text alignment from box regression further.
- **Resolution:** every checkpoint used the registered
  `FixScaleResize(800,1333)` contract. No resolution sweep or threshold
  change was used.

The foundation expression gate remains below its target, but expression
adaptation is contractually downstream of the generic gate and was not run.
No official-test labels were read, the 157 V2
`SOURCE_TARGET_ID_MISSING` references remain preserved, and no universal
tracking core was launched. The next step is supervisor review of this stop
record.
