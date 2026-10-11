# LocateMOT-U U1R2 expression adaptation

Expression adaptation starts from the selected generic checkpoint, not from
the original foundation. The legal-fit expression stream contains one whole
sentence positive span with all visible referred boxes, deterministic empty
frames at approximately 20%, and the preserved V2 missing-target policy. The
mix has 2,545 generic rows, 372 V1 rows, and 5,115 V2 rows (5,000 sampled mix
rows). The 157 V2 `SOURCE_TARGET_ID_MISSING` references are not converted to
negative boxes or silently dropped.

The first FP32/TF32 epoch used decoder/head `5e-5`, multimodal encoder `3e-5`,
Swin stages 2/3 `5e-6`, BERT `0`, frozen stages 0/1, and one shared checkpoint.
It completed 313 iterations with finite logged gradients and no BERT
unfreeze. The selected shared checkpoint is
`outputs/unified/u1r2/expression_adaptation_epoch1/epoch_1.pth` (SHA-256
`4efc35ce70be5cddf2c4d9a0f91bc4edef3e9ec5277a819211ba4ac42e0d7d26`).

## Expression gate

The formal sample has 7,160 query-frame rows and 7,199 valid target units with
zero missing units in calibration/validation. Top100 Recall@IoU0.50 is V1
`1.000000` and V2 `0.961367`, so the coverage gate passes. Diagnostics:

| scope | Top1 | Top5 | Top10 | Top300 | AP50 | precision@Top10 |
|---|---:|---:|---:|---:|---:|---:|
| V1 | 0.343195 | 0.810651 | 0.960059 | 1.000000 | 0.751406 | 0.187572 |
| V2 | 0.312126 | 0.715468 | 0.897900 | 0.970106 | 0.667921 | 0.169251 |
| overall | 0.315044 | 0.724406 | 0.903737 | 0.972913 | 0.675675 | 0.170953 |

The overall row's exact multi-target and absence diagnostics are
single-target Top10 recall `0.931177`, multi-target Top10 recall `0.889910`,
multi-target exact set coverage `0.760091`, empty-frame FP/query `5.540029`,
and predictions with score >= 0.05 per query `7.050978`. The per-dataset
values and full JSON are the source of truth.

The generic regression on this same checkpoint passes: V1 Top150 IoU0.50
`0.986592`, V2 `0.958739` (overall `0.971813`). No additional expression
epoch was needed.

Evidence: `outputs/unified/u1r2/expression_train_manifest.json`,
`outputs/unified/u1r2/formal_expression_adaptation_epoch1/result.json`, and
`outputs/unified/u1r2/formal_generic_expression_epoch1/result.json`.
