# Stage L5 — Capacity Scaling Ladder

日期：2026-08-11。

| 档位 | d_model | temporal layers | set layers | ffn | 参数量 |
|---|---:|---:|---:|---:|---:|
| Small | 128 | 2 | 2 | 512 | 1.44M |
| Base | 256 | 4 | 4 | 1024 | 7.58M |
| Large | 384 | 6 | 6 | 1536 | NOT_EXECUTED（无正信号放大必要） |

判据：Base 是否优于 Small；train 能否 fit；val 是否随容量提升。

## 结果

- train：Small 与 Base 都能拟合（train_row_acc 0.97+，loss 1.45-1.56）；
- val：两者完全相同（row_acc 0.9814，drift 28.7%/29.3%）；
- 结论：在 11 个视频的小集上，1.44M 已到该目标的可学习上限，
  容量不是瓶颈；Large 不启动。
