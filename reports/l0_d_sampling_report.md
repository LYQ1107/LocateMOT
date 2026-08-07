# Stage L0-D Sampling Report

生成时间：2026-08-07

## 训练数据

- 来源：L0-C frozen pair manifest，train split 6858 pairs。
- 全部使用预计算张量 `outputs/l0_d/precomputed/train_full.pt`（float32，9.6 GB；含 PBD box-end 特征）。
- seed=20260806 固定；采样使用 `WeightedRandomSampler(replacement=True)`，num_samples = 2 × 6858。

## 目标数分布与权重

| bucket | 实际样本数 | 实际占比 | 目标占比 | 单样本权重 |
|---|---:|---:|---:|---:|
| 1 | 3714 | 54.2% | 25% | 0.462 |
| 2–4 | 2943 | 42.9% | 45% | 1.049 |
| 5–8 | 201 | 2.9% | 30% | 6.0（封顶） |

5–8 权重按用户要求封顶为 6.0，只用真实存在的 201 个样本，不做复制增强；实际期望占比约 20–28%（受 hard 乘子影响）。

## Hard-competition 上采样

- 定义来源：`configs/l0_d_hard_subset.json`（冻结，预测端信息：top1/top2 IoU margin<0.10、PBD box-end cos margin<0.05、密度≥6、ref≥5、共享候选 IoU≥0.30）。
- train 中 hard=3421/6858（49.9%）。
- hard 样本额外权重 ×2.0。

## 说明

- 本报告描述的是 B5/B6 训练采样；held-out 评估不使用任何采样权重。
- 不复制少量 5–8 样本；不按最终结果调整阈值。
