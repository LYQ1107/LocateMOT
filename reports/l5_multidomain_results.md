# Stage L5 — Multi-Domain Results

日期：2026-08-11。

## 状态

**PARTIAL**：Route A 只做了小集 overfit（BDD/Dance）与官方 AC 评估
（Dance/BDD/MOT17/MOT20 的 U0 baseline + Route A ep40）。由于 Route A
未通过 full-scale 判据，未启动 4 GPU 的正式 multi-domain 训练。

## 官方 AC 数字

| 域 | U0 AssA / IDF1 / IDSW | Route A ep40 AssA / IDF1 / IDSW |
|---|---:|
| Dance | 0.4169 / 0.5694 / 2588 | 0.4182 / 0.5647 / 2558 |
| BDD | 0.2881 / 0.2923 / 11042 | 0.2951 / 0.2954 / 12399 |
| MOT17 | 0.6050 / 0.5825 / 259 | 0.5914 / 0.5834 / 279 |
| MOT20 | 0.2950 / 0.4012 / 2406 | 0.2763 / 0.3800 / 2588 |

一个 checkpoint（Route A Small ep40）覆盖四个域，无 dataset-specific
head/router/threshold；但指标未在全部域保持，不满足通过条件。
