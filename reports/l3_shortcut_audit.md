# Stage L3 — Regime Token Shortcut 审计

日期：2026-08-10。

## 1. 方法

对 U1 的 z_regime（32 维）做三类检查：

1. 域质心距离 vs 域内标准差（dataset separation）；
2. z 与 prediction-side density 的最大相关；
3. 在 z 上训练 domain 分类器（80/20 split，Logistic Regression）。

样本：每域 2,000 个训练样本的 collated batch 前向。

## 2. 结果

| Domain | n | z norm | intra std(mean) | max|corr| with density |
|---|---:|---:|---:|---:|
| dancetrack | 2,000 | 2.19 | 0.261 | 0.294 |
| bdd | 2,000 | 3.24 | 0.258 | 0.470 |
| mot17 | 2,000 | 3.64 | 0.293 | 0.515 |
| mot20 | 2,000 | 0.76 | 0.259 | 0.751 |

- 域质心两两距离：0.76–3.64（最大 MOT17↔BDD 3.64）；
- 域内标准差均值 ≈ 0.26–0.29，远小于域间距离；
- **domain classifier accuracy = 96.6%**（随机 25%）。

## 3. 结论

```text
REGIME_ROUTER_DATASET_SHORTCUT CONFIRMED
```

1. z_regime 主要编码 dataset 身份（regime 特征与 dataset 天然相关：
   BDD 5fps 大 gap、DanceTrack/MOT17/MOT20 密集）；
2. 该 shortcut 未带来任何跨域收益（U1 < U0）；
3. 即使增加 anti-shortcut 正则，pilot 也已证明 regime 条件化在当前
   association 任务上没有可测量正信号；
4. 结论：L3 主方法（latent regime conditioning）**在本次协议下不成立**，
   不得把 dataset-correlated z 当作 regime 泛化证据。
