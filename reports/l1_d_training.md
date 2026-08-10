# Stage L1-D Training Report

日期：2026-08-10。模型：Evidence-Gated Set-Level Residual Association
（EGRA），`locatemot/models/l1d_association.py`。

## 1. 训练数据

训练集 = 固定候选 manifest + LocateAnything cache 上，用校准后的
L1D 基座（0.4 IoU + 0.2 PBD + 0.4 Kalman-motion，thr=0.25）离线模拟
共享 AC shell 得到的真实在线状态（不是 GT 完美历史）：

| 数据域 | 帧数 | 有监督事件 | base 身份正确率 |
|---|---:|---:|---:|
| DanceTrack calibration | 8,016 | 60,378 | 0.462 |
| BDD100K train | 5,073 | 21,107 | 0.571 |
| MOT17 train | 237 | 4,596 | 0.837 |
| MOT20 train | 79 | 2,384 | 0.725 |
| 合计 | 13,405 | 88,465 | — |

采样：50% 均匀 + 50% 按“base 行错误率”加权（hard 帧），保留 easy 样本。

## 2. 模型与训练

- 参数：0.49M（可训练）；d_model=128，2 层 transformer，4 head。
- Loss：row+col assignment CE（主）+ reliability BCE（pos_weight=9，
  λ=0.3）+ base-correct 行的 |delta| 保留正则（λ=0.1）。
- 优化：AdamW lr=3e-4，OneCycle（warmup 300），batch=64，
  seed=20260806。
- 步数：8,360（40 epochs），单卡 A100（GPU 2），~255 秒。
- 训练 loss：3.96 → 3.21（row CE 3.4 → 2.5–2.9，主要残差来自 55%
  base 已身份错误且无法用有界残差恢复的行）。

## 3. 校准记录

- base 权重网格（calibration 只调 base，val 只评估）：
  7 组权重 × 3 个阈值；最优 L1DK (0.4,0.2,0.4,t0.25)：
  calibration AssA 0.4241 / IDF1 0.5713 / IDSW 512。
- delta scale 在 calibration 比较 {0.3, 0.6}：0.3 更优
  （AssA 0.4428 vs 0.4257；IDSW 532 vs 574），选 0.3。

