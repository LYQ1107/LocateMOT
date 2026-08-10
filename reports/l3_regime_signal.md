# Stage L3 — Latent Regime 信号审计

日期：2026-08-10。

## 1. 方法

用 prediction-side 状态特征（候选数、IoU 歧义、PBD 歧义、运动代理、
尺寸、时间间隔、语义多样性）把帧切成 regime 桶；对每个桶，用
保存的 AC tracker 输出计算 H=16 windowed AssA（与官方 TrackEval
公式一致），比较 IoU/Motion/L1DK/EGRA 的偏好。

数据：DanceTrack val 3,000 窗口、MOT17 30、MOT20 20、BDD 693。
（MOT17/MOT20 窗口少，结论以 DanceTrack/BDD + MOT17 大差距桶为准。）

## 2. 关键结果（每轴 marginal：lo/hi 桶内平均 windowed AssA）

### DanceTrack val

| Regime | best | AssA | L1DK | C1 | EGRA |
|---|---:|---:|---:|---:|---:|
| n_cand=lo | EGRA | 0.9592 | 0.9590 | 0.9558 | 0.9592 |
| n_cand=hi | L1DK | 0.9474 | 0.9474 | 0.9466 | 0.9472 |
| iou_amb=hi | EGRA | 0.9361 | 0.9358 | 0.9331 | 0.9361 |
| pbd_amb=hi | EGRA | 0.9696 | 0.9680 | 0.9660 | 0.9696 |
| motion=hi | EGRA | 0.9405 | 0.9393 | 0.9371 | 0.9405 |

DanceTrack 上差异小（0.1–0.5pp），L1DK/EGRA 稳定占优。

### MOT17（30 窗口，小样本但有强反差）

| Regime | best | AssA | L1DK | C1 | EGRA |
|---|---:|---:|---:|---:|---:|
| motion=hi | **C1** | 0.6400 | 0.6296 | 0.6400 | 0.6215 |
| motion=lo | L1DK | 0.8526 | 0.8526 | 0.8138 | 0.8461 |
| iou_amb=hi | **C1** | 0.6519 | 0.5853 | 0.6519 | 0.5388 |
| n_cand=lo | EGRA | 0.6182 | 0.5726 | 0.6081 | 0.6182 |

MOT17 的 regime 反差最明显：高运动/高 IoU 歧义时 Motion 规则胜
（最多 +6.7pp），低运动时 L1DK 胜，低密度时 EGRA 胜。

### BDD（693 窗口）

| Regime | best | AssA | L1DK | C1 | EGRA |
|---|---:|---:|---:|---:|---:|
| n_cand=hi | **C0** | 0.5572 | 0.5534 | 0.5562 | 0.5428 |
| n_cand=lo | L1DK | 0.5396 | 0.5396 | 0.5199 | 0.5351 |
| pbd_amb=hi | EGRA | 0.5152 | 0.5141 | 0.4995 | 0.5152 |
| size=hi | L1DK | 0.6787 | 0.6787 | 0.6646 | 0.6682 |
| size=lo | L1DK | 0.4422 | 0.4422 | 0.4356 | 0.4376 |

BDD 上 L1DK 多数桶最优，但高密度桶 C0（纯 IoU）超过 L1DK，
高 PBD 歧义桶 EGRA 略优。

## 3. 结论

**REGIME_SPECIALIZATION_SIGNAL_SUPPORTED（弱到中等）**：

1. 不同 regime 桶的最优方法确实不同（MOT17 反差最大，BDD/DanceTrack
   差异较小）；
2. 不存在一个固定规则在所有 regime 桶都严格占优；
3. 但 L1DK 是强共享先验，负迁移绝对值不大 → regime 条件化必须
   在“保持强先验 + 按 regime 微调证据权重”的设定下验证，而不是
   推翻基座。

风险：MOT17/MOT20 窗口样本少，正式结论以 DanceTrack/BDD 为主；
U1 pilot 的收益预期应设为“消除 MOT17 高运动桶的 0.5–6pp 差距 +
BDD 高密度桶差距”，而不是全域大幅提升。
