# Stage L3 — U0：Naive Shared 学习型关联核心

日期：2026-08-10。

## 1. 定义

U0 = L1DAssociator（set-level transformer + 有界残差，与 L1-D EGRA
同架构），在 DanceTrack calibration + BDD(11 类) + MOT17 + MOT20
联合数据上训练 30 epochs（~6,300 步，batch 64），无 dataset-specific
参数，推理用同一 checkpoint。

## 2. 结果（四域 AC，统一 fresh per-video 协议，官方 TrackEval）

| Domain | AssA | IDF1 | IDSW |
|---|---:|---:|---:|
| DanceTrack val | 0.4169 | 0.5694 | 2,588 |
| MOT17 | 0.6050 | 0.5825 | 259 |
| MOT20 | 0.2950 | 0.4012 | 2,406 |
| BDD（11 类 GT） | 0.2881 | 0.2923 | 11,042 |

## 3. 与强基座对比（同协议）

| Domain | L1DK AssA | U0 AssA | Δ | L1DK IDSW | U0 IDSW |
|---|---:|---:|---:|---:|---:|
| DanceTrack | 0.4165 | 0.4169 | +0.04pp | 2,558 | 2,588 |
| MOT17 | 0.5883 | 0.6050 | **+1.67pp** | 280 | **259** |
| MOT20 | 0.2778 | 0.2950 | **+1.72pp** | 2,603 | **2,406** |
| BDD | 0.2951 | 0.2881 | −0.70pp | 12,405 | **11,042** |

Macro AssA：U0 0.4013 vs L1DK 0.3944（+0.69pp）。

## 4. 结论

1. **naive shared 学习型核心（U0）已超过 L1DK 固定规则**：
   MOT17/MOT20 明显正向，BDD 略降，DanceTrack 持平；
2. 说明“负迁移”不是灾难性的：共享学习本身能吸收多域数据；
3. U0 是 L3 真正的 shared dense baseline，U1 必须在此基础上证明
   regime 条件化增益。

## 5. 协议说明

本表使用 per-video fresh OnlineTracker（L1 协议），与 L2 报告中
MOT17/MOT20/BDD 的 L1DK 数字（旧 shared-tracker 输出）不完全一致；
差异源于旧输出跨视频共享 tracker 状态，TrackEval 对 ID 重标号后
AssA 仍受关联历史影响。L3 所有方法在同一 fresh 协议下比较。
