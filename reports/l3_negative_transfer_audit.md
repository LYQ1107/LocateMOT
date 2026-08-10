# Stage L3 — Naive Shared 负迁移审计

日期：2026-08-10。

## 1. 定义

naive shared = 用同一固定规则/同一 checkpoint 跨全部域。若每个域的最优
规则不同，则任何单一 shared 规则必然在某域负迁移。

## 2. 每域最优方法（官方 TrackEval AC，L2 baseline 矩阵）

| Domain | 最优 | AssA | 次优 | AssA | 差距 |
|---|---:|---:|---:|---:|---:|
| DanceTrack val | C1（Motion） | 0.4193 | L1DK | 0.4165 | +0.28pp |
| MOT17 | L1DK | 0.6010 | C1 | 0.5530 | +4.80pp |
| MOT20 | L1DK_d03（EGRA） | 0.2864 | C1 | 0.2869 | +0.05pp |
| BDD | L1DK | 0.3292 | C1 | 0.3019 | +2.73pp |

单一 shared 选择（L1DK）相对 per-domain 最优的负迁移：

- DanceTrack −0.28pp；MOT17 0；MOT20 −0.85pp；BDD 0；
- macro 负迁移 ≈ −0.28pp（若 per-domain 最优为 C1/EGRA/L1DK/L1DK）。

结论：**负迁移存在但幅度小**；更强的是“不同域/regime 下方法偏好
确实不同”（见 `reports/l3_regime_signal.md`）。

## 3. 为什么幅度小

L1DK 的 0.4 IoU + 0.2 PBD + 0.4 Kalman-motion 线性融合是强先验，
在任何单域都不是最差；这既是优点（共享稳定），也意味着
“固定规则”的失败不是灾难性的，需要更细粒度（regime 内）的证据
来支撑条件化动机。

## 4. 与 L2 oracle 的关系

L2 证明：即使未来 oracle 也不能在 AC 协议内大幅提升 L1DK 的整视频
AssA。因此 L3 的 U1 目标不是“在 L1DK 上再涨 2pp”，而是：

- 统一模型在 4 域同时达到 per-domain 强基线的水平（消除负迁移）；
- 多类 BDD 与 spec/prompt 接口的统一能力；
- 跨域 regime 条件化的机制可解释性。
