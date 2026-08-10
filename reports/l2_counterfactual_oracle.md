# Stage L2 — Counterfactual Rollout Oracle

日期：2026-08-10。

## 1. 目的

在训练任何大模型之前，回答：

1. 当前关联决策（L1DK base）是否存在“换一个 action 后未来轨迹效用
   显著更好”的空间（oracle headroom）？
2. local correctness 与 future utility 是否真的不一致？

## 2. 状态与数据

- 基座：BEST_STRONG_BASE = L1DK base
  （0.4 IoU + 0.2 PBD + 0.4 Kalman motion，thr 0.25，max_age 30）；
- 重放：`tools/run_l2_oracle.py` 用与 baseline 完全相同的 AC shell
  逐帧重放，已验证与官方基线输出 100% 一致（MOT17-04-SDP 3589/3589）；
- 数据域：DanceTrack val（40 视频）、BDD100K train（30 视频采样）、
  MOT17 train（3 视频）、MOT20 train（2 视频）、DanceTrack calibration
  （8 视频）；
- 冲突定义：affinity 图（base≥0.25 或 row/col top-2）的连通分量，
  只保留 |T|≥2 或 |C|≥2 的组件；每帧最多采样 40 个冲突。

## 3. 动作空间

每个冲突组件生成 6–8 个候选 action：

1. base action（A0，必选）；
2. GT-local action（若轨迹真 ID 出现在组件候选内）；
3. 按 base 分数排序的 top-k 替代匹配（穷举小组件，大组件启发式）；
4. 最差匹配（sanity）；
5. all-new（组件内全部不匹配，候选全部出生）。

每个 action 用 `complete_assignment` 补全为合法全局一对一分配，
然后冻结 base policy 继续 rollout H 帧。

## 4. 效用定义

窗口效用用 TrackEval 同款公式（见
`docs/l2_trackeval_objective_audit.md`）：

- `U_H = windowed AssA`（主指标）；
- 辅助：windowed IDF1、window IDSW、TP/GT 计数。

窗口取 action 之后未来帧 `[t+1, t+H]`（不含 t，保证纯未来后果）。

## 5. 结果

### DanceTrack val（25 视频，1,000 冲突事件）

| H | base windowed AssA | oracle-best | mean gain | frac better |
|---|---:|---:|---:|---:|
| 4 | 0.9734 | 0.9772 | +0.38pp | 7.5% |
| 8 | 0.9593 | 0.9645 | +0.52pp | 13.5% |
| 16 | 0.9444 | 0.9509 | +0.65pp | 18.5% |
| 32 | 0.9219 | 0.9293 | +0.74pp | 21.9% |

### BDD100K train（30 视频，745 冲突事件，5fps）

| H | base windowed AssA | oracle-best | mean gain | frac better |
|---|---:|---:|---:|---:|
| 2 | 0.7237 | 0.7334 | +0.96pp | 18.0% |
| 4 | 0.6225 | 0.6332 | +1.07pp | 36.0% |
| 8 | 0.5313 | 0.5429 | +1.17pp | 53.3% |
| 16 | 0.4709 | 0.4809 | +1.01pp | 61.7% |

### MOT17 / MOT20（小样本）

| 域 | 事件 | H32 mean gain | frac better |
|---|---:|---:|---:|
| MOT17 train | 120 | +2.27pp | 70.8% |
| MOT20 train | 80 | +1.76pp | 75.0% |

单事件窗口 headroom 在 MOT17/MOT20 反而最大，但端到端验证 MOT17
为 −2.32pp（见 `reports/l2_oracle_headroom.md`）。

### 端到端 privileged greedy oracle

| 视频 | gain（整视频 AssA） | IDSW base→oracle |
|---|---:|---:|
| dancetrack0004 | +0.02pp | 149→151 |
| dancetrack0005 | +0.06pp | 68→74 |
| BDD×3 | −0.88pp（均值） | 260→283 |
| MOT17-02-SDP | −2.32pp | 783→866 |

结论：单事件窗口 headroom 存在但小；端到端实现后无正收益。

## 6. 输出文件

- `outputs/l2/oracle/events_{domain}.pkl`：逐事件动作与效用；
- `outputs/l2/oracle/oracle_{domain}.json`：聚合 headroom；
- `outputs/l2/oracle/analysis_{domain}.json`：mismatch/ranking 分析。
