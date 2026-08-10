# Stage L2 — Oracle Headroom（Gate 1）

日期：2026-08-10。
Gate 1 判定：**TRAJECTORY_UTILITY_HEADROOM_LOW**（不启动大型 TUM 训练）。

## 1. 定义

Oracle headroom = 在允许使用未来 GT / 未来 rollout 的 privileged 条件下，
选择 counterfactual future-best action 相对 BEST_STRONG_BASE（L1DK base）
能获得的整视频 TrackEval AssA / IDSW 提升。

## 2. 单事件窗口 headroom（隔离评估）

每个冲突组件枚举 6–8 个候选 action，冻结 base policy rollout
H∈{4,8,16,32} 帧，计算 windowed AssA（与官方 TrackEval 公式一致，
整视频校验 AssA/IDF1 精确等于官方值）。

### DanceTrack val（1,000 冲突事件，25 视频）

| H | 事件数 | base windowed AssA | oracle-best | mean gain | frac better | base IDSW | best IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1,000 | 0.9734 | 0.9772 | +0.38pp | 7.5% | 0.64 | 0.59 |
| 8 | 1,000 | 0.9593 | 0.9645 | +0.52pp | 13.5% | 1.70 | 1.65 |
| 16 | 1,000 | 0.9444 | 0.9509 | +0.65pp | 18.5% | 3.80 | 3.77 |
| 32 | 1,000 | 0.9219 | 0.9293 | +0.74pp | 21.9% | 8.13 | 8.10 |

### BDD100K train（745 冲突事件，30 视频，5fps）

| H | 事件数 | base | oracle-best | mean gain | frac better |
|---|---:|---:|---:|---:|---:|
| 2 | 745 | 0.7237 | 0.7334 | +0.96pp | 18.0% |
| 4 | 745 | 0.6225 | 0.6332 | +1.07pp | 36.0% |
| 8 | 745 | 0.5313 | 0.5429 | +1.17pp | 53.3% |
| 16 | 745 | 0.4709 | 0.4809 | +1.01pp | 61.7% |

### MOT17 / MOT20（小样本）

| H | MOT17 gain | MOT17 frac | MOT20 gain | MOT20 frac |
|---|---:|---:|---:|---:|
| 4 | +1.61pp | 55.0% | +1.76pp | 66.2% |
| 8 | +1.83pp | 62.5% | +1.61pp | 65.0% |
| 16 | +2.25pp | 68.3% | +1.59pp | 72.5% |
| 32 | +2.27pp | 70.8% | +1.76pp | 75.0% |

注意：MOT17 的单事件窗口 headroom 反而最大（H32 +2.27pp），但端到端
为 −2.32pp——这是“窗口效用与全局轨迹质量不同构”的最强直接证据。

## 3. 端到端（receding-horizon greedy oracle）

对每个冲突帧用 privileged rollout 选最佳 action 并应用，其余帧用 base，
整视频计算 TrackEval 同款 AssA：

| 视频 | 帧数 | base AssA | oracle AssA | gain | oracle IDSW |
|---|---:|---:|---:|---:|---:|
| dancetrack0004 | 1,203 | 0.1702 | 0.1704 | +0.02pp | 151（base 149） |
| dancetrack0005 | 1,203 | 0.4562 | 0.4568 | +0.06pp | 74（base 68） |
| BDD 0000f77c-…58 | 41 | 0.1319 | 0.1582 | +2.62pp | 121（105） |
| BDD 0000f77c-…88 | 40 | 0.2439 | 0.1963 | −4.76pp | 36（30） |
| BDD 0000f77c-…98 | 40 | 0.2352 | 0.2302 | −0.50pp | 126（125） |
| MOT17-02-SDP | 80 | 0.1830 | 0.1598 | −2.32pp | 866（783） |

均值：DanceTrack +0.04pp；BDD −0.88pp；MOT17 −2.32pp。

## 4. 结论

1. **单事件窗口 headroom 存在但很小**（DanceTrack 0.4–0.7pp、
   BDD 1.0–1.2pp），且窗口重叠导致端到端可实现收益大幅稀释；
2. **端到端 oracle 不提升整视频 AssA**：DanceTrack 2 视频 +0.02/+0.06pp，
   BDD 3 视频平均 −0.88pp，MOT17 −2.32pp；IDSW 全部变差；
3. 即使拥有完整 privileged future 和 oracle action 选择，相对
   BEST_STRONG_BASE 的整视频 AssA headroom **< 0.1pp（DanceTrack）**，
   远低于 1pp 阈值；
4. 因此按任务书停止条件：

```text
L2_ORACLE_HEADROOM_LOW
```

不启动大型 Trajectory Utility Model 训练；进入失败分析与最终报告。

## 5. 为什么 headroom 低（机制）

- L1DK base 的匈牙利匹配在短窗口内已接近最优（DanceTrack H4 base
  AssA=0.973，几乎没有可修空间）；
- 大多数冲突组件的替代 action 经 base policy 再优化后与 base 等价
  （效用相同的事件占比很高）；
- 窗口级 AssA 最优 ≠ 全局身份质量：贪婪 oracle 的局部选择反而增加
  IDSW（DanceTrack 149→151、BDD 260→283、MOT17 783→866）；
- 这与 L1-D 的结论一致：local/future-window 的修正与 TrackEval
  轨迹级统计不同构。
