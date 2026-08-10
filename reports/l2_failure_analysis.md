# Stage L2 — Failure Analysis

日期：2026-08-10。

## 1. 失败结论

```text
L2_ORACLE_HEADROOM_LOW
```

核心假设“用 counterfactual future trajectory utility 训练当前关联
决策可以改善统一 MOT”在 L1DK base + Association-Controlled 协议下
**没有可实现的 oracle headroom**，因此按任务书停止大型 TUM 训练。

## 2. 证据链

1. 单事件窗口 headroom：DanceTrack H32 平均 +0.74pp，BDD H16
   +1.01pp（见 `reports/l2_oracle_headroom.md`）；
2. 端到端 greedy oracle（privileged）：DanceTrack +0.02/+0.06pp，
   BDD 均值 −0.88pp，MOT17 −2.32pp；IDSW 全部变差；
3. local vs future mismatch 存在（DanceTrack H32 21.9% 事件未来最优
   不同于 base），但幅度不足以转化为收益（见
   `reports/l2_local_vs_future_mismatch.md`）；
4. EGRA 修正审计（专用审计）：DanceTrack val helpful 334 /
   harmful 264 / same_gt 213 / other 484；BDD helpful 163 /
   harmful 247 / same_gt 748 / other 641；BDD 59.8% 修正位于
   purity<0.5 轨迹且 harmful 多于 helpful。

## 3. 根因分析

### 3.1 基座窗口内已接近最优

DanceTrack 冲突事件 H4 窗口 base AssA=0.9734，几乎没有可修空间；
可改进事件占比仅 7.5%（H4）到 21.9%（H32）。

### 3.2 局部窗口最优 ≠ 全局轨迹质量

greedy oracle 优化 windowed AssA，但整视频 IDSW 反而上升：

- DanceTrack 149→151 / 68→74；
- BDD 260→283；
- MOT17 783→866。

这与 L1-D 观察一致：TrackEval 的 IDSW/AssA 是全局 ID 共现统计，
局部窗口效用与其不同构。

### 3.3 动作空间经 base 再优化后趋同

大量替代 action 在 `complete_assignment` + base policy 再优化后
与 base 行为相同（效用完全相同的事件占比高），导致
“可行动的”headroom 更小。

### 3.4 历史污染不可在短窗口内修复

已污染轨迹（purity<0.8）上的修正即使局部 GT 正确，也不能恢复
该轨迹过去已经发生的 ID 错误；未来窗口效用不奖励这种“迟到的正确”。

## 4. 如果继续会怎样

- TUM 即使能精确预测 oracle 偏好，预测出的 action 也只能带来
  ≤0.1pp 整视频 AssA（DanceTrack），无法通过 +2pp / IDSW −15%
  的强信号门槛；
- 因此训练大型模型是浪费；训练小模型也只能得到一个“预测很准但
  没收益”的 utility learner，不能支撑 ICLR 实证要求。

## 5. 可保留的科学产出

1. **Objective mismatch 的实验证据**：local correctness 与 future
   windowed utility 在 20–60% 冲突事件上不一致；
2. **TrackEval 目标审计**：AssA/IDF1 是轨迹级 ID 共现统计；
3. **验证过的 windowed AssA 实现**：整视频结果与官方 TrackEval
   完全一致；
4. **L1DK base 四域公平矩阵**；
5. **2025–2026 官方代码审计**：无直接等价方法。

这些证据指向：若未来要追求该方向，应改变**效用定义**
（例如整序列 ID 映射 + IDSW 惩罚）或**基座协议**
（例如允许恢复/重映射 ID），而不是简单堆模型容量。
