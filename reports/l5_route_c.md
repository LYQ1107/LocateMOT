# Stage L5 — Route C: Multi-Path / Subset-Perturbation Trajectory Consistency

日期：2026-08-11。

## 状态

**NOT_EXECUTED（独立路线）**。

Route C 的核心思想（在不同 object-subset observation path 上学习一致的
trajectory identity，GT anchored）已被 Route A 的 assignment-level
cross-spec KL 部分覆盖：Route A 在 ALL/restricted 两个 subset path 上
要求 common candidate 的 identity 分配一致，且两个视图分别由 GT 监督。

在 Route A 未通过 full-scale 判据、Route B 已证伪的前提下，没有剩余
算力再独立实现 Route C 的完整 trajectory-level 变体（需要在线回滚式
训练，约 4–8 GPU·小时）。

如果继续，推荐实现：对同一 clip 的多个 subset perturbation（随机 dropout
candidate）rollout 模型自身轨迹，并监督跨路径同一 GT 的 track-chain
一致性（用可微的 soft-Hungarian 或 Gumbel-Sinkhorn 近似）。
