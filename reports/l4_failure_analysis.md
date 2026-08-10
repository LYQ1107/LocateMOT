# Stage L4 — Failure Analysis

日期：2026-08-10。

## 1. 结论

Stage L4 pilot 未通过：paired-view 的逐帧 assignment/state consistency
（A5）与 partition-level co-assignment consistency（A5p，一次最小修正）
都没有降低跨 spec identity drift；官方 TrackEval ALL 模式保持不变，
audit 均值 ALL 轻微下降。

## 2. 为什么一致性训练无效

### 2.1 身份漂移是时间现象，不是单帧分配现象

审计指标里的 drift 是「同一 GT 对象在两个视图中的长期轨迹 partition
不一致」，主要由不同时刻的 merge/split/switch 造成。A2/A5 的
consistency loss 只约束单帧的 assignment 分布（row/col KL）或单帧
co-assignment 矩阵（partition MSE），没有约束「跨时间身份轨迹」，
因此不能降低 drift。

### 2.2 轨迹对齐噪声

A5 用 birth-GT 对齐两视图轨迹。当 base tracker 发生身份错误时
（track 的 birth 身份 ≠ 当前匹配对象的身份），这种对齐把「错误身份」
当共同身份来压一致，反而把错误固化。A5p 改为 partition-level 对齐
（co-assignment 矩阵 MSE），绕开轨迹对齐，但该 loss 数值接近 0
（~1e-4），仍只覆盖单帧 co-assignment，无法约束跨帧 ID 迁移。

### 2.3 base affinity 主导 + 集合级特征

EGRA 的 final = base + gated residual；base 本身包含
`log_n_cand/margins/top1` 等集合级特征，候选子集变化会同时改变
base 与上下文 token。0.49M 模型在 20 epochs 内没有学到能抵消
set-context 变化的残差。

### 2.4 训练-评估协议不匹配

训练时每帧 paired softmax 被一致性正则化；评估时 OnlineTracker 在
每视图独立做 Hungarian + 生命周期。softmax 的一致不保证 Hungarian
分配与跨帧 ID 迁移一致。

## 3. 为什么不继续

- 任务书允许「一次最小机制修正」，已执行（A5p）；
- 不允许无限调 lambda / 堆容量；
- 当前证据指向「逐帧 association-level consistency」不是正确的
  mechanism，需要 trajectory-level / differentiable-tracking 层面
  的重设计，超出本阶段时间预算。

## 4. Stage Decision

```text
L4_PILOT_GATE_FAIL
L4_NOT_SUPPORTED（pilot mechanism）
Problem Signal：L4_SPEC_RESTRICTION_SIGNAL_SUPPORTED（真实存在）
ICLR readiness：NOT_READY
```

## 5. 保留资产

- P0/P1 审计证明 specification restriction 真实改变身份（U0）；
- 配对数据、指标（co-identity agreement）、TAO 恢复；
- A2/A5/A5p 训练管线可复用；
- 失败模式记录：单帧 consistency ≠ temporal identity consistency。
