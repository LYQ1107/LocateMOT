# Stage L4 — Novelty Collision Audit

日期：2026-08-10。

## 1. 核心主张

> MOT identity should be stable to how the target object set is specified.
> We formulate specification as a set-restriction operator and enforce
> restriction-equivariant identity tracking (one shared tracker across
> heterogeneous MOT domains and specification interfaces).

等价形式：

```
T_theta(R_s(X)) ≈ R_s(T_theta(X))   （common objects 上，permutation-invariant）
s_a ~ s_b 语义等价 → T(R_sa(X)) ≈ T(R_sb(X))
```

## 2. 逐项回答

### 2.1 是否已有「same video + different specification → same persistent identity」

否。已核实 SAM3/GLEE/OVTR/OVTrack/Grounded-SAM2/SAM2MOT/STORM/QTrack/
EPIPTrack/TempRMOT/TellTrack/NOVA/COVTrack/GOVTrack/ViewSAM 等：
所有方法在同一查询/spec 内维护身份，不评估、不优化「同一个视频换一种
specification 后身份是否保持」。V²-SAM/ViewSAM 的一致性跨**视角**，
不是候选子集限制。

### 2.2 是否已有 MOT 方法要求 Track(Restrict(X)) ≈ Restrict(Track(X))

否。Path Consistency 是最接近的「一致性」方法，但其路径来自观测子集
（时间采样），不是 specification 诱导的候选对象限制；且没有
Track-All-Then-Filter vs Pre-Filter 的对偶审计。

### 2.3 是否只是 SAM3/GLEE 的重述

不是。SAM3/GLEE 的贡献是「多种 prompt 输入接口 + 统一检测/分割/跟踪」；
它们不要求不同候选子集下身份等价，也不把 specification 定义为
set-restriction operator。L4 不新增 prompt 编码器，核心是
restriction-equivariant identity learning。

### 2.4 是否只是 post-filter

不是。post-filter（P0）是必须击败/解释的强 baseline；Stage L4-A 的
paired audit 正是为了证明 P0 与 P1 在 common objects 上身份不一致，
从而说明「单纯 post-filter 不保证身份稳定」。

### 2.5 是否只是 Path Consistency 的 prompt 版本

不是。Path Consistency 在无 GT 身份监督下约束时间子采样路径的关联一致；
L4 使用 privileged GT 配对视图（诊断 + 训练）、TrackEval 主指标和
candidate-subset restriction，不是路径采样的直接推广。Path Consistency
仅作为 assignment-consistency loss 的数学参考。

## 3. 结论

**NO_DIRECTLY_EQUIVALENT_VERIFIED_METHOD_FOUND**

注意：

1. 这是「在 2026-08-10 可核实范围内未发现」，不是「first」；
2. 检索盲区包括付费期刊、未公开代码、非英语来源；最终稿前需再次检索；
3. 若发现等价方法，必须修改 claim 并明确差异。
