# Stage L3 — Unified MOT 任务定义

日期：2026-08-10。

## 1. 两个正交轴

Unified MOT 被严格定义为两个正交轴：

### Axis A — Tracking Regime / Domain

- dense same-class（DanceTrack：高同类外观歧义、交叉、非线性运动）；
- standard pedestrian（MOT17）；
- extreme crowd（MOT20）；
- multi-class driving（BDD100K：11 类、ego-motion、5fps 采样）；
- long-tail / open-world / sparse annotation（TAO，延迟纳入）。

### Axis B — Object Specification

- ALL / track-all（无 prompt）；
- category text（person/car/…）；
- open-vocabulary text；
- referring description（诊断级，STORM-Bench/QTrack 若数据可获取）；
- visual prompt（box 必选；point/mask 视成本）。

## 2. 模型目标

```text
Tracker(video, specification) -> persistent object trajectories
```

- specification 可以为空（track-all）；
- 同一 core、同一 checkpoint；
- 无 dataset-specific head / adapter / threshold；
- 推理严格 online causal；
- dataset name / path / annotation source 禁止作为输入。

## 3. 统一学习表述

```text
What to track  -> Object Specification Token (SpecToken)
How to track   -> Latent Tracking Regime Token (z_regime)
TrackState_{t+1} = F_theta(TrackState_t, CurrentObjects_t, SpecToken, z_regime)
```

## 4. 边界（避免失焦）

- 主输出仍为 box trajectory + ID；mask 只作为 prompt 输入接口
  （mask→初始 spec/object token），除非标准 benchmark 强制 mask 输出；
- 不把 SOT/VOS 无限纳入；promptable segmentation 本身不是 novelty
  （SAM3/GLEE 已覆盖，见 `reports/l3_novelty_collision_audit.md`）；
- 不做 retrospective ID revision / bounded-latency correction；
- 不做 RL 主线（L2 已否决 future-utility RL）。

## 5. 论文 claim 边界（暂定，实验支持后才可用）

异质 MOT benchmark 的差异不仅是 visual domain，还包括决定
“哪些时间/语义证据可靠”的 tracking regime。naive shared tracker
因此产生负迁移。我们提出 specification × regime 因子分解，
用一个共享 tracking core 统一多域与多对象指定接口，且
无 dataset-specific 参数。

若实验不支持，则退回 `L3_REGIME_NOT_SUPPORTED` 或
`L3_PROMPT_UNIFICATION_PARTIAL`。
