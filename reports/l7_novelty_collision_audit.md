# Stage L7 Novelty Collision Audit

目标 claim（工作版，不写 first）：

> **Specification-conditioned target selection + specification-shared causal
> identity dynamics**：不同 WHAT-TO-TRACK specification（ALL / open-vocabulary
> category / referring language）共享同一个 HOW-TO-TRACK 持久身份动力学核心。

## 是否存在直接等价方法

检查项：closed-set MOT + OVMOT + RMOT；one shared identity core；one shared
checkpoint；persistent learned identity dynamics；specification-conditioned
target selection；无 dataset-specific tracker。

- **OVTR（ICLR 2025）**：同时有 closed-set 检测与 OV 类别嵌入，track query
  传播，但没有 RMOT、没有持久记忆/生命周期/身份转移核心；spec 是固定 1732
  类 CLIP embedding，不是任意 language query。
- **COVTrack（ICCV 2025）**：OVMOT 的 cue 融合与置信度门控，非跨 formulation
  身份动力学核心，无 RMOT。
- **QTrack（2026）**：query-driven RMOT（3B VLM + RL），无 closed-set/OVMOT
  共享核心。
- **STORM（2026）**：end-to-end RMOT 大模型，无 closed-set/OVMOT。
- **TempRMOT（2024）**：仅 RMOT（Refer-KITTI），MOTR + 记忆库，无跨 task 核心。

结论：`NO_DIRECT_EQUIVALENT_VERIFIED`（截至 2026-08-14 检索的官方仓库）。
没有任何已核验公开方法同时满足上述全部条件。

## 强碰撞与 claim 边界

1. **cue reliability / adaptive multi-cue fusion 已不是新机制**：
   COVTrack（ICCV 2025）公开实现 association-embedding 空间的
   appearance/motion/semantic 门控残差融合 + intra/inter-frame confidence。
   因此我们的身份转移解码器内的 cue reliability 只能作为“统一身份动力学
   的一个组成部分”，不能单独作为论文第一创新；必须在论文中明确对比 COVTrack
   并说明差异（decision-level、set-interaction、persistent state、lifecycle）。
2. **query-driven RMOT 已有 QTrack/STORM**：RMOT 任务的 novelty 不在
   “language 条件化跟踪”，而在 shared HOW-to-track core 的跨 formulation
   迁移证据（closed-set→OV→referring 用同一核心）。
3. **OVMOT 端到端已有 OVTR**：OVMOT 的 novelty 不在 end-to-end OV 跟踪，
   而在 Base/Novel 均进入同一 shared UIDM 且无 novel 专用 head。

## 我方需要修改的表述

- 不用 “FIRST”；用 `NO_DIRECT_EQUIVALENT_VERIFIED`。
- 论文核心机制表述为三件套：Specification Encoder（WHAT）、Shared Causal
  Identity Dynamics（HOW）、Reliability-aware Identity Transition（证据异质性），
  其中第三项作为支撑组件而非独立创新。
- COVTrack 的 MCF 在 association embedding 层、无 lifecycle/NEW/NO-MATCH/
  持久因果转移；我们的是 identity-transition decoder 内的 decision-level
  可靠性 + 轨迹集合交互。若最终实验显示两者无法区分，则删去第三项。

## 未核实的仓库（PAPER_ONLY）

- COVTrack++（2026-03，arXiv）：官方页显示 code/dataset "will be publicly
  available"，当前未找到官方仓库，记为 PAPER_ONLY。
- ReaMOT：README 承诺接受后开源，当前无模型代码。
- STORM 模型代码：benchmark repo 未含模型，PAPER_ONLY。

