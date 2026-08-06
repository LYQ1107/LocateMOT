# RL / GRPO 参考记录（Stage L0 仅记录，不训练）

Stage L0 明确禁止执行 GRPO 或其他 RL。本文件只记录已检索到的官方/高质量实现、训练资源需求与可迁移模块，供 Stage L1 参考。

## 1. 视觉 grounding GRPO

### UniVG-R1（最相关）

- 论文：UniVG-R1: Reasoning Guided Universal Visual Grounding with Reinforcement Learning（arXiv:2505.14231）
- 官方仓库：https://github.com/AMAP-ML/UniVG-R1
- HEAD commit：`44868ea30073c104d026186418291757454ae9d7`（2026-08-06 查询）
- 许可证：仓库未发现 LICENSE 文件（未 clone、未复制）。
- 实现要点（来自 README）：
  - 基于 VLM-R1 / Migician / Visual-RFT，两阶段：Stage 1 CoT-SFT，Stage 2 GRPO。
  - GRPO 训练脚本：`src/open-r1-multimodal/run_scripts/run_grpo_univg.sh`。
  - 提出 difficulty-aware weight adjustment 缓解 GRPO 的难度偏差。
- 资源需求：需要 VLM（Qwen2.5-VL 类）多模态环境、8+ GPU；Stage 2 数据为 `stage2_rl.json` + MGrounding-630k。
- 可迁移模块：GRPO reward 框架、box 结构化输出 reward、cold-start 初始化策略。

### Vision_GRPO

- 官方仓库：https://github.com/FusionBrainLab/Vision_GRPO
- HEAD commit：`65ec90d090f93dae1f303ea7eeed8c9d0c06f64b`
- 许可证：Apache-2.0
- 实现要点：Qwen2VLGRPOTrainer，reward funcs = `grounding_reward` + `format_reward`；输出 `<answer>[x1,y1,x2,y2]</answer>`；基于 TextVQA 子集的 grounding 教学实现。
- 可迁移模块：grounding reward（框坐标/格式校验）与 TRL GRPO trainer 适配模式。

### MedLoc-R1

- 论文/仓库：CVPR 2026；https://github.com/MembrAI/MedLoc-R1
- HEAD commit：`9ae2b29854db16d09d6c5c6d59b13b3c70c1b48b`
- 状态：代码正在整理，尚未正式发布；基于 VLM-R1。
- 记录点：performance-aware curriculum reward scheduling for GRPO-based medical visual grounding。

### 其他记录

- MedGround-R1：论文提到 github.com/bio-mlhui/MedGround-R1，但检索时仓库 404；仅论文可查。
- R1-SAM：未检索到官方仓库（可能只有论文/博客），不列为参考。
- Eagle issue #53：官方答复未公开正文，但问题本身列出了 grounding-R1 风格的 point-in-box reward、检测集合级 F1/mAP reward、negative block 奖励、PBD 与 CoT 的潜在冲突。

## 2. 目标跟踪 RL

未发现针对 MOT 的官方 GRPO/RL 公开实现（截至 2026-08-06 检索）。相关可参考方向：

- 把 TrackEval 指标（HOTA/IDF1）分解为可微/可验证 reward 尚无官方代码。
- MOTIP/MOTR 均使用监督训练；RL 用于跟踪的公开工作多为仿真/无人机控制，与本项目两帧关联任务不直接对应。
- Stage L1 若做 RL，应自行定义并验证 reward：assignment 正确性、NO_MATCH 正确性、ID 一致性（两帧交换惩罚）、box IoU。

## 3. 结构化框奖励

- 可参考 `grounding_reward`（Vision_GRPO）与 UniVG-R1 的框输出校验。
- LocateAnything 的 PBD 输出是固定 6-token block，RL 奖励应作用于解析后的 block（`<box>` 合法性、坐标范围、`none`），而不是逐 token 奖励；PBD 的 block 结构使格式奖励可直接结构化。

## 4. ID 一致性奖励

- 无官方实现记录。设计建议：两帧关联正确（track i -> candidate j 且 GT 相同）得正奖励；交换/重复分配得负奖励；NO_MATCH 正确/错误分别奖励。
- 可验证性：两帧 GT 是公开可验证信号，符合 GRPO 可验证奖励要求。

## 5. 可验证轨迹奖励

- 无官方实现记录。建议：使用 held-out 两帧 pair 上的 assignment accuracy / NO_MATCH F1 作为 group-level 可验证奖励。

## 资源需求估计（供 Stage L1 规划）

- GRPO 通常需要 8×A100/H100、VLM 可训练 LoRA、rollout 缓存与采样吞吐。
- 若基于 LocateAnything PBD 做 RL，额外复杂点：hybrid 解码中 MTP->AR fallback 的可微/奖励口径、PBD block mask 与 generation 采样兼容性。
- 本阶段不做任何 RL 训练，不做权重或计算预算投入。

## 结论

截至 2026-08-06，视觉 grounding 的 GRPO 官方实现存在（UniVG-R1、Vision_GRPO、MedLoc-R1 未发布）；MOT 的 RL/ID 一致性奖励无官方实现，Stage L1 需自行设计并在本文件基础上补充证据。
