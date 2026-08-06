# 许可证审计（License Audit）

## LocateAnything / Eagle

- 代码：Apache-2.0（仓库顶层 `LICENSE`）。
- 模型权重：NVIDIA License（`Embodied/LICENSE_MODEL`），非商业使用（research/evaluation only），允许复制、修改、再分发（保留完整许可证与版权声明），禁止商业使用（NVIDIA 及关联方除外）。
- 构成组件：
  - Qwen2.5-3B-Instruct：Qwen Research License（HF 模型卡片声明）。
  - MoonViT-SO-400M：MIT License（HF 模型卡片声明）。
- 对本项目的含义：学术研究可用；不能用于商业发布；派生模型/输出若分发须保留 NVIDIA 许可证与声明；不能把权重提交到 Git。

## MOTIP / MOTIP-2

- MOTIP：Apache-2.0。
- MOTIP-2：Apache-2.0。
- 本项目未复制其代码，仅作设计参考，无派生义务；若未来提取代码片段须保留原版权声明并记录来源 commit。

## TrackFormer

- Apache-2.0。未复制代码。

## MOTR

- 仓库内 `LICENSE` 为 MIT（Copyright (c) 2021 megvii-model）。GitHub API 显示 “Other” 与仓库文件不一致，以仓库文件为准。未复制代码。

## MeMOTR

- MIT（Copyright (c) 2023 Multimedia Computing Group, Nanjing University）。未复制代码。

## SAM 2

- Apache-2.0。未复制代码。

## ViPT

- MIT（Copyright (c) 2023 Jiawen Zhu）。未复制代码。

## RL 参考（仅记录，不 clone 代码）

- UniVG-R1：仓库未发现 LICENSE 文件，许可证未声明；只阅读 README/论文，不复制代码。
- Vision_GRPO：Apache-2.0；只阅读 README。
- MedLoc-R1：无许可证文件且代码未正式发布；只阅读 README。

## 约束汇总

- 允许：阅读、理解、基于公开接口自行实现（clean reimplementation），并在源码头部注明参考来源。
- 允许（带条件）：Apache/MIT 许可代码在保留版权声明与许可证的前提下复制/修改。
- 不允许：绕过 NVIDIA 非商业许可使用 LocateAnything 权重；把整个第三方仓库复制进 `locatemot` 核心包；无来源复制实现。
- 当前状态：未从任何参考仓库复制代码；所有借鉴均记录在 `docs/implementation_evidence.md` 与 `docs/reference_repository_inventory.md`。
