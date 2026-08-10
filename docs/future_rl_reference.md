# Stage L2 — 未来 RL/GRPO 参考记录（仅调研，不启动训练）

日期：2026-08-10。
规则：Stage L2 禁止启动 RL/GRPO。本文只记录已核实的官方代码、
训练资源需求与可迁移模块，供后续 Stage 使用。

## 1. 视觉 grounding GRPO

- Ground-R1（arXiv 2503.24358）：基于 Qwen2.5-VL 的 visual grounding
  RL，使用 GRPO + grounding 规则奖励；官方代码
  `github.com/zehao-wang/Ground-R1`。
- UniVG-R1（arXiv 2505.20466）：统一视觉 grounding 的 RL 框架，
  官方代码 `github.com/zhongyingpeng/UniVG-R1`。
- 可迁移模块：rule-based grounding reward、GRPO 的 reward
  normalization、KL 保持、多模态 LoRA/RL 的训练配方。
- 不可迁移：其 reward 是单图 grounding 框匹配，不是轨迹身份效用。

## 2. 目标跟踪 RL

- RELO（ICML 2026，arXiv 2605.07379）：RL to localize 的单目标 VOT，
  官方代码 `github.com/Multimedia-Analytics-Laboratory/RELO`。
- MATT-Diff / AOT-ARL：主动目标跟踪（相机控制）RL，与 MOT 关联无关。
- Query-MARFT（2026，Pattern Recognition）：端到端 MOT 的
  多 agent RL fine-tuning（DetAgent/AssocAgent/UpdateAgent/CorrAgent，
  Flexible Markov Game）；**未找到官方公开代码**（仅论文）。
- 结论：MOT 关联级 RL 官方实现稀缺；Query-MARFT 是最接近的
  “关联策略 RL”，但没有 counterfactual trajectory utility。

## 3. 结构化框奖励 / ID 一致性奖励 / 可验证轨迹奖励

- 结构化框奖励：Ground-R1/UniVG-R1 提供框级规则奖励
  （IoU 阈值分档），可作为“框奖励”参考。
- ID 一致性奖励：未找到直接官方实现；本项目若进入 RL，需自行定义
  windowed AssA / IDSW 作为 verifiable trajectory reward（见
  `docs/l2_trackeval_objective_audit.md`）。
- 可验证轨迹奖励：可将 TrackEval AssA/IDF1/IDSW 的窗口版本作为
  rule-based reward；这正是 Stage L2 supervised utility 的同一
  目标函数，RL 只是优化器差异。

## 4. 训练资源需求估计

- Ground-R1 类 7B VLM RL：通常需要 8×A100（80G）+ 数天；
  本项目 4×40G 不满足直接复刻。
- 若本项目 RL 化：建议 0.5–50M 参数轻量 utility/policy 模型，
  trajectory rollout 在 CPU/低并发完成，GPU 只训练小模型；
  单卡 4×40G 足够。

## 5. 进入 RL 的条件（Stage L2 设定）

1. supervised counterfactual utility learning 已证明有效；
2. 仍存在明显 exposure bias / long-horizon mismatch；
3. 有官方 RL 框架可复用依据（如 GRPO 的 reward normalization）。

否则保持 supervised utility / preference learning。
