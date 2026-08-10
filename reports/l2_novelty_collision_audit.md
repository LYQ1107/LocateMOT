# Stage L2 — Novelty Collision Audit

日期：2026-08-10。
核心问题：是否存在已公开方法明确做——

> 训练时使用未来轨迹/反事实 rollout 评估当前 association action 的
> counterfactual long-term trajectory utility，再把该偏好蒸馏给
> online causal association policy。

## 1. 检索范围

- 关键词（2025/2026 优先）：future utility MOT、trajectory-level MOT
  loss、long-horizon MOT association、future-aware association、
  counterfactual tracking、track detection link prediction、
  trajectory consistency、sequence-level MOT objective、
  differentiable assignment、decision-focused tracking、
  reinforcement learning MOT。
- 逐项核实官方仓库（见 `docs/l2_reference_audit.md`），全部实际
  clone + 阅读 README/模型/loss/推理代码。

## 2. 逐项结论

| 候选方法 | 是否等价 | 差异 |
|---|---|---|
| TDLP | 否 | 训练监督=下一帧 link 正确性；无轨迹效用、无 counterfactual |
| SambaMOTR | 否 | 查询自回归传播；目标仍是当前帧检测+身份 |
| TRACT | 否 | 轨迹一致性/聚合，无未来效用 |
| UniTrack | 否 | 轨迹平滑 hinge loss，无关联未来后果 |
| Path Consistency | 否 | 不同观测路径的一致性，无 GT 身份效用 |
| QuoVadis | 否 | 未来位置回归，非身份效用 |
| MOTIP / MOTIP-2 | 否 | 当前帧 ID 分类 |
| CAMELTrack | 否 | 当前帧多 cue 判别 |
| FDTA / HATReID-MOT / HNCD-MOTR | 否 | 局部判别 / 历史变换 / 训练时 hard negative |
| MeMOTR | 否 | memory bank，无未来效用 |
| Query-MARFT | 否（且无官方代码） | multi-agent RL fine-tuning，监督仍是局部关联收益，无 counterfactual trajectory utility |
| RL 视觉 grounding（Ground-R1 / UniVG-R1 等） | 否 | 非 MOT 关联任务 |

## 3. 结论

**NO DIRECTLY EQUIVALENT VERIFIED METHOD FOUND**

即：在 2026-08-10 可核实的公开官方实现范围内，未发现与本项目
核心设定（counterfactual future trajectory utility → causal policy
蒸馏）直接等价的方法。

注意：

1. 该结论是“未发现”，不是“首创”。论文中不得使用 first；
2. 检索存在固有盲区（付费期刊、未公开代码、非英语来源），
   最终稿需在时间允许时再次检索；
3. 若未来发现等价方法，必须修改 claim 并明确差异。
