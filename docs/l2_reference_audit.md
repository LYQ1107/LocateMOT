# Stage L2 — 2025/2026 参考实现审计

日期：2026-08-10。
范围：Stage L2 要求的“训练用未来 / 轨迹级效用 / counterfactual /
future-aware association”方向，全部实际 clone 并阅读官方代码；不依据
摘要或博客转述。仓库固定 commit 记录于 `docs/reference_repository_inventory.md`
（本文件只列出 L2 新增与重点仓库）。

## 0. 审计问题模板

对每个方法回答：

- 官方仓库是否验证（URL、commit、license）
- 训练是否使用未来帧/未来 GT；推理是否 causal
- 关联表示（pair embedding / link prediction / ID classification / query）
- 训练目标（local correctness / trajectory-level / future utility）
- 是否 counterfactual rollout；是否 RL
- 与 Stage L2 的关系：可借鉴 / 不采用及原因

## 1. TDLP（arXiv 2512.22105，2025/2026）

- 官方仓库：`github.com/Robotmurlock/TDLP`（已 clone）
- Commit：`50344b92`（2026-05-23）；License：MIT
- 已读文件：
  - `README.md`
  - `tdlp/datasets/dataset/mot.py`（clip 采样：observed clip + unobserved 下一帧）
  - `tdlp/trainer/trainer.py`（`_forward_and_loss`：track_x=observed，
    det_x=unobserved）
  - `tdlp/trainer/losses/bce.py`、`infonce.py`、`triplet.py`
  - `tdlp/tracker/online.py`（在线推理）
- 问题：track–detection link prediction，逐帧关联。
- 局部/轨迹级：逐帧 link（tracklet 历史作为上下文）。
- 训练用未来：是。监督是 **下一帧** 的 GT 身份匹配（clip 内预测未来 1 帧）。
- 推理用未来：否，严格在线。
- 关联表示：bbox/特征序列 → 双塔/多模态编码 → link logits
  （TDCP/TDSP），匈牙利 + 阈值解码。
- 训练目标：BCE / InfoNCE / Triplet 的 **同身份判别**；无轨迹级效用。
- 长期目标：无（预测下一帧）。
- 可微分配：link logits 可微；解码用匈牙利（不可微）。
- RL：无。
- 可借鉴：clip=历史+未来一帧的构造；在线 tracker 的 tracklet history
  编码；link prediction 形式的 set-level 竞争。
- 不采用：单帧 link 目标正是 Stage L2 要超越的 local correctness；
  TDLP 没有 counterfactual 长期效用。

## 2. SambaMOTR（ICLR 2025 Spotlight）

- 官方仓库：`github.com/mattiasegu/sambamotr`（已 clone）
- Commit：`f1c139a6`（2025-03-31）；License：MIT
- 已读文件：
  - `README.md`
  - `models/sambamotr.py`（tracking-by-propagation，autoregressive query）
  - `models/query_updater.py`、`models/query_updaters/samba.py`
    （Samba 状态空间同步跨 tracklet 序列）
- 问题：端到端 MOT，用 set-of-sequences 状态空间模型传播 track queries。
- 局部/轨迹级：轨迹查询序列级建模（长时依赖、遮挡记忆）。
- 训练用未来：端到端 clip 训练，查询沿时间自回归传播；
  监督是每帧检测框+GT 身份匹配。无“未来作为效用标签”。
- 推理用未来：否，逐帧在线。
- 关联表示：detection queries + propagated track queries，跨帧 self-attn。
- 训练目标：检测+关联的 set loss（当前帧 GT 匹配）。
- 长期目标：隐式（记忆），无显式 trajectory utility。
- 可微分配：是（端到端）。
- RL：无。
- 可借鉴：长期查询记忆（long memory）+ MaskObs 不确定性处理，
  可作为 TUM 的 track-state 编码参考。
- 不采用：其目标是生成/传播查询，不是“当前 action 的未来效用”。

## 3. TRACT（ICCV 2025）

- 官方仓库：`github.com/Nathan-Li123/TRACT`（已 clone）
- Commit：`19f01d72`（2025-10-04）；License：仓库无 LICENSE 文件
- 已读文件：`README.md`（TCR/TFA/TSE 三个组件说明）、目录结构
  （`masa/` 轨迹感知 MASA、`TraCLIP/` 轨迹感知 CLIP 分类）
- 问题：Open-Vocabulary MOT，利用轨迹级信息改善关联与分类。
- 局部/轨迹级：轨迹级（TCR 轨迹一致性强化、TFA 历史特征聚合）。
- 训练用未来：未发现显式未来标签；TFA/TCR 使用轨迹历史。
- 推理：在线关联。
- 训练目标：局部关联（相似度）+ 分类；TCR 是轨迹内部一致性。
- 可借鉴：轨迹历史特征聚合、轨迹一致性正则的表示层面思路。
- 不采用：无 counterfactual/future-utility；且与 MASA 耦合较深，
  不直接兼容本项目的 AC 协议。

## 4. UniTrack（ICLR 2026）

- 官方仓库：`github.com/ostadabbas/UniTrack`（已 clone）
- Commit：`afdd9869`（2026-02-03）；License：MIT
- 已读文件：`unitrack_criterion.py`（`_compute_spatial_consistency`、
  `_compute_temporal_consistency`、tracking score hinge loss）
- 问题：可插入任意 MOT 框架的通用 hinge loss。
- 局部/轨迹级：训练时对 clip 内同一 track 的 boxes 施加
  尺寸一致性与加速度平滑约束（temporal consistency）。
- 训练用未来：clip 内多帧都作为监督（时间窗口），但不是“未来效用”。
- 推理：无改动。
- 训练目标：跟踪分数 hinge + 空间/时间一致性（轨迹平滑）。
- 可借鉴：把轨迹级平滑约束写成可微 loss 的工程形式；
  可作为 TUM 的辅助正则候选。
- 不采用：它优化轨迹平滑而非关联决策的未来后果；没有 counterfactual。

## 5. Path Consistency（CVPR 2024）

- 官方仓库：`github.com/amazon-science/path-consistency`（已 clone）
- Commit：`f4b7d26d`（2024-07-22）；License：Apache-2.0
- 已读文件：`README.md`、`src/model/model.py`（`pcl` loss 调用）、
  `src/model/loss_tools.py`（`PathConsistencyLoss`、路径采样、路径 logprob）
- 问题：自监督 MOT 匹配，无需人工身份标注。
- 局部/轨迹级：轨迹路径级（跨帧组合匹配的 consistency）。
- 训练用未来：路径横跨未来帧（观察子集不同），但目标是
  多路径关联一致，不是未来轨迹效用。
- 推理：在线关联。
- 训练目标：Path Consistency Loss（不同观测路径的关联结果应一致）。
- 可借鉴：路径组合 + 一致性监督的数学框架；可作为 Stage L2 的
  辅助/消融（路径一致性 auxiliary）。
- 不采用：无 GT 身份监督、无 counterfactual action 评估；
  与“privileged future supervision”目标不同。

## 6. QuoVadis（NeurIPS 2022）

- 官方仓库：`github.com/dendorferpatrick/QuoVadis`（已 clone）
- Commit：`241233a5`（2023-01-10）；License：LICENSE.md
- 已读文件：`README.md`、`src/run_quovadis.py`、目录结构
- 问题：长期遮挡下的轨迹恢复；用未来轨迹预测缩小关联搜索空间。
- 局部/轨迹级：轨迹预测（trajectory forecasting）辅助长期关联。
- 训练用未来：是（轨迹预测模型用未来 GT 轨迹训练）。
- 推理：在线（预测器前向，不访问未来）。
- 训练目标：轨迹位置预测（回归），不是关联效用。
- 可借鉴：horizon 设计（长时预测）与“用未来训练、在线推理”的
  原则性一致；但目标是 box 未来位置而非身份效用。
- 不采用：不直接评估关联 action 的长期身份质量。

## 7. FDTA（CVPR 2026）

- 官方仓库：`github.com/...`（本地 `references/association_2025_2026/FDTA`）
- Commit：`b3b3b778`（2026-03-21）；License：仓库 LICENSE
- 已读文件：`README.md`（MOTIP-based DETR 范式）、目录结构
  （`models/`、`runtime_tracker.py`、`data/`）
- 问题：DETR 端到端 MOT 的 object embedding 过于相似，显式增强判别。
- 局部/轨迹级：局部判别。
- 训练用未来：clip 训练，无未来效用。
- 可借鉴：判别性 embedding 的 loss 设计（对比/去相关）可作为
  特征表示增强。
- 不采用：无轨迹级/未来效用。

## 8. HATReID-MOT（arXiv 2503.12562）

- 官方仓库：本地 `references/association_2025_2026/HATReID-MOT`
- Commit：`3eb440c2`（2026-07-23）；License：仓库 LICENSE
- 已读文件：`README.md`（HAT-SORT：历史感知 ReID 特征变换）
- 问题：用轨迹历史把 ReID 特征变换到更可分的子空间。
- 局部/轨迹级：轨迹历史 → 特征变换（局部关联）。
- 训练用未来：无。
- 可借鉴：track 级特征变换（history-aware）可作为 TUM 输入编码。
- 不采用：无未来效用。

## 9. HNCD-MOTR（2025/2026）

- 官方仓库：本地 `references/association_2025_2026/HNCD-MOTR`
- Commit：`2026-05-31`；License：仓库无 LICENSE 文件
- 已读文件：`models/hncd.py`（training-time contrastive denoising
  queries，`get_contrastive_denoising_training_group`）、README
- 问题：训练时用目标附近 hard negative 构造 denoising queries，
  推理不变。
- 局部/轨迹级：局部（当前帧 hard negatives）。
- 训练用未来：无。
- 可借鉴：training-time privileged augmentation 的思路
  （训练时知道目标、推理时不知道）与 Stage L2 的
  privileged future supervision 同族，但监督内容不同。
- 不采用：hard-negative denoising 仍是 local correctness。

## 10. MOTIP / MOTIP-2 / CAMELTrack / LG-Track / LLTrack / MeMOTR

这些已在 `docs/l1_c_reference_audit.md`、`docs/l1_d_reference_audit.md`
和 `docs/reference_repository_inventory.md` 中完成逐文件审计
（commit 见本文 0 节表格）。Stage L2 结论：

- MOTIP/MOTIP-2：ID prediction + track query/candidate 交互 + 匈牙利 +
  NEW 槽；训练目标是当前帧 ID 分类（local correctness）。
- CAMELTrack：多 cue（motion/appearance/geometry）set-level GAFFE +
  InfoNCE；训练目标是当前帧同身份判别。
- LG-Track/LLTrack：多 cue 成本融合 + 置信门控；启发式关联。
- MeMOTR：跨帧 memory key/value 更新；无未来效用。

均不构成“用未来轨迹效用训练当前关联决策”的等价方法。

## 11. RL / 决策学习方向（仅记录，Stage L2 不先启动 RL）

已检索并核实（无 MOT 关联级未来效用 RL 官方实现）：

- Query-MARFT（2026，Pattern Recognition）：query 级多 agent RL
  fine-tuning（DetAgent/AssocAgent/UpdateAgent/CorrAgent），无公开官方
  代码（ScienceDirect 论文，未找到 repo）。
- RELO（ICML 2026）：单目标 VOT 定位 RL，非 MOT 关联。
- Ground-R1 / UniVG-R1 / R1-Track / ReasoningTrack：视觉 grounding /
  tracking 的 GRPO/SFT 类工作，主要针对单目标或 grounding 任务，
  与本项目“关联决策未来效用”非直接等价；可借鉴 reward
  normalization、rule-based reward、KL 保持，但禁止无条件迁移。
- 结论：**未找到 MOT 关联的 counterfactual future-utility RL 官方实现**。
  Stage L2 第一主路线仍是 supervised utility learning / preference
  learning（见 `reports/l2_novelty_collision_audit.md`）。

## 12. 总表

| Method | Year/Venue | 官方 repo verified | Commit | License | 训练用未来 | 推理 causal | 轨迹级目标 | Counterfactual | RL | 采用/不采用 |
|---|---|---|---|---|---|---|---|---|---|---|
| TDLP | 2025/26 arXiv | ✅ | 50344b92 | MIT | 未来1帧GT | ✅ | 否 | 否 | 否 | 借鉴结构，不采用目标 |
| SambaMOTR | ICLR 2025 | ✅ | f1c139a6 | MIT | clip传播 | ✅ | 隐式 | 否 | 否 | 借鉴记忆编码 |
| TRACT | ICCV 2025 | ✅ | 19f01d72 | 无 | 否 | ✅ | 轨迹一致 | 否 | 否 | 借鉴特征聚合 |
| UniTrack | ICLR 2026 | ✅ | afdd9869 | MIT | clip窗口 | ✅ | 平滑 | 否 | 否 | 借鉴可微轨迹正则 |
| Path Consistency | CVPR 2024 | ✅ | f4b7d26d | Apache-2.0 | 多路径未来 | ✅ | 路径一致 | 否 | 否 | 可选辅助 |
| QuoVadis | NeurIPS 2022 | ✅ | 241233a5 | LICENSE.md | 未来轨迹回归 | ✅ | 预测 | 否 | 否 | 借鉴 horizon |
| FDTA | CVPR 2026 | ✅ | b3b3b778 | 有 | clip | ✅ | 否 | 否 | 否 | 借鉴判别 loss |
| HATReID-MOT | 2025 | ✅ | 3eb440c2 | 有 | 否 | ✅ | 历史变换 | 否 | 否 | 借鉴 history 编码 |
| HNCD-MOTR | 2025/26 | ✅ | 2026-05-31 | 无 | 否 | ✅ | 否 | 否 | 否 | 借鉴 privileged 训练 |
| CAMELTrack | 2025 | ✅ | 46a74bb | 有 | 否 | ✅ | 否 | 否 | 否 | 已用于 L1-D |
| MOTIP(-2) | CVPR 2025 | ✅ | ffc0e905 / 012856c1 | Apache-2.0 | clip | ✅ | 否 | 否 | 否 | 已用于 L1-C/D |
| LG-Track / LLTrack | 2023/25 | ✅ | 432a467 / 2ab7994 | MIT/有 | 否 | ✅ | 否 | 否 | 否 | 已用于 L1-D |
| MeMOTR | ECCV 2022 | ✅ | 仓库 HEAD | 有 | clip | ✅ | memory | 否 | 否 | 已审计 |

## 13. 审计结论

所有已核实方法中：

1. 只有 TDLP、QuoVadis、SambaMOTR、UniTrack、Path Consistency
   在**训练期利用多帧/未来信息**；
2. 它们的监督都是 **下一帧匹配正确性 / 轨迹平滑 / 路径一致性 /
   未来位置回归**，没有一个是
   “对当前关联 action 做 counterfactual long-horizon trajectory
   utility 评估并蒸馏给 online policy”；
3. 推理全部保持 online causal。

因此 Stage L2 的核心问题（local correctness vs trajectory utility）
在本批审计中未发现直接等价官方方法；新颖性声明必须限定为
“未发现直接等价方法”，不得使用 “first”。
