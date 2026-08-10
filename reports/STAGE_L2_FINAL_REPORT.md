# Stage L2 Final Report

日期：2026-08-10。项目：LocateMOT。

## 1. Executive Summary

Stage L2 验证了核心科学问题（local association correctness 与 future
trajectory utility 不同构），但 **Oracle Headroom 不足**：

- 单事件 counterfactual 窗口 headroom 存在但很小
  （DanceTrack H32 +0.74pp、BDD H16 +1.01pp windowed AssA）；
- 端到端 privileged greedy oracle 在整视频 TrackEval AssA 上
  无正收益（DanceTrack +0.02/+0.06pp，BDD 均值 −0.88pp，
  MOT17 −2.32pp），IDSW 全部变差；
- 因此判定 `L2_ORACLE_HEADROOM_LOW`，**不启动大型 Trajectory
  Utility Model 训练**，按任务书直接进入失败分析与最终报告。

保留产出：四域公平 baseline 矩阵、TrackEval 目标审计、2025–2026
官方代码审计、local-vs-future mismatch 统计、历史污染审计、可复现的
oracle 工具链（replay 与官方基线 100% 一致，windowed AssA/IDF1 与
官方 TrackEval 完全一致）。

## 2. Unified MOT Goal

目标：一个核心模型、一个主 checkpoint、无数据集专属 head 的 Unified
MOT，覆盖 DanceTrack、MOT17、MOT20、BDD100K、TAO（TAO 因 cache 缺失
仅文档记录）。训练用未来，推理严格 online causal。

## 3. Why L1DK Is Only a Baseline

L1DK = 0.4 IoU + 0.2 PBD + 0.4 Kalman-motion 线性融合（thr 0.25），
无训练参数。它是当前最强统一 AC 基座（macro AssA 0.4062），但只是
手工 cue 融合，不具备论文级创新性；Stage L2 的目标是在其上学习
“当前关联决策的未来轨迹效用”，而不是继续调权重。

## 4. L1-A → L1-D Evidence Chain

- L1-B Identity Adapter：跨域/LODO 不能稳定超过 raw representation；
- L1-C UAF from-scratch decoder：50k 步 DanceTrack AssA≈0.133，
  IDSW≈26,804，失败；
- L1-C LoRA：grounding/PBD discrimination/association 全部下降
  （`LORA_PBD_EXTRACTION_SUPPORTED + LORA_PBD_DEGRADED`）；
- L1-D EGRA：局部修正 precision 0.898 / coverage 0.782，但 DanceTrack
  AssA 0.4165→0.3993 且跨域方向不一致；
- 结论链：LOCAL CORRECTION SUCCESS ≠ TRAJECTORY ASSOCIATION SUCCESS，
  这正是 Stage L2 的出发点。

## 5. Core Scientific Problem

传统监督回答“candidate j 是否是 track i 当前帧的正确 match”
（local correctness）；Stage L2 想回答“现在做这个 assignment 对未来
identity persistence / IDSW / fragmentation / purity 的长期后果”
（trajectory utility）。

## 6. 2025–2026 Literature Audit

完整审计见 `docs/l2_reference_audit.md`。重点：

| 方法 | 结论 |
|---|---|
| TDLP（2025/26） | 训练用下一帧 GT link prediction；无轨迹效用 |
| SambaMOTR（ICLR 2025） | 自回归 track query；监督仍是当前帧 |
| TRACT（ICCV 2025） | 轨迹级特征聚合/一致性；无未来效用 |
| UniTrack（ICLR 2026） | 轨迹平滑 hinge loss；无关联未来后果 |
| Path Consistency（CVPR 2024） | 路径一致性自监督；无 GT 身份效用 |
| QuoVadis（NeurIPS 2022） | 未来轨迹位置回归；非身份效用 |
| MOTIP/MOTIP-2、CAMELTrack、FDTA、HATReID-MOT、HNCD-MOTR | 局部/历史判别 |

## 7. Novelty Collision Audit

**NO DIRECTLY EQUIVALENT VERIFIED METHOD FOUND**（详见
`reports/l2_novelty_collision_audit.md`）。注意：这是“未发现”，
不是 “first”；论文不得声称首创。

## 8. Strong Baseline Matrix

同一 AC 协议（固定 boxes/scores/帧数，只改 IDs），官方 TrackEval：

| Variant | DanceTrack AssA | MOT17 AssA | MOT20 AssA | BDD AssA | Macro |
|---|---:|---:|---:|---:|---:|
| C0 IoU | 0.3899 | 0.4504 | 0.2071 | 0.3044 | 0.3380 |
| C1 Motion | 0.4193 | 0.5530 | 0.2869 | 0.3019 | 0.3903 |
| C2 PBD | 0.1555 | 0.0975 | 0.0740 | 0.1659 | 0.1232 |
| C3 IoU+PBD | 0.3934 | 0.3856 | 0.2171 | 0.2255 | 0.3054 |
| **L1DK base** | **0.4165** | **0.6010** | 0.2779 | **0.3292** | **0.4062** |
| L1DK_d03 (EGRA) | 0.3992 | 0.5922 | **0.2864** | 0.2841 | 0.3905 |

IDSW（DanceTrack/MOT17/MOT20/BDD）：L1DK 2558/276/3736/12149；
EGRA 2598/274/2408/11151。

**BEST_STRONG_BASE = L1DK base**。详见 `reports/l2_baseline_matrix.md`。

## 9. TrackEval AssA Objective Audit

`docs/l2_trackeval_objective_audit.md`：

- AssA(alpha)：全序列 (gt_id, trk_id) 共现计数 → Jaccard →
  按匹配次数加权平均；`AssA(0)` 实际是 alpha=0.05；
- IDF1：全序列最优 ID 映射下的 F1；
- IDSW：CLEAR 逐帧匹配下的身份切换计数；
- 三者都是轨迹级 ID 共现统计，与单帧 pair correctness 不同构。

本项目实现 windowed AssA/IDF1 与官方公式一致；整视频校验
（DanceTrack 0004/0005、MOT17-04）与官方 TrackEval 数值完全一致。

## 10. Online State Construction

`tools/run_l2_oracle.py` 用与 baseline 完全相同的 AC shell 重放
L1DK base（Kalman 生命周期、Hungarian、birth/lost/terminate），
记录每个冲突帧的完整 causal 状态：

- track：box/prev_box/age/hits/lost_age/gap/Kalman 状态/ref&anchor PBD/
  GT 历史（审计用）；
- candidate：box/PBD/gen/GT（审计用）；
- base 矩阵、base assignment、冲突组件；
- 预测侧污染统计（purity proxy、past IDSW、fragments）。

重放与官方基线 100% 一致（MOT17-04-SDP 3589/3589；
DanceTrack 因参考文件 ID 全局偏移无法逐字节比对，行为等价）。

## 11. Historical Identity Contamination

详见 `reports/l2_history_contamination_audit.md`。

- DanceTrack val：EGRA 修正 1,295 次，helpful 334（25.8%）/
  harmful 264（20.4%）/ same_gt 213 / other 484；57% 修正发生在
  past IDSW≥4 轨迹，且该桶 helpful 率更低（19.8%）；
- BDD：修正 1,799 次，helpful 163（9.1%）/ harmful 247（13.7%）/
  same_gt 748 / other 641；59.8% 位于 purity<0.5 重度污染轨迹，
  且该桶 harmful（175）远多于 helpful（74）。

结论：EGRA 的修正大量落在已污染轨迹上，而这些轨迹的历史错误在固定
ID 符号的 AC 协议下不可恢复。

## 12. Local-vs-Future Mismatch

详见 `reports/l2_local_vs_future_mismatch.md`。

- DanceTrack H32：219/1000 事件 future-best ≠ base；
  local_correct_future_bad 128 / local_wrong_future_good 60；
- BDD H16：460/745 事件 future-best ≠ base；
  local_correct_future_bad 173 / local_wrong_future_good 110。

结论：mismatch 真实存在，但幅度小，不足以转化为端到端收益。

## 13. Counterfactual Action Space

每个冲突组件生成 6–8 个候选 action：base、GT-local、top-k 分数匹配、
随机全局匹配（大组件）、最差匹配、all-new；`complete_assignment`
补全为合法全局一对一分配。

## 14. Counterfactual Rollout

强制执行 action 后冻结 base policy，rollout H∈{4,8,16,32}（BDD
5fps 用 {2,4,8,16}），逐帧应用匈牙利并更新 Kalman，窗口计算
TrackEval 同款效用。

## 15. Future Trajectory Utility Definition

Primary：`U_H = windowed AssA`（窗口 [t+1, t+H]）。
Auxiliary：windowed IDF1、IDSW。全部按 TrackEval 公式实现并验证。

## 16. Multi-Horizon Design

效用排序随 horizon 收敛：DanceTrack H16 vs H32 pair agreement 73.7%，
BDD H8 vs H16 67.7%；最短 horizon 一致性低（27.6%/19.0%）。
说明多 horizon 必要，但收益幅度未随 horizon 变大而足够。

## 17. Oracle Headroom

**这是本阶段最关键结果。**

单事件窗口 headroom（隔离评估）：

- DanceTrack：H4 +0.38pp → H32 +0.74pp（mean windowed AssA），
  frac_better 7.5%→21.9%；
- BDD：H2 +0.96pp → H16 +1.01pp，frac_better 18%→62%。
- MOT17：H4 +1.61pp → H32 +2.27pp，frac_better 55%→71%；
- MOT20：H4 +1.76pp → H32 +1.76pp，frac_better 66%→75%。

注意：MOT17 单事件窗口 headroom 反而最大（H32 +2.27pp），但端到端
为 −2.32pp——这是“窗口效用与全局轨迹质量不同构”的最强直接证据。

端到端 privileged greedy oracle（整视频 TrackEval 同款 AssA）：

| 视频 | base AssA | oracle AssA | gain | IDSW（base→oracle） |
|---|---:|---:|---:|---:|
| dancetrack0004 | 0.1702 | 0.1704 | +0.02pp | 149→151 |
| dancetrack0005 | 0.4562 | 0.4568 | +0.06pp | 68→74 |
| BDD-…58 | 0.1319 | 0.1582 | +2.62pp | 105→121 |
| BDD-…88 | 0.2439 | 0.1963 | −4.76pp | 30→36 |
| BDD-…98 | 0.2352 | 0.2302 | −0.50pp | 125→126 |
| MOT17-02-SDP | 0.1830 | 0.1598 | −2.32pp | 783→866 |

结论：**整视频 AssA headroom < 0.1pp（DanceTrack），不满足 1pp 门槛；
IDSW 无改善甚至变差。** 判定 `L2_ORACLE_HEADROOM_LOW`。

## 18. Oracle Failure Cases

- BDD/MOT17 上 greedy oracle 反而显著变差：局部 windowed AssA 最优
  的 action 在全局 ID 统计上造成更多碎片（IDSW 上升）；
- MOT17 是极端案例：单事件窗口 headroom +2.27pp（H32），端到端
  整视频 −2.32pp，证明窗口最优选择在全局 ID 统计上系统性有害；
- 大量组件 action 经 base 再优化后与 base 等价（效用相同事件占比高），
  真正“可行动”的 headroom 比单事件均值更小。

## 19. 2025–2026 Method Evidence for Model Design

若未来恢复该方向，模型设计依据（已读官方代码）：

- set-level 竞争：TDLP / CAMELTrack；
- 轨迹历史编码：TDLP tracklet history、TRACT TFA、HATReID-MOT；
- 长时记忆：SambaMOTR、MeMOTR；
- 轨迹一致性正则：UniTrack、Path Consistency。

## 20. Trajectory Utility Model

设计了 TUM（4 层/256 hidden，action-conditioned set transformer，
per-horizon utility head），**未训练**（Gate 1 未通过）。设计文档：
`reports/l2_utility_model.md`。

## 21. Causal Student / Policy

未实现（Gate 1 未通过）。设计：推理时对冲突组件生成候选 action，
TUM 打分后取 argmax 执行。

## 22. Why Inference Remains Online

即使训练成功，推理仅使用当前帧 causal 状态 + 候选 action，未来只出现
在训练标签中；但本阶段未走到该步骤。

## 23. Training Objective

未执行。设计目标：oracle windowed AssA 回归 + listwise ranking，
multi-horizon 联合训练。

## 24. Dataset Sampling

未执行训练。oracle 数据：DanceTrack val 25 视频 1,000 冲突事件、
BDD 30 视频 745 事件、MOT17/MOT20 全部视频（小样本）。

## 25. Model Size / Compute

未训练。计划 TUM-small ~1.5–3M 参数单卡 pilot；通过后 5–15M DDP 2–4 卡。

## 26. Utility Prediction Results

未执行（Gate 1 未通过）。

## 27. Ranking Results

未执行（Gate 1 未通过）。已记录 oracle 效用排序的 horizon 一致性
（§16），可作为未来训练验证基准。

## 28. Regret

未执行训练。oracle 单事件 mean regret = mean headroom
（DanceTrack H32 0.74pp、BDD H16 1.01pp）；端到端实现后为负或近零。

## 29. Association-Controlled Main Results

本阶段未产生新模型结果；baseline 矩阵见 §8。Stage L2 方法无 AC 主结果。

## 30. DanceTrack

oracle headroom 0.02–0.06pp（整视频 AssA），IDSW 149→151 / 68→74。

## 31. MOT17

oracle（1 视频）：−2.32pp AssA，IDSW 783→866；无正收益。

## 32. MOT20

oracle 事件极少（2 视频，冲突少），未做端到端；单事件 headroom
未单独统计（并入训练域时不会改变结论）。

## 33. BDD Multi-Class

oracle（3 视频）：均值 −0.88pp AssA；1 正 2 负；IDSW 变差。
BDD 为多类域，AC 协议仅用 pedestrian GT。

## 34. TAO-Compatible

TAO cache 缺失（4,200 帧无 .complete），本阶段仅文档记录；
未运行 oracle。

## 35. Macro Unified Result

无 Stage L2 统一模型结果；宏结果为 L1DK base（macro AssA 0.4062）。

## 36. One-Checkpoint Verification

未执行（无新 checkpoint）。

## 37. Leave-DanceTrack-Out

未执行（Gate 1 未通过，按任务书不执行）。

## 38. Leave-Multiclass-Out

未执行。

## 39. Local CE vs Future Utility

核心消融未执行（无训练）。替代证据：local-vs-future mismatch 统计
（§12）与 oracle headroom（§17）。

## 40. Counterfactual Ablation

未执行。

## 41. Multi-Horizon Ablation

已完成 oracle 层面：horizon 越长 headroom 越大但端到端仍不足；
排序一致性随 horizon 收敛（§16）。

## 42. Future Teacher Ablation

未执行（无 teacher）。

## 43. Path Consistency Auxiliary Ablation

未执行（Gate 1 未通过）。

## 44. Base-Preservation

oracle 的 base action 始终在候选集中；多数事件最优 action 就是 base
（DanceTrack 78.1% 事件 H32 无更优 action）。

## 45. ID Switch Chain Analysis

端到端 oracle 的 IDSW 全部变差（DanceTrack 149→151、68→74；
BDD 260→283；MOT17 783→866）：窗口最优的局部选择在 ID 链上造成
更多切换，与 L1-D 的 IDSW 结论一致。

## 46. Long-Horizon Trajectory Analysis

H32 窗口 headroom（0.74pp）大于 H4（0.38pp），说明 horizon 越长
效用差异越明显，但端到端仍然不可实现；原因见 §18。

## 47. Full Tracker Results

未执行（AC 未通过，按任务书禁止）。

## 48. RL Experiment

未执行。RL 参考记录：`docs/future_rl_reference.md`（Ground-R1、
UniVG-R1、RELO、Query-MARFT 无官方代码等）。

## 49. Utility-Aligned LoRA

未执行（L1-C 已证明普通 grounding LoRA 失败；
L1-D 状态 `LORA_PBD_EXTRACTION_SUPPORTED + LORA_PBD_DEGRADED`）。

## 50. Why Not MOTIP / CAMELTrack / TDLP?

客观说明：这些方法学习当前帧 link/ID 判别（local correctness），
是 Stage L2 的对比基线与结构参考；本阶段提出的新增问题是
“future trajectory consequence supervision”。文献审计未发现该
新增问题已被等价实现，但 oracle 证据表明在 L1DK+AC 协议下该问题
无可学习正收益。

## 51. Why Not L1DK Alone?

L1DK 是手工线性融合，无学习能力；但它已经接近 AC 协议下该
状态空间的局部最优，导致未来效用学习的可修空间很小。

## 52. Why Not EGRA?

EGRA 已证明局部修正与轨迹级指标不同构（L1-D + 本阶段污染审计），
且其 delta 上限 0.6 无法修复多数身份错误。

## 53. Does Future Supervision Really Matter?

理论上是的（mismatch 统计支持），但实验上：在 L1DK base + AC 协议下，
即使未来监督完全正确（oracle），整视频收益 < 0.1pp，无法构成
“future supervision matters”的实证 claim。

## 54. Does It Generalize Across Domains?

未验证（无学生模型）；oracle 端到端在 BDD/MOT17 为负，跨域方向
不支持。

## 55. Multi-Class Evidence

BDD oracle 事件已生成（745 事件），但未训练；无多类正收益证据。

## 56. Failure Cases

见 `reports/l2_failure_analysis.md`：

1. 基座短窗口接近最优（DanceTrack H4 base AssA 0.9734）；
2. 窗口效用与全局 ID 统计不同构（IDSW 恶化）；
3. 动作空间经 base 再优化后趋同；
4. 历史污染不可在短窗口修复。

## 57. Runtime

- oracle 单事件：DanceTrack ~0.5s/event（6 action × 32 帧 rollout）；
- 端到端：DanceTrack ~5 分钟/视频（1,200 帧 × 3 组件 × 6 action
  × H16），BDD 短视频 ~10–20 秒；
- 全程 CPU，未占用 GPU（符合 4 卡约束）。

## 58. GPU / RAM

- GPU：未使用（oracle 全部 CPU）；
- RAM：峰值 ~10–15GB（两个 oracle 并发 + raw pkl），远低于
  MemTotal 125GB 的 25% 要求；
- 磁盘：事件 pkl 约 1.1GB（DanceTrack 0.5GB + BDD 0.6GB），
  /data1 剩余 157GB，充足。

## 59. Scientific Interpretation

本阶段最重要的科学发现是：**在固定检测/固定 ID 符号的 AC 协议下，
L1DK base 的关联状态空间几乎没有“未来可修”空间；local correctness
与 trajectory utility 的 mismatch 是真实的，但幅度不足以支撑
future-utility 训练的正收益。** 这解释了为什么 L1-D 的局部修正成功
而全局失败，并说明单纯改变训练目标（local→future）不会自动产生
ICLR 级改进。

## 60. ICLR Readiness Audit

客观评价：

| 维度 | 评价 |
|---|---|
| Novelty | 中：问题定义与文献审计完整，无直接等价方法 |
| Technical Quality | 中：replay 与官方基线 100% 一致，windowed 指标与官方一致 |
| Empirical Strength | **低：无端到端正收益，oracle headroom < 0.1pp** |
| Generalization | 低：BDD/MOT17 端到端为负 |
| Clarity of Scientific Question | 高：local vs future mismatch 清晰 |

结论：**不满足 ICLR 实证要求（A–H 中 B/C/D/E/F/G/H 均不满足）**。

## 61. Claim Boundary

- 可声称：local correctness 与 future windowed utility 存在可测量
  的 mismatch（DanceTrack 21.9%、BDD 61.7% 冲突事件）；
- 不可声称：future-utility 训练能改善统一 MOT；
- 不可声称：方法首创（只能写“未发现直接等价方法”）。

## 62. Stage Decision

```text
主状态：L2_ORACLE_HEADROOM_LOW
ICLR readiness：NOT_READY
```

## 63. Next Single Recommendation

**改变效用/协议再验证**：若继续该方向，优先实验
“允许 ID 重映射/轨迹恢复的 full-tracker 协议 + 整序列 ID-mapping
效用（IDF1 式全局映射 + IDSW 惩罚）”，而不是在当前 AC 协议上堆
模型容量；否则终止该方向，回到强 baseline 的 full-tracker 工程路线。

## 64. Important Paths

本报告已自包含：以下每个产物的**完整原文**嵌入本报告“附录 A–L”，
可直接复制整份报告，无需再打开其他 md。

- 本阶段产物（附录索引）：
  - 附录 A：`docs/l2_reference_audit.md`（2025–2026 官方代码审计）
  - 附录 B：`docs/l2_trackeval_objective_audit.md`（TrackEval AssA/IDF1 审计）
  - 附录 C：`docs/future_rl_reference.md`（未来 RL/GRPO 参考）
  - 附录 D：`reports/l2_baseline_matrix.md`（四域 AC baseline 矩阵）
  - 附录 E：`reports/l2_novelty_collision_audit.md`（Novelty Collision 审计）
  - 附录 F：`reports/l2_counterfactual_oracle.md`（Counterfactual Oracle）
  - 附录 G：`reports/l2_oracle_headroom.md`（Oracle Headroom / Gate 1）
  - 附录 H：`reports/l2_local_vs_future_mismatch.md`（Local vs Future Mismatch）
  - 附录 I：`reports/l2_history_contamination_audit.md`（历史污染审计）
  - 附录 J：`reports/l2_utility_model.md`（TUM 设计）
  - 附录 K：`reports/l2_failure_analysis.md`（失败分析）
  - 附录 L：`reports/STAGE_L2_GPT_HANDOFF.md`（GPT Handoff）
- 工具（代码不在本报告内，源码在仓库）：
  - `tools/run_l2_oracle.py`
  - `tools/l2_analyze_oracle.py`
  - `tools/l2_history_contamination.py`
  - `tools/l2_endtoend_oracle.py`
  - `tools/train_l2_tum.py`（未使用，保留设计）
- 数据：`outputs/l2/oracle/`（events pkl、analysis json、e2e json、
  contamination json）

## 65. Git Commit

提交信息：`Stage L2 complete: counterfactual future-utility learning for unified MOT`
（含本报告与全部 L2 文档/工具；commit hash 见 Git 记录）。


## 附录 A — 2025–2026 官方代码审计

> 来源文件：`docs/l2_reference_audit.md`（已嵌入本报告）

### Stage L2 — 2025/2026 参考实现审计

日期：2026-08-10。
范围：Stage L2 要求的“训练用未来 / 轨迹级效用 / counterfactual /
future-aware association”方向，全部实际 clone 并阅读官方代码；不依据
摘要或博客转述。仓库固定 commit 记录于 `docs/reference_repository_inventory.md`
（本文件只列出 L2 新增与重点仓库）。

#### 0. 审计问题模板

对每个方法回答：

- 官方仓库是否验证（URL、commit、license）
- 训练是否使用未来帧/未来 GT；推理是否 causal
- 关联表示（pair embedding / link prediction / ID classification / query）
- 训练目标（local correctness / trajectory-level / future utility）
- 是否 counterfactual rollout；是否 RL
- 与 Stage L2 的关系：可借鉴 / 不采用及原因

#### 1. TDLP（arXiv 2512.22105，2025/2026）

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

#### 2. SambaMOTR（ICLR 2025 Spotlight）

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

#### 3. TRACT（ICCV 2025）

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

#### 4. UniTrack（ICLR 2026）

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

#### 5. Path Consistency（CVPR 2024）

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

#### 6. QuoVadis（NeurIPS 2022）

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

#### 7. FDTA（CVPR 2026）

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

#### 8. HATReID-MOT（arXiv 2503.12562）

- 官方仓库：本地 `references/association_2025_2026/HATReID-MOT`
- Commit：`3eb440c2`（2026-07-23）；License：仓库 LICENSE
- 已读文件：`README.md`（HAT-SORT：历史感知 ReID 特征变换）
- 问题：用轨迹历史把 ReID 特征变换到更可分的子空间。
- 局部/轨迹级：轨迹历史 → 特征变换（局部关联）。
- 训练用未来：无。
- 可借鉴：track 级特征变换（history-aware）可作为 TUM 输入编码。
- 不采用：无未来效用。

#### 9. HNCD-MOTR（2025/2026）

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

#### 10. MOTIP / MOTIP-2 / CAMELTrack / LG-Track / LLTrack / MeMOTR

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

#### 11. RL / 决策学习方向（仅记录，Stage L2 不先启动 RL）

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

#### 12. 总表

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

#### 13. 审计结论

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


## 附录 B — TrackEval AssA/IDF1 目标审计

> 来源文件：`docs/l2_trackeval_objective_audit.md`（已嵌入本报告）

### Stage L2 — TrackEval AssA / IDF1 Objective Audit

日期：2026-08-10。审计对象：`references/TrackEval-official`
（commit `HEAD` 以仓库内 `.git` 为准；本文依据实际源码推导，不凭印象）。

#### 1. 为什么必须先审计

Stage L2 的 Future Trajectory Utility 必须与最终评测目标对齐。
评测协议是 Association-Controlled（固定 boxes/scores/帧数，只改 IDs），
因此真正受关联影响的是：

- HOTA AssA（关联质量）；
- IDF1（全局身份匹配质量）；
- IDSW（身份切换次数）；
- Frag（轨迹断裂次数）。

DetA 在 AC 协议下固定，不作为关联收益。

#### 2. 数据预处理：ID 重标号

`trackeval/datasets/mot_challenge_2d_box.py::get_preprocessed_seq_data`：

1. 只保留 pedestrian 类，移除 distractor 匹配的 tracker det、ignore 区域 det、
   zero_marked GT det（MOTChallenge 规则）；
2. 对每个序列收集 `unique_gt_ids` / `unique_tracker_ids` 并重标号为
   **contiguous 整数**（`np.unique` 顺序）：

```python
unique_gt_ids = np.unique(unique_gt_ids)
unique_tracker_ids = np.unique(unique_tracker_ids)
...
new_id = dict(zip(unique_gt_ids, range(len(unique_gt_ids))))
```

含义：TrackEval 内部不关心外部 track ID 的具体数值，只关心
**每个预测 ID 与每个 GT ID 共现的帧集合**。任何关联策略的效用最终都
反映为“预测轨迹 ID 与 GT 轨迹 ID 的共现统计”。

#### 3. HOTA AssA 的精确计算（`trackeval/metrics/hota.py`）

##### 3.1 全局共现计数

对每一帧 t，用 IoU 相似度矩阵 `similarity` 计算归一化相似度
`sim_iou`（Jaccard 形式），累加到 `potential_matches_count[gt_id, trk_id]`；
同时累加每个 GT ID 的出现次数 `gt_id_count` 与每个 tracker ID 的出现次数
`tracker_id_count`。

##### 3.2 全局对齐分数

```python
global_alignment_score = potential_matches_count / (
    gt_id_count + tracker_id_count - potential_matches_count)
```

即：每个 (gt_id, trk_id) 对的 **co-occurrence Jaccard 分数**。

##### 3.3 逐帧匹配（Hungarian）

每帧的分数矩阵 = `global_alignment_score * similarity`，
`linear_sum_assignment(-score_mat)` 得到唯一匹配；仅当
`similarity >= alpha` 时计入该 alpha 的 TP。

##### 3.4 AssA(alpha)

```python
matches_count = matches_counts[a]                     # (num_gt_ids, num_tracker_ids)
ass_a = matches_count / max(1, gt_id_count + tracker_id_count - matches_count)
AssA[a] = sum(matches_count * ass_a) / max(1, HOTA_TP[a])
```

等价解释：对每对 (gt_id, trk_id)，先算二者在整个序列中
**共同匹配的检测次数**与 Jaccard 分母的比值 `ass_a`，再按共同匹配次数
加权平均，分母是总匹配次数 `HOTA_TP[a]`。

注意：`HOTA_AssA(0)` 实际是 `array_labels[0] = 0.05`（不是 alpha=0），
即严格匹配阈值最低档。HOTA(0)/AssA(0)/DetA(0) 均取 alpha=0.05。

##### 3.5 跨序列合并

`AssA` 跨序列用 `HOTA_TP` 加权平均；IDF1 的整数计数跨序列直接求和。
因此最终报告值是“按序列长度加权的整体统计”，不是序列平均。

#### 4. IDF1 的精确计算（`trackeval/metrics/identity.py`）

1. 每帧统计 `similarity >= 0.5` 的 (gt_id, trk_id) 共现次数
   `potential_matches_count`；
2. 构造带 dummy 的全局分配矩阵，最小化
   `IDFP = tracker_id_count - matches`、`IDFN = gt_id_count - matches`
   （Hungarian，允许 gt 只分给一个 tracker id）；
3. `IDTP = total_gt_dets - IDFN`；
4. `IDF1 = IDTP / (IDTP + 0.5*IDFP + 0.5*IDFN)`。

含义：IDF1 是 **整个序列上的最优 ID 映射** 下的 F1，仍然只依赖
ID 共现计数。关联动作改变的是 tracker ID 的时间分配，因此 IDF1 能
反映轨迹级身份质量。

#### 5. IDSW 与 Frag 的精确计算（`trackeval/metrics/clear.py`）

- 每帧先匹配（先最小化 IDSW，再最大化 MOTP 的 Hungarian）；
- `IDSW`：当某个 GT ID 在**任意历史帧**匹配过的 tracker ID 与当前帧
  不同时，计一次（`prev_tracker_id` 记录上次匹配的 tracker id）；
- `Frag`：GT 从匹配到不匹配再到匹配，每次“重新进入”计一次。

因此 IDSW 本质是 **轨迹身份在时间上的切换次数**，对局部单帧纠正非常
敏感：即使单帧分配在“局部 IoU 最优”意义下正确，只要它与该轨迹历史
身份不同，就会引入 IDSW。

#### 6. 对 Stage L2 Future Utility 的设计约束

由上述源码可导出以下设计规则：

1. **Utility 必须以轨迹 ID 共现统计为对象**：
   不能只用单帧 pair accuracy；必须把“该 action 导致的 track ID 序列”
   与 GT ID 序列比较。
2. **Windowed utility 的排序应与 TrackEval AssA 一致**：
   在窗口内复现 `matches_count` → Jaccard → 加权平均的公式，即可得到
   `windowed_assA`。这与官方 AssA 在窗口边界上存在截断差异（官方是
   全序列匹配 + 逐帧 Hungarian），需要作为已知近似。
3. **IDF1 与 AssA 可能给出不同排序**：AssA 按检测加权、IDF1 按 ID 分配
   优化；两者都记录，但 AC 主指标以 AssA 为准（HOTA 的组成项）。
4. **本地正确 ≠ 未来效用**：单帧 GT-edge 正确不会自动转化为
   AssA/IDF1/IDSW 收益，因为后者是轨迹级统计。这正是 Stage L2 要
   用 Counterfactual Rollout 证明的核心机制问题。

#### 7. 采用的设计（Stage L2 Oracle 阶段）

Primary utility：

```text
U_H(s_t, a) = windowed_assA(window = [t, t+H))
```

其中 windowed_assA 按官方公式实现：

```text
matches_count[g, p] = Σ_{frames in window, alpha-matched pairs}
    (gt_id=g, pred_id=p 的匹配计数)
ass_a[g,p] = matches_count[g,p] / (gt_count[g] + pred_count[p] - matches_count[g,p])
U = Σ matches_count * ass_a / Σ matches_count
```

同时记录 windowed IDF1、window IDSW、轨迹 purity（主导 GT ID 占比）
作为辅助证据，避免人工 reward 设计偏差。

约束：窗口内匹配采用与 TrackEval 相同的逐帧 Hungarian +
`global_alignment_score` 动态更新；窗口截断导致的边界效应在报告中
单独量化（headroom 验证时同时给整序列 TrackEval 数字）。

#### 8. 已核对文件

- `references/TrackEval-official/trackeval/metrics/hota.py`
  （`eval_sequence`、`combine_sequences`、`_compute_final_fields`）
- `references/TrackEval-official/trackeval/metrics/identity.py`
  （`eval_sequence`、`_compute_final_fields`）
- `references/TrackEval-official/trackeval/metrics/clear.py`
  （`eval_sequence` 的 IDSW/Frag 计数）
- `references/TrackEval-official/trackeval/datasets/mot_challenge_2d_box.py`
  （`get_preprocessed_seq_data` 的 distractor 过滤与 ID 重标号）

#### 9. 结论

AssA / IDF1 / IDSW 全部是 **轨迹级身份共现统计**，与单帧 pair
correctness 不同构。Stage L2 的 future utility 采用 windowed
association-aware 公式（以官方 AssA 为排序基准），并通过 oracle
headroom 实验验证窗口效用与整序列 TrackEval 的一致性。


## 附录 C — 未来 RL/GRPO 参考记录

> 来源文件：`docs/future_rl_reference.md`（已嵌入本报告）

### Stage L2 — 未来 RL/GRPO 参考记录（仅调研，不启动训练）

日期：2026-08-10。
规则：Stage L2 禁止启动 RL/GRPO。本文只记录已核实的官方代码、
训练资源需求与可迁移模块，供后续 Stage 使用。

#### 1. 视觉 grounding GRPO

- Ground-R1（arXiv 2503.24358）：基于 Qwen2.5-VL 的 visual grounding
  RL，使用 GRPO + grounding 规则奖励；官方代码
  `github.com/zehao-wang/Ground-R1`。
- UniVG-R1（arXiv 2505.20466）：统一视觉 grounding 的 RL 框架，
  官方代码 `github.com/zhongyingpeng/UniVG-R1`。
- 可迁移模块：rule-based grounding reward、GRPO 的 reward
  normalization、KL 保持、多模态 LoRA/RL 的训练配方。
- 不可迁移：其 reward 是单图 grounding 框匹配，不是轨迹身份效用。

#### 2. 目标跟踪 RL

- RELO（ICML 2026，arXiv 2605.07379）：RL to localize 的单目标 VOT，
  官方代码 `github.com/Multimedia-Analytics-Laboratory/RELO`。
- MATT-Diff / AOT-ARL：主动目标跟踪（相机控制）RL，与 MOT 关联无关。
- Query-MARFT（2026，Pattern Recognition）：端到端 MOT 的
  多 agent RL fine-tuning（DetAgent/AssocAgent/UpdateAgent/CorrAgent，
  Flexible Markov Game）；**未找到官方公开代码**（仅论文）。
- 结论：MOT 关联级 RL 官方实现稀缺；Query-MARFT 是最接近的
  “关联策略 RL”，但没有 counterfactual trajectory utility。

#### 3. 结构化框奖励 / ID 一致性奖励 / 可验证轨迹奖励

- 结构化框奖励：Ground-R1/UniVG-R1 提供框级规则奖励
  （IoU 阈值分档），可作为“框奖励”参考。
- ID 一致性奖励：未找到直接官方实现；本项目若进入 RL，需自行定义
  windowed AssA / IDSW 作为 verifiable trajectory reward（见
  `docs/l2_trackeval_objective_audit.md`）。
- 可验证轨迹奖励：可将 TrackEval AssA/IDF1/IDSW 的窗口版本作为
  rule-based reward；这正是 Stage L2 supervised utility 的同一
  目标函数，RL 只是优化器差异。

#### 4. 训练资源需求估计

- Ground-R1 类 7B VLM RL：通常需要 8×A100（80G）+ 数天；
  本项目 4×40G 不满足直接复刻。
- 若本项目 RL 化：建议 0.5–50M 参数轻量 utility/policy 模型，
  trajectory rollout 在 CPU/低并发完成，GPU 只训练小模型；
  单卡 4×40G 足够。

#### 5. 进入 RL 的条件（Stage L2 设定）

1. supervised counterfactual utility learning 已证明有效；
2. 仍存在明显 exposure bias / long-horizon mismatch；
3. 有官方 RL 框架可复用依据（如 GRPO 的 reward normalization）。

否则保持 supervised utility / preference learning。


## 附录 D — 四域 AC Baseline 矩阵

> 来源文件：`reports/l2_baseline_matrix.md`（已嵌入本报告）

### Stage L2 — 四域 Association-Controlled Baseline Matrix

日期：2026-08-10。
协议：所有方法使用完全相同的候选集（boxes/scores/features）、相同帧数、
相同输出数量，只改变 track IDs（Association-Controlled）。
评测：官方 TrackEval（HOTA/CLEAR/Identity），输出中 AssA(0)/IDF1(0)
按 TrackEval `array_labels[0]=0.05` 档位报告。

#### 1. 方法定义

| Variant | 定义 |
|---|---|
| C0 | 纯 IoU（last box），Hungarian + 阈值 0.3 |
| C1 | Kalman motion IoU（pred box）+ second-stage last-IoU，阈值 0.3 |
| C2 | raw PBD cosine，阈值 0.3 |
| C3 | 固定线性融合 IoU+PBD（0.7/0.3），阈值 0.3 |
| L1DK base | Kalman motion IoU + IoU + PBD 线性融合（0.4/0.4/0.2），阈值 0.25，无训练参数 |
| L1DK_d03 | 同一基座 + EGRA set-transformer 有界残差（delta scale 0.3） |

#### 2. DanceTrack val（40 序列，官方 GT）

| Variant | HOTA | AssA | IDF1 | IDSW | Frag |
|---|---:|---:|---:|---:|---:|
| C0 | 0.6078 | 0.3899 | 0.5291 | 3,554 | 5,283 |
| C1 | 0.6301 | 0.4193 | 0.5660 | 2,916 | 5,221 |
| C2 | 0.3836 | 0.1555 | 0.3188 | 15,616 | 5,621 |
| C3 | 0.6103 | 0.3934 | 0.5367 | 2,981 | 5,254 |
| **L1DK base** | **0.6280** | **0.4165** | **0.5630** | **2,558** | **5,209** |
| L1DK_d03 | 0.6149 | 0.3992 | 0.5522 | 2,598 | 5,218 |

#### 3. MOT17 train（3 序列）

| Variant | HOTA | AssA | IDF1 | IDSW | Frag |
|---|---:|---:|---:|---:|---:|
| C0 | 0.5682 | 0.4504 | 0.5055 | 569 | 328 |
| C1 | 0.6308 | 0.5530 | 0.5606 | 340 | 329 |
| C2 | 0.2645 | 0.0975 | 0.2155 | 1,325 | 345 |
| C3 | 0.5259 | 0.3856 | 0.4507 | 351 | 325 |
| **L1DK base** | **0.6569** | **0.6010** | **0.5784** | **276** | **323** |
| L1DK_d03 | 0.6525 | 0.5922 | 0.5775 | 274 | 325 |

#### 4. MOT20 train（2 序列）

| Variant | HOTA | AssA | IDF1 | IDSW | Frag |
|---|---:|---:|---:|---:|---:|
| C0 | 0.4196 | 0.2071 | 0.3206 | 3,113 | 430 |
| C1 | 0.4942 | 0.2869 | 0.3956 | 2,824 | 426 |
| C2 | 0.2508 | 0.0740 | 0.1755 | 3,139 | 445 |
| C3 | 0.4299 | 0.2171 | 0.3248 | 2,396 | 425 |
| L1DK base | 0.4864 | 0.2779 | 0.3232 | 3,736 | 423 |
| **L1DK_d03** | **0.4937** | **0.2864** | **0.3916** | **2,408** | **427** |

#### 5. BDD100K train（200 视频，5fps 采样）

| Variant | HOTA | AssA | IDF1 | IDSW | Frag |
|---|---:|---:|---:|---:|---:|
| C0 | 0.3731 | 0.3044 | 0.2981 | 14,457 | 2,595 |
| C1 | 0.3715 | 0.3019 | 0.2989 | 14,137 | 2,594 |
| C2 | 0.2754 | 0.1659 | 0.2146 | 12,424 | 2,587 |
| C3 | 0.3210 | 0.2255 | 0.2499 | 11,256 | 2,594 |
| **L1DK base** | **0.3878** | **0.3292** | **0.3167** | **12,149** | **2,589** |
| L1DK_d03 | 0.3603 | 0.2841 | 0.2889 | 11,151 | 2,588 |

#### 6. Macro 汇总（等权四域）

| Variant | Macro AssA | Macro IDF1 | AssA 胜场 |
|---|---:|---:|---:|
| C0 | 0.3380 | 0.4133 | 0 |
| C1 | 0.3903 | 0.4553 | 1（DanceTrack） |
| C2 | 0.1232 | 0.2331 | 0 |
| C3 | 0.3054 | 0.3905 | 0 |
| **L1DK base** | **0.4062** | 0.4453 | **3（DanceTrack/MOT17/BDD）** |
| L1DK_d03 | 0.3905 | **0.4526** | 1（MOT20） |

#### 7. 结论与 BEST_STRONG_BASE

1. **L1DK base 是当前最强统一基座**：macro AssA 0.4062 最高，且在
   DanceTrack、MOT17、BDD 三个域同时最优；MOT20 仅略低于 L1DK_d03
   （AssA 0.2779 vs 0.2864）。
2. **Motion 单独（C1）在 DanceTrack/MOT20 很强，但 BDD 上弱**；
   PBD 单独（C2）在所有域最弱（macro AssA 0.1232），印证 L1-C 结论：
   raw PBD 不能独立关联。
3. **EGRA residual（L1DK_d03）只在 MOT20 正向**，其余域 AssA 均下降，
   与 L1-D 的“局部修正成功但轨迹级失败”一致；这正是 Stage L2 要解决的
   objective mismatch 证据链的一环。
4. **BEST_STRONG_BASE = L1DK base**（0.4 IoU + 0.2 PBD + 0.4 Kalman
   motion，thr 0.25）。后续所有 counterfactual oracle、utility 训练与
   最终对比均以此为准。

#### 8. 数据文件

- DanceTrack：`outputs/l1_c/association_controlled_main.csv`
  （本次只含 C0/C1/C2/C3/L1DK_BASE/L1DK_d03；旧版已备份到
  `outputs/l2/old_l1c/`）
- BDD/MOT17/MOT20：`outputs/l1_d/ac_{bdd,mot17,mot20}.csv`
  （旧版已备份到 `outputs/l2/old_l1d/`）
- 轨迹文件：`outputs/l2/baseline_AC/{bdd100k_train,mot17_train,mot20_train}/{variant}/`
  与 `outputs/l1_c/trackeval/{variant}/`


## 附录 E — Novelty Collision 审计

> 来源文件：`reports/l2_novelty_collision_audit.md`（已嵌入本报告）

### Stage L2 — Novelty Collision Audit

日期：2026-08-10。
核心问题：是否存在已公开方法明确做——

> 训练时使用未来轨迹/反事实 rollout 评估当前 association action 的
> counterfactual long-term trajectory utility，再把该偏好蒸馏给
> online causal association policy。

#### 1. 检索范围

- 关键词（2025/2026 优先）：future utility MOT、trajectory-level MOT
  loss、long-horizon MOT association、future-aware association、
  counterfactual tracking、track detection link prediction、
  trajectory consistency、sequence-level MOT objective、
  differentiable assignment、decision-focused tracking、
  reinforcement learning MOT。
- 逐项核实官方仓库（见 `docs/l2_reference_audit.md`），全部实际
  clone + 阅读 README/模型/loss/推理代码。

#### 2. 逐项结论

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

#### 3. 结论

**NO DIRECTLY EQUIVALENT VERIFIED METHOD FOUND**

即：在 2026-08-10 可核实的公开官方实现范围内，未发现与本项目
核心设定（counterfactual future trajectory utility → causal policy
蒸馏）直接等价的方法。

注意：

1. 该结论是“未发现”，不是“首创”。论文中不得使用 first；
2. 检索存在固有盲区（付费期刊、未公开代码、非英语来源），
   最终稿需在时间允许时再次检索；
3. 若未来发现等价方法，必须修改 claim 并明确差异。


## 附录 F — Counterfactual Rollout Oracle

> 来源文件：`reports/l2_counterfactual_oracle.md`（已嵌入本报告）

### Stage L2 — Counterfactual Rollout Oracle

日期：2026-08-10。

#### 1. 目的

在训练任何大模型之前，回答：

1. 当前关联决策（L1DK base）是否存在“换一个 action 后未来轨迹效用
   显著更好”的空间（oracle headroom）？
2. local correctness 与 future utility 是否真的不一致？

#### 2. 状态与数据

- 基座：BEST_STRONG_BASE = L1DK base
  （0.4 IoU + 0.2 PBD + 0.4 Kalman motion，thr 0.25，max_age 30）；
- 重放：`tools/run_l2_oracle.py` 用与 baseline 完全相同的 AC shell
  逐帧重放，已验证与官方基线输出 100% 一致（MOT17-04-SDP 3589/3589）；
- 数据域：DanceTrack val（40 视频）、BDD100K train（30 视频采样）、
  MOT17 train（3 视频）、MOT20 train（2 视频）、DanceTrack calibration
  （8 视频）；
- 冲突定义：affinity 图（base≥0.25 或 row/col top-2）的连通分量，
  只保留 |T|≥2 或 |C|≥2 的组件；每帧最多采样 40 个冲突。

#### 3. 动作空间

每个冲突组件生成 6–8 个候选 action：

1. base action（A0，必选）；
2. GT-local action（若轨迹真 ID 出现在组件候选内）；
3. 按 base 分数排序的 top-k 替代匹配（穷举小组件，大组件启发式）；
4. 最差匹配（sanity）；
5. all-new（组件内全部不匹配，候选全部出生）。

每个 action 用 `complete_assignment` 补全为合法全局一对一分配，
然后冻结 base policy 继续 rollout H 帧。

#### 4. 效用定义

窗口效用用 TrackEval 同款公式（见
`docs/l2_trackeval_objective_audit.md`）：

- `U_H = windowed AssA`（主指标）；
- 辅助：windowed IDF1、window IDSW、TP/GT 计数。

窗口取 action 之后未来帧 `[t+1, t+H]`（不含 t，保证纯未来后果）。

#### 5. 结果

##### DanceTrack val（25 视频，1,000 冲突事件）

| H | base windowed AssA | oracle-best | mean gain | frac better |
|---|---:|---:|---:|---:|
| 4 | 0.9734 | 0.9772 | +0.38pp | 7.5% |
| 8 | 0.9593 | 0.9645 | +0.52pp | 13.5% |
| 16 | 0.9444 | 0.9509 | +0.65pp | 18.5% |
| 32 | 0.9219 | 0.9293 | +0.74pp | 21.9% |

##### BDD100K train（30 视频，745 冲突事件，5fps）

| H | base windowed AssA | oracle-best | mean gain | frac better |
|---|---:|---:|---:|---:|
| 2 | 0.7237 | 0.7334 | +0.96pp | 18.0% |
| 4 | 0.6225 | 0.6332 | +1.07pp | 36.0% |
| 8 | 0.5313 | 0.5429 | +1.17pp | 53.3% |
| 16 | 0.4709 | 0.4809 | +1.01pp | 61.7% |

##### MOT17 / MOT20（小样本）

| 域 | 事件 | H32 mean gain | frac better |
|---|---:|---:|---:|
| MOT17 train | 120 | +2.27pp | 70.8% |
| MOT20 train | 80 | +1.76pp | 75.0% |

单事件窗口 headroom 在 MOT17/MOT20 反而最大，但端到端验证 MOT17
为 −2.32pp（见 `reports/l2_oracle_headroom.md`）。

##### 端到端 privileged greedy oracle

| 视频 | gain（整视频 AssA） | IDSW base→oracle |
|---|---:|---:|
| dancetrack0004 | +0.02pp | 149→151 |
| dancetrack0005 | +0.06pp | 68→74 |
| BDD×3 | −0.88pp（均值） | 260→283 |
| MOT17-02-SDP | −2.32pp | 783→866 |

结论：单事件窗口 headroom 存在但小；端到端实现后无正收益。

#### 6. 输出文件

- `outputs/l2/oracle/events_{domain}.pkl`：逐事件动作与效用；
- `outputs/l2/oracle/oracle_{domain}.json`：聚合 headroom；
- `outputs/l2/oracle/analysis_{domain}.json`：mismatch/ranking 分析。


## 附录 G — Oracle Headroom（Gate 1）

> 来源文件：`reports/l2_oracle_headroom.md`（已嵌入本报告）

### Stage L2 — Oracle Headroom（Gate 1）

日期：2026-08-10。
Gate 1 判定：**TRAJECTORY_UTILITY_HEADROOM_LOW**（不启动大型 TUM 训练）。

#### 1. 定义

Oracle headroom = 在允许使用未来 GT / 未来 rollout 的 privileged 条件下，
选择 counterfactual future-best action 相对 BEST_STRONG_BASE（L1DK base）
能获得的整视频 TrackEval AssA / IDSW 提升。

#### 2. 单事件窗口 headroom（隔离评估）

每个冲突组件枚举 6–8 个候选 action，冻结 base policy rollout
H∈{4,8,16,32} 帧，计算 windowed AssA（与官方 TrackEval 公式一致，
整视频校验 AssA/IDF1 精确等于官方值）。

##### DanceTrack val（1,000 冲突事件，25 视频）

| H | 事件数 | base windowed AssA | oracle-best | mean gain | frac better | base IDSW | best IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1,000 | 0.9734 | 0.9772 | +0.38pp | 7.5% | 0.64 | 0.59 |
| 8 | 1,000 | 0.9593 | 0.9645 | +0.52pp | 13.5% | 1.70 | 1.65 |
| 16 | 1,000 | 0.9444 | 0.9509 | +0.65pp | 18.5% | 3.80 | 3.77 |
| 32 | 1,000 | 0.9219 | 0.9293 | +0.74pp | 21.9% | 8.13 | 8.10 |

##### BDD100K train（745 冲突事件，30 视频，5fps）

| H | 事件数 | base | oracle-best | mean gain | frac better |
|---|---:|---:|---:|---:|---:|
| 2 | 745 | 0.7237 | 0.7334 | +0.96pp | 18.0% |
| 4 | 745 | 0.6225 | 0.6332 | +1.07pp | 36.0% |
| 8 | 745 | 0.5313 | 0.5429 | +1.17pp | 53.3% |
| 16 | 745 | 0.4709 | 0.4809 | +1.01pp | 61.7% |

##### MOT17 / MOT20（小样本）

| H | MOT17 gain | MOT17 frac | MOT20 gain | MOT20 frac |
|---|---:|---:|---:|---:|
| 4 | +1.61pp | 55.0% | +1.76pp | 66.2% |
| 8 | +1.83pp | 62.5% | +1.61pp | 65.0% |
| 16 | +2.25pp | 68.3% | +1.59pp | 72.5% |
| 32 | +2.27pp | 70.8% | +1.76pp | 75.0% |

注意：MOT17 的单事件窗口 headroom 反而最大（H32 +2.27pp），但端到端
为 −2.32pp——这是“窗口效用与全局轨迹质量不同构”的最强直接证据。

#### 3. 端到端（receding-horizon greedy oracle）

对每个冲突帧用 privileged rollout 选最佳 action 并应用，其余帧用 base，
整视频计算 TrackEval 同款 AssA：

| 视频 | 帧数 | base AssA | oracle AssA | gain | oracle IDSW |
|---|---:|---:|---:|---:|---:|
| dancetrack0004 | 1,203 | 0.1702 | 0.1704 | +0.02pp | 151（base 149） |
| dancetrack0005 | 1,203 | 0.4562 | 0.4568 | +0.06pp | 74（base 68） |
| BDD 0000f77c-…58 | 41 | 0.1319 | 0.1582 | +2.62pp | 121（105） |
| BDD 0000f77c-…88 | 40 | 0.2439 | 0.1963 | −4.76pp | 36（30） |
| BDD 0000f77c-…98 | 40 | 0.2352 | 0.2302 | −0.50pp | 126（125） |
| MOT17-02-SDP | 80 | 0.1830 | 0.1598 | −2.32pp | 866（783） |

均值：DanceTrack +0.04pp；BDD −0.88pp；MOT17 −2.32pp。

#### 4. 结论

1. **单事件窗口 headroom 存在但很小**（DanceTrack 0.4–0.7pp、
   BDD 1.0–1.2pp），且窗口重叠导致端到端可实现收益大幅稀释；
2. **端到端 oracle 不提升整视频 AssA**：DanceTrack 2 视频 +0.02/+0.06pp，
   BDD 3 视频平均 −0.88pp，MOT17 −2.32pp；IDSW 全部变差；
3. 即使拥有完整 privileged future 和 oracle action 选择，相对
   BEST_STRONG_BASE 的整视频 AssA headroom **< 0.1pp（DanceTrack）**，
   远低于 1pp 阈值；
4. 因此按任务书停止条件：

```text
L2_ORACLE_HEADROOM_LOW
```

不启动大型 Trajectory Utility Model 训练；进入失败分析与最终报告。

#### 5. 为什么 headroom 低（机制）

- L1DK base 的匈牙利匹配在短窗口内已接近最优（DanceTrack H4 base
  AssA=0.973，几乎没有可修空间）；
- 大多数冲突组件的替代 action 经 base policy 再优化后与 base 等价
  （效用相同的事件占比很高）；
- 窗口级 AssA 最优 ≠ 全局身份质量：贪婪 oracle 的局部选择反而增加
  IDSW（DanceTrack 149→151、BDD 260→283、MOT17 783→866）；
- 这与 L1-D 的结论一致：local/future-window 的修正与 TrackEval
  轨迹级统计不同构。


## 附录 H — Local vs Future Mismatch

> 来源文件：`reports/l2_local_vs_future_mismatch.md`（已嵌入本报告）

### Stage L2 — Local Correctness vs Future Utility Mismatch

日期：2026-08-10。

#### 1. 定义

- 局部正确：action 中被匹配的 (track, candidate) 的 candidate GT 等于
  track 的出生 GT（true identity）；
- 未来最优：windowed AssA 最高的 action；
- 类别：
  - `local_correct_future_bad`：base 局部更正确，但未来效用反而更低；
  - `local_wrong_future_good`：base 局部更错，但未来效用更高；
  - `base_correct_future_suboptimal`：base 已局部全对，但未来仍可改进。

#### 2. DanceTrack val（1,000 冲突事件）

| H | future_best ≠ base | local_correct_future_bad | local_wrong_future_good | base_correct_future_suboptimal |
|---|---:|---:|---:|---:|
| 4 | 75 | 30 | 39 | 4 |
| 8 | 135 | 69 | 49 | 9 |
| 16 | 185 | 98 | 58 | 11 |
| 32 | 219 | 128 | 60 | 19 |

#### 3. BDD100K train（745 冲突事件）

| H | future_best ≠ base | local_correct_future_bad | local_wrong_future_good | base_correct_future_suboptimal |
|---|---:|---:|---:|---:|
| 2 | 134 | 40 | 46 | 10 |
| 4 | 268 | 90 | 74 | 23 |
| 8 | 397 | 137 | 110 | 38 |
| 16 | 460 | 173 | 110 | 39 |

#### 4. 多 horizon 排序一致性

同一事件内 action 的效用排序随 horizon 增长逐渐稳定：

| 对比 | DanceTrack pair agreement | BDD pair agreement |
|---|---:|---:|
| 最短 vs 最长 | 27.6%（H4 vs H32） | 19.0%（H2 vs H16） |
| 中间 vs 最长 | 54.3%（H8 vs H32） | 42.0%（H4 vs H16） |
| 次长 vs 最长 | 73.7%（H16 vs H32） | 67.7%（H8 vs H16） |

MOT17/MOT20（小样本）：H16 vs H32 pair agreement 78.2% / 76.8%。

#### 4b. MOT17 / MOT20 mismatch（H32）

| 域 | 事件 | future_best ≠ base | local_correct_future_bad | local_wrong_future_good |
|---|---:|---:|---:|---:|
| MOT17 | 120 | 85（70.8%） | 60 | 23 |
| MOT20 | 80 | 60（75.0%） | 47 | 7 |

MOT17/MOT20 的 mismatch 更偏向 “local correct but future bad”，
且单事件窗口 headroom 最大（+2.27/+1.76pp），但端到端实现后
MOT17 为 −2.32pp，进一步证明窗口效用与全局轨迹质量不同构。

#### 5. 结论

1. **local 与 future 确实不同构**：DanceTrack H32 有 219/1000（21.9%）
   事件的未来最优 action 不同于 base；其中 128 个属于
   “base 局部正确但未来更差”，60 个属于“base 局部错但未来更好”；
   BDD H16 有 460/745（61.7%）、MOT17 H32 有 85/120（70.8%）、
   MOT20 H32 有 60/80（75.0%）事件未来最优不同于 base。
2. **但该 mismatch 的规模不足以支撑端到端收益**：这些差异的 windowed
   AssA 幅度平均只有 0.5–1.2pp，且端到端实现后互相抵消（见
   `reports/l2_oracle_headroom.md`）。
3. 科学含义：objective mismatch 真实存在（支撑论文核心问题），
   但在 L1DK base + AC 协议下 **不存在可学习的正收益空间**；
   这与 L1-D 的“局部修正成功但全局失败”完全一致。


## 附录 I — 历史身份污染审计

> 来源文件：`reports/l2_history_contamination_audit.md`（已嵌入本报告）

### Stage L2 — 历史身份污染审计

日期：2026-08-10。

#### 1. 问题

L1-D 发现 EGRA 的局部修正（连续性 0.841→0.951）没有转化为 TrackEval
AssA 收益。本审计检查：这些修正是否主要发生在“已经被历史 IDSW 污染”
的预测轨迹上；若是，则局部正确但未来无效的机制成立。

#### 2. 方法

用与 baseline 完全一致的 AC shell 重放 L1DK base 与 EGRA（L1DK_d03），
逐帧对比 base 与 EGRA 的 track 分配。对每个发生修正的 track，在修正前
记录：

- track 出生 GT（true identity）与最近匹配 GT；
- 轨迹 purity（历史匹配中主导 GT 占比）；
- 历史 IDSW 次数、fragment 数、age、hits；
- base 边与 EGRA 边的 candidate GT。

分类（按出生身份）：

- helpful：EGRA 把分配改到 true identity 的候选；
- harmful：base 已分配到 true identity，EGRA 改走；
- same_gt / other：其余。

#### 3. 聚合统计（专用审计 `tools/l2_history_contamination.py`）

##### DanceTrack val

- 修正事件：1,295；
- helpful：334（25.8%）；harmful：264（20.4%）；
- same_gt：213（16.4%）；other：484（37.4%）。

##### BDD100K train（30 视频）

- 修正事件：1,799；
- helpful：163（9.1%）；harmful：247（13.7%）；
- same_gt：748（41.6%）；other：641（35.6%）。

#### 4. 污染状态分布

##### DanceTrack val（按修正前轨迹状态）

| purity bucket | n | helpful | harmful |
|---|---:|---:|---:|
| <0.5 | 257 | 52 | 30 |
| 0.5–0.8 | 400 | 92 | 75 |
| 0.8–0.95 | 288 | 75 | 54 |
| ≥0.95 | 350 | 115 | 105 |

| past IDSW | n | helpful | harmful |
|---|---:|---:|---:|
| 0 | 317 | 116 | 87 |
| 1 | 104 | 31 | 9 |
| 2–3 | 131 | 40 | 35 |
| ≥4 | 743 | 147 | 133 |

| age | n | helpful | harmful |
|---|---:|---:|---:|
| 1–3 | 82 | 22 | 22 |
| 4–10 | 107 | 37 | 29 |
| 11–30 | 173 | 49 | 43 |
| >30 | 933 | 226 | 170 |

##### BDD100K train（30 视频）

| purity bucket | n | helpful | harmful |
|---|---:|---:|---:|
| <0.5 | 1,076 | 74 | 175 |
| 0.5–0.8 | 101 | 11 | 6 |
| 0.8–0.95 | 18 | 1 | 1 |
| ≥0.95 | 604 | 77 | 65 |

| past IDSW | n | helpful | harmful |
|---|---:|---:|---:|
| 0 | 1,207 | 121 | 201 |
| 1 | 264 | 14 | 20 |
| 2–3 | 191 | 19 | 17 |
| ≥4 | 137 | 9 | 9 |

| age | n | helpful | harmful |
|---|---:|---:|---:|
| 1–3 | 683 | 65 | 107 |
| 4–10 | 532 | 31 | 86 |
| 11–30 | 512 | 58 | 49 |
| >30 | 72 | 9 | 5 |

#### 5. 解读

1. EGRA 的修正大多数 **不指向出生身份**（DanceTrack helpful 25.8%、
   BDD 9.1%），说明学习到的修正方向与 GT 身份不一致；
2. BDD 上 59.8% 的修正发生在 purity<0.5 的重度污染轨迹上，且这些
   轨迹上的修正 harmful（175）远多于 helpful（74）；
3. DanceTrack 上 57% 的修正发生在 past IDSW≥4 的轨迹上，helpful 率
   反而低于 IDSW=0 轨迹（19.8% vs 36.6%）；
4. 这解释了 L1-D 的“局部正确但 AssA 下降”：在 AC 协议下，
   ID 是持久符号，污染轨迹上的任何局部修复都只能增加碎片化，
   无法恢复已损失的全局 ID 统计。

#### 6. 结论

历史污染审计支持 Stage L2 的核心机制判断：**已污染状态上的局部
correct action 与未来轨迹效用不同构**。但同时也说明，在固定 ID
符号的 AC 协议下，这类修正没有可恢复空间（与 oracle headroom 低
一致）。


## 附录 J — Trajectory Utility Model 设计

> 来源文件：`reports/l2_utility_model.md`（已嵌入本报告）

### Stage L2 — Trajectory Utility Model (TUM) 设计

日期：2026-08-10。本文记录模型设计；是否训练由 Oracle Headroom Gate
决定（见 `reports/l2_oracle_headroom.md`）。

#### 1. 科学定位

TUM 不是一个新的 tracker，而是“当前关联 action → 未来轨迹效用”的
回归/排序模型：

```text
Q(s_t, a) ≈ U_H(s_t, a)
```

- 训练监督：counterfactual rollout 的 windowed AssA（privileged future）；
- 输入：严格 causal（不含未来帧）；
- 推理：从候选 action 集合中选 argmax Q 的 action。

#### 2. 输入表示（全部来自 L1DK base 的在线状态）

每个决策事件：

- Track tokens（每轨迹 16 维，复用 L1-D `TRACK_FEATURES`）：
  box、velocity、gap、age、hits、IoU/PBD/base top1 与 margin、
  anchor-cos 等；
- Candidate tokens（每候选 12 维，复用 `CAND_FEATURES`）：
  box、gen、size、top1/margin 等；
- Pair features（19 维，复用 `PAIR_FEATURES`）：
  IoU、Kalman-pred IoU、PBD cos、中心距离、scale、margin、
  set context；
- Action 指示矩阵：component 内 (track, candidate) 是否被该 action 匹配；
- Base 矩阵（强先验）。

历史身份污染只用 prediction-side proxy（track age、gap、history
consistency、top1 margin 等），禁止输入真实 GT purity。

#### 3. 模型

- TUM-small（pilot）：d_model=256、4 层 TransformerEncoder、8 heads、
  FFN 1024；track/cand token 投影后一起做 set self-attention；
  pair head 拼接 `[track_emb, cand_emb, pair_feats, base, action]` →
  每 pair logit，按 action 边加权池化 → per-horizon head。
- 参数量：约 1.5–3M（pilot 足够验证 signal）。
- 若 pilot 通过：TUM-base 6–8 层、d_model 384，参数量 5–15M，
  DDP 2–4 卡训练。

#### 4. 训练目标

每个 event × action × H 有一个 oracle utility（windowed AssA）。

- 主 loss：utility 回归 MSE（同一 event 内 action 间归一化可选）；
- 辅助：listwise ranking loss（softmax CE over actions per event），
  使排序与 oracle 一致；
- Multi-horizon：共享 trunk + per-H head；训练时所有 H 联合。

#### 5. 数据与 split

- 训练域：DanceTrack calibration + BDD100K train（30 视频 oracle 事件）；
- 评估域：DanceTrack val（主 AC 验证）+ MOT17/MOT20 train（跨域）；
- split 按 video 隔离，保证同一视频不跨 train/eval。

#### 6. 评估指标（Gate 2）

- top1 future-best action 准确率；
- pairwise ranking AUC（action 对）；
- NDCG@k；
- regret = U(best) − U(predicted choice)；
- 对比：base 恒选、local-correctness probe（选 GT-local 最优 action，
  仅 oracle 分析用）、随机。

#### 7. 与 EGRA 的区别

| | EGRA | TUM |
|---|---|---|
| 监督 | 当前帧 GT 身份 | counterfactual 未来 windowed AssA |
| 决策 | 每 pair 残差 | 集合级 action 效用排序 |
| 推理 | 单帧 Hungarian | online，同帧 action 选择 + Hungarian |
| 目标 | local correctness | trajectory utility |

#### 8. 不采用的设计

- 不直接回归未来 IDSW（方差大、难归一化）；
- 不把未来 GT 身份作为输入（泄漏）；
- 不首版使用 RL（见 `docs/future_rl_reference.md`）。


## 附录 K — 失败分析

> 来源文件：`reports/l2_failure_analysis.md`（已嵌入本报告）

### Stage L2 — Failure Analysis

日期：2026-08-10。

#### 1. 失败结论

```text
L2_ORACLE_HEADROOM_LOW
```

核心假设“用 counterfactual future trajectory utility 训练当前关联
决策可以改善统一 MOT”在 L1DK base + Association-Controlled 协议下
**没有可实现的 oracle headroom**，因此按任务书停止大型 TUM 训练。

#### 2. 证据链

1. 单事件窗口 headroom：DanceTrack H32 平均 +0.74pp，BDD H16
   +1.01pp（见 `reports/l2_oracle_headroom.md`）；
2. 端到端 greedy oracle（privileged）：DanceTrack +0.02/+0.06pp，
   BDD 均值 −0.88pp，MOT17 −2.32pp；IDSW 全部变差；
3. local vs future mismatch 存在（DanceTrack H32 21.9% 事件未来最优
   不同于 base），但幅度不足以转化为收益（见
   `reports/l2_local_vs_future_mismatch.md`）；
4. EGRA 修正审计（专用审计）：DanceTrack val helpful 334 /
   harmful 264 / same_gt 213 / other 484；BDD helpful 163 /
   harmful 247 / same_gt 748 / other 641；BDD 59.8% 修正位于
   purity<0.5 轨迹且 harmful 多于 helpful。

#### 3. 根因分析

##### 3.1 基座窗口内已接近最优

DanceTrack 冲突事件 H4 窗口 base AssA=0.9734，几乎没有可修空间；
可改进事件占比仅 7.5%（H4）到 21.9%（H32）。

##### 3.2 局部窗口最优 ≠ 全局轨迹质量

greedy oracle 优化 windowed AssA，但整视频 IDSW 反而上升：

- DanceTrack 149→151 / 68→74；
- BDD 260→283；
- MOT17 783→866。

这与 L1-D 观察一致：TrackEval 的 IDSW/AssA 是全局 ID 共现统计，
局部窗口效用与其不同构。

##### 3.3 动作空间经 base 再优化后趋同

大量替代 action 在 `complete_assignment` + base policy 再优化后
与 base 行为相同（效用完全相同的事件占比高），导致
“可行动的”headroom 更小。

##### 3.4 历史污染不可在短窗口内修复

已污染轨迹（purity<0.8）上的修正即使局部 GT 正确，也不能恢复
该轨迹过去已经发生的 ID 错误；未来窗口效用不奖励这种“迟到的正确”。

#### 4. 如果继续会怎样

- TUM 即使能精确预测 oracle 偏好，预测出的 action 也只能带来
  ≤0.1pp 整视频 AssA（DanceTrack），无法通过 +2pp / IDSW −15%
  的强信号门槛；
- 因此训练大型模型是浪费；训练小模型也只能得到一个“预测很准但
  没收益”的 utility learner，不能支撑 ICLR 实证要求。

#### 5. 可保留的科学产出

1. **Objective mismatch 的实验证据**：local correctness 与 future
   windowed utility 在 20–60% 冲突事件上不一致；
2. **TrackEval 目标审计**：AssA/IDF1 是轨迹级 ID 共现统计；
3. **验证过的 windowed AssA 实现**：整视频结果与官方 TrackEval
   完全一致；
4. **L1DK base 四域公平矩阵**；
5. **2025–2026 官方代码审计**：无直接等价方法。

这些证据指向：若未来要追求该方向，应改变**效用定义**
（例如整序列 ID 映射 + IDSW 惩罚）或**基座协议**
（例如允许恢复/重映射 ID），而不是简单堆模型容量。


## 附录 L — Stage L2 GPT Handoff

> 来源文件：`reports/STAGE_L2_GPT_HANDOFF.md`（已嵌入本报告）

### Stage L2 GPT Handoff

日期：2026-08-10。项目：LocateMOT（不是 TrackOCD）。

#### 一句话结论

Stage L2 验证了核心科学问题（local correctness 与 future trajectory
utility 不同构），但 **oracle headroom 不足**：即使拥有 privileged
future，端到端整视频 AssA 提升 < 0.1pp（DanceTrack）且 BDD/MOT17
为负；判定 `L2_ORACLE_HEADROOM_LOW`，未启动大型 TUM 训练，按任务书
直接进入失败分析与最终报告。

#### 关键数字

##### Baseline（四域 AC 公平矩阵，官方 TrackEval）

| Variant | DanceTrack AssA | MOT17 AssA | MOT20 AssA | BDD AssA | Macro |
|---|---:|---:|---:|---:|---:|
| C0 IoU | 0.3899 | 0.4504 | 0.2071 | 0.3044 | 0.3380 |
| C1 Motion | 0.4193 | 0.5530 | 0.2869 | 0.3019 | 0.3903 |
| C2 PBD | 0.1555 | 0.0975 | 0.0740 | 0.1659 | 0.1232 |
| C3 IoU+PBD | 0.3934 | 0.3856 | 0.2171 | 0.2255 | 0.3054 |
| **L1DK base** | **0.4165** | **0.6010** | 0.2779 | **0.3292** | **0.4062** |
| L1DK_d03 | 0.3992 | 0.5922 | **0.2864** | 0.2841 | 0.3905 |

BEST_STRONG_BASE = L1DK base（0.4 IoU + 0.2 PBD + 0.4 Kalman motion，
thr 0.25）。

##### Oracle headroom

单事件窗口（冻结 base policy rollout）：

- DanceTrack val：H4 +0.38pp → H32 +0.74pp（mean windowed AssA），
  frac better 7.5%→21.9%（1,000 事件）；
- BDD：H2 +0.96pp → H16 +1.01pp，frac better 18%→62%（745 事件）。

端到端 privileged greedy oracle（整视频 TrackEval 同款 AssA）：

- dancetrack0004：+0.02pp；dancetrack0005：+0.06pp；
- BDD×3：均值 −0.88pp（+2.62/−4.76/−0.50）；
- MOT17-02-SDP：−2.32pp；
- IDSW 全部变差（DanceTrack 149→151、68→74；BDD 260→283；
  MOT17 783→866）。

##### Local vs future mismatch

- DanceTrack H32：219/1000 事件 future-best ≠ base；
  local_correct_future_bad 128 / local_wrong_future_good 60；
- BDD H16：460/745 事件 future-best ≠ base；
  local_correct_future_bad 173 / local_wrong_future_good 110；
- 多 horizon 排序一致性：H16 vs H32 73.7%（DanceTrack）、
  H8 vs H16 67.7%（BDD）。

##### 历史污染审计（EGRA 修正）

- DanceTrack val：1,295 修正，helpful 334（25.8%）/ harmful 264 /
  same_gt 213 / other 484；57% 位于 past IDSW≥4 轨迹且 helpful 率更低；
- BDD：1,799 修正，helpful 163（9.1%）/ harmful 247 / same_gt 748 /
  other 641；59.8% 位于 purity<0.5 轨迹，harmful 175 > helpful 74。

#### 方法学可信度

- replay 与官方基线 100% 一致（MOT17-04-SDP 3589/3589）；
- 本项目 windowed AssA/IDF1 与官方 TrackEval 整视频数值完全一致
  （DanceTrack 0004/0005、MOT17-04 验证）；
- 文献审计（TDLP/SambaMOTR/TRACT/UniTrack/PathConsistency/QuoVadis/
  FDTA/HATReID-MOT/HNCD-MOTR 全部实际 clone 阅读）：
  **NO DIRECTLY EQUIVALENT VERIFIED METHOD FOUND**。

#### 为什么没有训练 TUM

任务书明确停止条件：oracle 相对 BEST_STRONG_BASE 无 ~1pp AssA
headroom 且 IDSW 无改善 → `L2_ORACLE_HEADROOM_LOW`，不训练大模型。
本阶段 oracle 整视频 headroom < 0.1pp（DanceTrack），端到端 IDSW
变差，因此停止。

#### 未完成 / 下一步建议

1. 若继续该方向：换效用定义（整序列 ID 映射 + IDSW 惩罚）或
   换协议（允许 ID 重映射的 full-tracker）后重新验证 oracle；
2. 否则回到 L1DK base 的 full-tracker 工程路线；
3. TAO cache 缺失，后续需补 cache 才能做 TAO 域；
4. 无阻塞问题；所有产物已写入仓库（见 STAGE_L2_FINAL_REPORT.md §64）。

---
（本报告为自包含版本：附录 A–L 为各产物完整原文。）
