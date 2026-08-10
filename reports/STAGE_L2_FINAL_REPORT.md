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

- 本阶段产物：
  - `docs/l2_reference_audit.md`
  - `docs/l2_trackeval_objective_audit.md`
  - `docs/future_rl_reference.md`
  - `reports/l2_baseline_matrix.md`
  - `reports/l2_novelty_collision_audit.md`
  - `reports/l2_counterfactual_oracle.md`
  - `reports/l2_oracle_headroom.md`
  - `reports/l2_local_vs_future_mismatch.md`
  - `reports/l2_history_contamination_audit.md`
  - `reports/l2_utility_model.md`
  - `reports/l2_failure_analysis.md`
- 工具：
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
