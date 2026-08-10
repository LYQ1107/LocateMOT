# Stage L3 Final Report

日期：2026-08-10。项目：LocateMOT。

## 1. Executive Summary

Stage L3 的目标是验证 Specification × Regime 因子分解的统一 MOT。
审计与 pilot 结果：

- 2025/2026 官方代码审计完成（SAM3/SAM3.1、GLEE、OVTR、OVTrack、
  Grounded SAM2、SAM2MOT、STORM-Bench、QTrack、AnyTrack 等）；
- **Claim 2（统一 prompt 接口）被 SAM3/GLEE 强碰撞**；Claim 3
  （latent regime 条件化）未发现直接等价实现；
- 数据审计：**BDD manifest 已含 11 类 GT**（多类可直接用）；
  TAO cache 为旧布局，延迟处理；
- 四域 regime 诊断：方法偏好随 regime 变化（MOT17 高运动桶 Motion
  +6.7pp），但总体信号弱；
- **U0（naive shared learned）pilot 超过 L1DK**：macro AssA 0.4013
  vs 0.3944；MOT17/MOT20 正向；
- **U1（regime 条件化）未过 Gate A**：macro AssA 0.3915（−0.98pp vs
  U0），仅 MOT20 微正；z_regime 呈 dataset shortcut
  （domain classifier 96.6%）。

Stage Decision：

```text
L3_REGIME_NOT_SUPPORTED（pilot）
REGIME_ROUTER_DATASET_SHORTCUT
（ICLR readiness：NOT_READY）
```

## 2. Original Unified MOT Goal

一个核心模型、一个主 checkpoint、无 dataset-specific head/adapter/
router，统一异质 box-MOT 域（DanceTrack/MOT17/MOT20/BDD/TAO）与
对象指定接口（ALL/category/open-vocab/referring/visual prompt）。

## 3. L1-A → L2 Evidence Chain

- L1-B Identity Adapter 失败；L1-C UAF/LoRA 失败；
- L1-D EGRA 局部成功/全局失败（L1_D_PARTIAL，L1DK base 采用）；
- L2 证明 counterfactual future-utility oracle headroom 不足
  （L2_ORACLE_HEADROOM_LOW）；
- L3 起点：L1DK base 是最强固定规则基座（macro AssA 0.4062，
  旧协议），但无学习能力；U0/U1 在其状态空间上学习。

## 4. Why Previous Association-Centric Routes Were Closed

- 统一 identity vector（L1-B）、from-scratch UAF（L1-C）、grounding
  LoRA（L1-C）、EGRA residual（L1-D）、future-utility oracle
  （L2）均被实验否定；L3 不回头。

## 5. Unified MOT Definition: Regime × Specification

见 `docs/l3_unified_task_definition.md`（附录 A）。

- Axis A：tracking regime（dense same-class / standard / crowd /
  multi-class driving / long-tail sparse）；
- Axis B：object specification（ALL / category / open-vocab /
  referring / visual prompt）；
- 模型目标：`Tracker(video, specification) -> persistent trajectories`。

## 6. 2025–2026 Literature/GitHub Audit

完整表见 `docs/l3_reference_audit.md`（附录 B）。要点：

| 方法 | 与 L3 关系 |
|---|---|
| SAM 3 / 3.1（Meta 2026） | Claim 2 强碰撞；无 latent regime |
| GLEE（CVPR 2024） | Claim 1/2 部分碰撞；多域联合训练 |
| OVTR / OVTrack | open-vocab MOT（Claim 2） |
| STORM-Bench / QTrack | referring/query MOT（Claim 2） |
| SAM2MOT | 代码未发布 |
| COVTrack/++、GOVTrack | 无官方代码 |
| MoE conditional（ICML 2026 等） | 结构参考，非 MOT |

## 7. Novelty Collision with SAM3/GLEE/OVTR/etc.

`reports/l3_novelty_collision_audit.md`（附录 C）：

- Claim 1：标准 box-MOT AC 协议下未见等价实现（GLEE 是检测
  foundation，非 AC association）；
- Claim 2：**碰撞（SAM3/GLEE/OVTR/QTrack/STORM）**，不能作为
  独立 novelty；
- Claim 3：未见直接等价实现，但 pilot 未证明其有效。

Gate 0 结论：`NO_DIRECT_EQUIVALENT_VERIFIED_METHOD_FOUND`
（对完整组合），并明确 Claim 2 边界。

## 8. Final Claim Boundary

- 可写：一个 shared learned checkpoint（U0）在 DanceTrack/MOT17/
  MOT20/BDD（多类 GT）上达到或超过固定规则基座；
- 不可写：latent regime 条件化有效（pilot 否定）；
- 不可写：prompt 接口统一为 novelty（SAM3/GLEE 已覆盖）；
- 不可写：first / foundation tracker / universal tracker。

## 9. Dataset & Protocol Matrix

`docs/l3_protocol_matrix.md`（附录 D）与 `docs/l3_dataset_audit.md`
（附录 E）。主实验集合：DanceTrack calib/val、MOT17、MOT20、BDD
（11 类）；TAO 延迟。

## 10. BDD Full Multi-Class Audit

- 现有 manifest `bdd100k_train.jsonl`（8,001 帧，200 视频）已含
  11 类 GT（`gt_categories`：bicycle/bus/car/motorcycle/other person/
  other vehicle/pedestrian/rider/trailer/train/truck）；
- `matched` 含全类候选匹配（5,191 帧 ≥2 匹配）；
- 现有 `outputs/l1_d/raw/bdd100k_train.pkl` 的 `cand_gt` 即全类；
- L1/L2 的 person-only 结果只是历史基线；
- L3 的 BDD 结果基于全类 GT 的 AC 评估（各方法同候选集）。

## 11. TAO/Open-World Audit

- manifest 存在（105 视频 / 4,200 帧，2,256 帧有候选）；
- cache 为旧布局（`train/{BDD,AVA,YFCC100M,HACS,LaSOT}/...`），
  与 manifest `cache_key` 不匹配、无 `.complete`；
- C-TAO 仅有清单级文件；
- 结论：TAO 需 cache 修复后纳入；L3 主 pilot 未包含 TAO。

## 12. Prompt/Specification Benchmarks

- STORM-Bench / RMOT26：官方基准已 clone，数据未下载；
- SAM3/GLEE 已证明 prompt 统一能力；本项目 B 轴未进入训练
  （A Gate 未过，且 Claim 2 有 collision）。

## 13. Negative Transfer in Naive Unified Tracking

`reports/l3_negative_transfer_audit.md`（附录 F）：

- per-domain 最优（C1/L1DK/EGRA/L1DK）vs 单一 L1DK：
  macro 负迁移约 −0.28pp（旧协议）；
- 负迁移存在但小；共享学习本身（U0）能吸收多域数据。

## 14. Regime-Specific Baseline Behavior

`reports/l3_regime_signal.md`（附录 G）：

- MOT17：motion=hi 桶 C1 0.6400 vs L1DK 0.6296（+1.0pp）；
  iou_amb=hi 桶 C1 0.6519 vs L1DK 0.5853（+6.7pp）；
  n_cand=lo 桶 EGRA 0.6182 vs L1DK 0.5726；
- BDD：n_cand=hi 桶 C0 0.5572 vs L1DK 0.5534；
- DanceTrack：L1DK/EGRA 稳定，差异 <0.5pp。

## 15. Evidence for Latent Tracking Regimes

**REGIME_SPECIALIZATION_SIGNAL_SUPPORTED（弱到中等）**：方法偏好随
regime 变化，但样本少、幅度小；不足以单独支撑 U1 必要性。

## 16. Final Architecture Decision

Pilot 采用：U0 = L1DAssociator（共享 dense）；U1 = L3Associator
（RegimeEncoder + FiLM）。最终架构未选定（U1 未过 Gate）。

## 17. Object Specification Encoder

设计：learned spec embedding + 候选类别兼容输入（未训练，见
`locatemot/models/l3_unified.py` 的 `SPECS/SPEC2ID`）。
对比基座：S0 LocateAnything/PBD（工程成熟）、S1 SAM3（License/
显存/登录限制）、S2 GLEE（MIT，但需全模型）。详见
`reports/l3_spec_backbone_audit.md`（附录 H）。

## 18. Latent Regime Encoder

48 维 prediction-side 统计 → MLP → 32 维 z_regime；
FiLM 条件化。Pilot 结论：z 与 dataset 强相关（shortcut），
未带来收益。

## 19. Shared Conditional Tracking Core

`locatemot/models/l3_unified.py::L3Associator`（~0.55M 可训练参数，
set transformer 2 层 + pair head + FiLM）。

## 20. Track Memory / Association Decoder

与 L1-D 相同：set-level transformer + pair residual + reliability gate；
匈牙利解码；无长期 memory bank。

## 21. Birth/Lifecycle

复用 L1 AC shell（OnlineTracker，output_all_candidates，max_age 30）。

## 22. Why No Dataset ID

训练/推理均无 dataset 输入；但 regime 特征与 dataset 天然共线
（BDD 5fps gap、密度等），pilot 的 z 仍学会 dataset shortcut，
这是必须公开承认的机制问题。

## 23. Training Data Mixture

U0/U1 均用 DanceTrack calibration + BDD（11 类）+ MOT17 + MOT20
的离线 AC 样本（`outputs/l1_d/data/*_k.pkl`，13,405 帧 / 88,465
监督事件）。硬样本加权采样与 L1-D 相同。

## 24. Sampling / Loss Balancing

hard-frame 加权（base 错误率）；row/col CE + reliability BCE +
base preservation（L1-D loss）。未做 dataset-balanced 重采样
（pilot 公平性：U0/U1 同数据）。

## 25. Parameter Count

U0：0.49M；U1：~0.55M。

## 26. 4-GPU Training Details

Pilot 只用 1 卡（GPU 8/9 各一）；30 epochs ≈ 6–7 分钟/模型；
正式训练未启动（Gate 未过）。

## 27. Specialized-vs-Shared Baselines

见附录 F/G。per-domain 最优与 shared L1DK 的差距 ≤1pp（旧协议）。

## 28. Naive Shared U0

`reports/l3_u0_shared_baseline.md`（附录 I）：

| Domain | AssA | IDF1 | IDSW |
|---|---:|---:|---:|
| DanceTrack | 0.4169 | 0.5694 | 2,588 |
| MOT17 | 0.6050 | 0.5825 | 259 |
| MOT20 | 0.2950 | 0.4012 | 2,406 |
| BDD | 0.2881 | 0.2923 | 11,042 |

Macro AssA 0.4013（fresh 协议），超 L1DK 0.3944。

## 29. Conditional Unified U1

`reports/l3_u1_conditional_pilot.md`（附录 J）：

| Domain | U0 AssA | U1 AssA | Δ |
|---|---:|---:|---:|
| DanceTrack | 0.4169 | 0.4050 | −1.19pp |
| MOT17 | 0.6050 | 0.5859 | −1.91pp |
| MOT20 | 0.2950 | 0.2958 | +0.08pp |
| BDD | 0.2881 | 0.2792 | −0.89pp |

Macro 0.3915（−0.98pp）。Gate A 未通过。

## 30. DanceTrack

U0 ≈ L1DK（0.4169 vs 0.4165）；U1 0.4050（−1.19pp vs U0），
但 U1 IDSW 2528 最优。

## 31. MOT17

U0 0.6050（+1.67pp vs L1DK 0.5883）；U1 0.5859（−1.91pp vs U0）。

## 32. MOT20

U0 0.2950（+1.72pp vs L1DK）；U1 0.2958（+0.08pp vs U0，唯一正域）。

## 33. BDD Full Multi-Class

全类 GT 评估：L1DK 0.2951 / U0 0.2881 / U1 0.2792（fresh 协议）。
U0/U1 均略低于 L1DK，但 IDSW 明显更低（U0 11,042 vs L1DK 12,405）。

## 34. TAO/Open-World

未执行（cache 旧布局）。

## 35. Macro Cross-Domain Result

| Method | Macro AssA |
|---|---:|
| C0 | 0.3380 |
| C1 | 0.3903 |
| C2 | 0.1232 |
| C3 | 0.3054 |
| L1DK | 0.3944 |
| L1DK_d03 | 0.3905 |
| **U0** | **0.4013** |
| U1 | 0.3915 |

## 36. One-Checkpoint Verification

U0/U1 均为单 checkpoint 四域评估（无 dataset-specific 参数）。

## 37. Dataset-Specific Parameter Audit

U0/U1：0 个 dataset-specific head/adapter；阈值统一 0.25、
delta 0.3（全局共享）。z_regime 无 dataset 输入但呈 dataset shortcut
（机制缺陷，非参数缺陷）。

## 38. Leave-DanceTrack-Out

未执行（Gate A 未过；按任务书不执行）。

## 39. Leave-Multiclass/Openworld-Out

未执行。

## 40. Track-All

U0/U1 的 ALL 隐式语义（全部候选关联），未做显式 spec。

## 41. Category Text

未训练（A Gate 未过；spec 设计见附录 H）。

## 42. Open-Vocabulary Text

未训练。

## 43. Visual Prompt

未训练。

## 44. Referring Prompt

未训练；STORM-Bench/RMOT26 已审计（附录 B）。

## 45. Point/Mask Prompt

未训练。

## 46. Cross-Prompt Consistency

未执行。

## 47. Regime Routing Visualization

未做可视化；z_regime 统计见附录 K（shortcut audit）。

## 48. Intra-Dataset Regime Variation

域内 z 标准差 ≈0.26，远小于域间距离（0.76–3.64）。

## 49. Cross-Dataset Regime Alignment

未成立：z 按 dataset 分离，无跨域对齐。

## 50. Dataset-Shortcut Audit

`reports/l3_shortcut_audit.md`（附录 K）：

- domain classifier on z = **96.6%**（随机 25%）；
- **REGIME_ROUTER_DATASET_SHORTCUT CONFIRMED**。

## 51. Specification-only Ablation

未执行（F1）。

## 52. Regime-only Ablation

未执行独立消融（U1 = regime-only，已作为主 pilot）。

## 53. Factorization Ablation

未执行（F0/F3/F4 因 U1 失败无意义）。

## 54. Conditional Computation Ablation

未执行（FiLM 已无正信号，不跑 MoE/hypernetwork）。

## 55. Prompt Backbone Ablation

未执行；审计见附录 H。

## 56. Sampling/Loss Ablation

未执行。

## 57. Why Not L1DK?

L1DK 是固定规则，无学习能力；U0 已在同一协议下超过它
（macro +0.69pp）。

## 58. Why Not Dataset-Specific Trackers?

它们违反 one-checkpoint 约束；U0 证明共享学习可行。

## 59. Why Not SAM3 Alone?

SAM3 覆盖 promptable segmentation/VIS（BURST HOTA 43.3），但
不覆盖标准 box-MOT AC 协议（DanceTrack/MOT17/MOT20 固定检测）；
其 SAM License、HF 登录、CUDA 12.6/torch 2.10 依赖与现有 40G
四卡环境不匹配；未作为主实现。

## 60. Why Not GLEE?

GLEE 是检测/分割 foundation（数亿参数），不是 AC association
核心；且其 text 接口依赖 CLIP 类模型，本项目 env 无 CLIP。

## 61. Why Not OVTR/OVTrack/TRACT?

它们解决 open-vocab MOT（TAO），与本项目 AC 四域协议不同；
可作为 B 轴 comparison，不作为 A 主实现。

## 62. What Is Actually Unified?

逐项回答（真实状态）：

| 项目 | 状态 |
|---|---|
| Architecture 一个？ | U0/U1 各自一个 |
| Checkpoint 一个？ | 是（各模型单 checkpoint） |
| Tracking core 一个？ | 是 |
| Association decoder 一个？ | 是 |
| Prompt encoder 共享？ | 未实现（B 未训练） |
| Dataset-specific heads | 0 |
| Dataset-specific adapters | 0 |
| Dataset-specific thresholds | 0（全局 0.25/0.3） |
| 训练联合？ | 是（四域混合） |
| LODO 无需适配？ | 未验证 |
| 类别 open-vocab？ | 否 |
| visual prompt 共享 track state？ | 否 |

## 63. Failure Cases

见 `reports/l3_failure_analysis.md`（附录 L）：

1. regime 特征与 dataset 共线 → z 学 dataset shortcut；
2. association 可条件化空间小（L2 oracle headroom <0.1pp）；
3. local CE 目标与轨迹级指标不同构。

## 64. Compute / Runtime / Memory

- U0/U1 训练：单卡 40G，峰值 ~5GB，6–7 分钟；
- 四域 AC 评估：约 10 分钟/模型；
- Regime 诊断：CPU，~3 分钟；
- 总占用：未超过 2 张卡（8/9），符合 ≤4 卡约束。

## 65. Scientific Interpretation

最重要的发现：**“按场景动态调节关联证据”的想法，在当前
prediction-side 特征与 local-CE 训练目标下，学到的是 dataset
身份而非可迁移 regime**。U0 的正收益说明多域共享训练本身有价值；
U1 的负收益说明 regime 条件化需要（a）与 dataset 去共线的特征、
（b）轨迹级目标，否则只是 dataset-conditioned 偏置。

## 66. Claim Boundary

- 支持：U0 shared learned 超 L1DK；BDD 多类 GT 可直接使用；
  SAM3/GLEE 覆盖 prompt 统一；
- 不支持：latent regime 有效；prompt 接口为 novelty；LODO 泛化。

## 67. ICLR Readiness Audit

| 维度 | 评级 |
|---|---|
| Novelty | 中（Claim 3 无等价实现，但 pilot 无实证） |
| Technical Quality | 中（协议/审计严谨，shortcut 已公开） |
| Empirical Strength | 低（U1 无正收益，B 未训练） |
| Generalization | 低（无 LODO） |
| Scientific Clarity | 中（负结果清晰，但主线未建立） |

**NOT_READY**。

## 68. Stage Decision

```text
L3_REGIME_NOT_SUPPORTED
REGIME_ROUTER_DATASET_SHORTCUT
```

## 69. Next Single Recommendation

用 **spec-conditioned U0（B 轴，category/compat 输入，无 latent
regime）** 验证“统一对象指定”在共享 core 上的可行性；若有效，
再以“spec 条件化 + 去共线 regime 特征（如 scene-level 光度/
光流代理而非 benchmark 统计）”重试 U1；否则收口 U0 为统一
checkpoint 的工程路线。

## 70. Important Paths

本报告为自包含版本：以下产物完整原文嵌入附录 A–L。

- 附录 A：`docs/l3_unified_task_definition.md`
- 附录 B：`docs/l3_reference_audit.md`
- 附录 C：`reports/l3_novelty_collision_audit.md`
- 附录 D：`docs/l3_protocol_matrix.md`
- 附录 E：`docs/l3_dataset_audit.md`
- 附录 F：`reports/l3_negative_transfer_audit.md`
- 附录 G：`reports/l3_regime_signal.md`
- 附录 H：`reports/l3_spec_backbone_audit.md`
- 附录 I：`reports/l3_u0_shared_baseline.md`
- 附录 J：`reports/l3_u1_conditional_pilot.md`
- 附录 K：`reports/l3_shortcut_audit.md`
- 附录 L：`reports/l3_failure_analysis.md`
- 附录 M：`docs/l3_implementation_evidence.md`
- 工具：`tools/train_l3.py`、`tools/eval_l3.py`、
  `tools/l3_regime_diagnostics.py`、`tools/analyze_l3_routing.py`
- 数据/模型：`outputs/l3/`（checkpoints、trackers、merged、
  regime_diagnostics.json、routing_audit.json）

## 71. Git Commit

提交信息：`Stage L3 complete: regime- and specification-conditioned unified MOT`
（commit hash 见 Git 记录）。


## 附录 A — 统一任务定义

> 来源文件：`docs/l3_unified_task_definition.md`（已嵌入本报告）

### Stage L3 — Unified MOT 任务定义

日期：2026-08-10。

#### 1. 两个正交轴

Unified MOT 被严格定义为两个正交轴：

##### Axis A — Tracking Regime / Domain

- dense same-class（DanceTrack：高同类外观歧义、交叉、非线性运动）；
- standard pedestrian（MOT17）；
- extreme crowd（MOT20）；
- multi-class driving（BDD100K：11 类、ego-motion、5fps 采样）；
- long-tail / open-world / sparse annotation（TAO，延迟纳入）。

##### Axis B — Object Specification

- ALL / track-all（无 prompt）；
- category text（person/car/…）；
- open-vocabulary text；
- referring description（诊断级，STORM-Bench/QTrack 若数据可获取）；
- visual prompt（box 必选；point/mask 视成本）。

#### 2. 模型目标

```text
Tracker(video, specification) -> persistent object trajectories
```

- specification 可以为空（track-all）；
- 同一 core、同一 checkpoint；
- 无 dataset-specific head / adapter / threshold；
- 推理严格 online causal；
- dataset name / path / annotation source 禁止作为输入。

#### 3. 统一学习表述

```text
What to track  -> Object Specification Token (SpecToken)
How to track   -> Latent Tracking Regime Token (z_regime)
TrackState_{t+1} = F_theta(TrackState_t, CurrentObjects_t, SpecToken, z_regime)
```

#### 4. 边界（避免失焦）

- 主输出仍为 box trajectory + ID；mask 只作为 prompt 输入接口
  （mask→初始 spec/object token），除非标准 benchmark 强制 mask 输出；
- 不把 SOT/VOS 无限纳入；promptable segmentation 本身不是 novelty
  （SAM3/GLEE 已覆盖，见 `reports/l3_novelty_collision_audit.md`）；
- 不做 retrospective ID revision / bounded-latency correction；
- 不做 RL 主线（L2 已否决 future-utility RL）。

#### 5. 论文 claim 边界（暂定，实验支持后才可用）

异质 MOT benchmark 的差异不仅是 visual domain，还包括决定
“哪些时间/语义证据可靠”的 tracking regime。naive shared tracker
因此产生负迁移。我们提出 specification × regime 因子分解，
用一个共享 tracking core 统一多域与多对象指定接口，且
无 dataset-specific 参数。

若实验不支持，则退回 `L3_REGIME_NOT_SUPPORTED` 或
`L3_PROMPT_UNIFICATION_PARTIAL`。


## 附录 B — 2025–2026 官方代码审计

> 来源文件：`docs/l3_reference_audit.md`（已嵌入本报告）

### Stage L3 — 2025/2026 官方实现审计

日期：2026-08-10。
原则：只记录实际 clone + 阅读的官方代码；commit 以仓库 HEAD 为准；
不把 README 摘要当实现依据。

#### 0. 审计方法

对每个仓库：

- `git remote get-url origin` 验证官方 URL；
- `git rev-parse HEAD` 记录 commit；
- 阅读 README / LICENSE / 模型定义 / 训练 loss / 数据加载 / 推理与
  tracker 状态 / prompt 编码 / association / query update。

#### 1. SAM 3 / SAM 3.1（Meta，2026）

- 官方仓库：`github.com/facebookresearch/sam3` ✅
- Commit：`96914d24`；License：SAM License（2025-11-19，需单独确认条款）
- 已读文件：
  - `README.md`、`RELEASE_SAM3p1.md`
  - `sam3/model/video_tracking_multiplex.py`（Object Multiplex 联合多目标跟踪）
  - `sam3/model/video_tracking_multiplex_demo.py`（推理状态管理）
  - `sam3/model/sam3_multiplex_tracking.py`、`sam3_multiplex_video_predictor.py`
  - `sam3/model/memory.py`（mask memory）、`multiplex_utils.py`（MultiplexState）
- Input interface：text phrase / exemplar / point / box / mask；
  open-vocabulary 概念（SA-CO 270K concepts）。
- Output interface：masks + boxes + object pointers；video 下检测-分割-跟踪。
- Supports MOT identity？是（多目标、ID 持久、BURST HOTA 43.3/44.5）。
- Supports multi-object？是（SAM 3.1 Object Multiplex，~7x 加速）。
- 如何关联：SAM2 式 memory propagation + 检测器-跟踪器关联启发式
  （“detection-tracker association and other heuristics”）；对象 pointer
  cross-attention；非重叠 mask 约束。
- Conditional computation：无 latent regime；固定 tracking core。
- 多域训练：SA-CO/VEval/SA-V/YT-Temporal/BURST/YTVIS/OVIS/MOSE 等
  promptable 视频域，非标准 box-MOT 域。
- 复用性：prompt encoder / spec 表示可作为 B 的 foundation 参考；
  MOT 身份关联在标准 AC 协议上不覆盖本项目四域。
- 碰撞：**Claim 2（统一 prompt 接口）强烈碰撞**；Claim 1/3 不直接覆盖。

#### 2. GLEE（CVPR 2024 Highlight，ByteDance）

- 官方仓库：`github.com/FoundationVision/GLEE` ✅
- Commit：`f36a49e8`；License：MIT
- 已读文件：
  - `projects/GLEE/glee/GLEE.py`（多数据集 category 表：COCO/LVIS/Obj365/
    OpenImage/BDD/TAO/YTVIS/OVIS/LVVIS/BURST/UVO/SA1B/grounding/RVOS）
  - `projects/GLEE/glee/models/glee_model.py`（tracking contrastive loss
    IDOL-style、匈牙利匹配、query 结构）
  - `glee/data/datasets/bdd100k.py`、`tao.py`、`uni_video_image_mapper.py`
- Input interface：image/video + text（CLIP variants）+ 可扩展任务头
  （det/inst-seg/video-inst-seg/RVOS/grounding）。
- Output interface：boxes/masks + track embeddings。
- Supports MOT identity？是（track embedding + contrastive loss +
  query 匹配；TAO MOT 榜）。
- 如何关联：每帧 query 检测 + `pred_track_embed` 跨帧对比损失 +
  Hungarian；无 latent regime。
- 多域训练：真实多数据集联合训练（含 BDD multi-class 与 TAO）。
- 复用性：联合训练配方、text 接口、BDD/TAO 数据接入。
- 碰撞：**Claim 1（多域统一）与 Claim 2（spec 接口）均部分碰撞**；
  但 GLEE 是检测/分割 foundation，不是标准 box-MOT 的 association
  条件化核心，且无 regime 条件化。

#### 3. OVTR（ICLR 2025）

- 官方仓库：`github.com/jinyanglii/OVTR` ✅
- Commit：`500e72c1`；License：MIT
- 已读文件：`ovtr/models/ovtr.py`、`utils.py`（`protect_track_preds`、
  `text_query` 处理、miss_tolerance/obj_idxes）、`transformer.py`、
  `updater.py`、`matcher.py`
- Input：text category embeddings + detection/track queries。
- Output：open-vocab 类别 + 跟踪框；TAO TETA。
- 关联：MOTR 式 query propagation（detection queries + track queries，
  obj_idxes/disappear_time，track update）。
- 训练目标：DETR set loss + 文本类别匹配（CIP 类别传播）。
- Conditional computation：无 regime。
- 碰撞：Claim 2 open-vocab MOT 方向。

#### 4. OVTrack（CVPR 2023）

- 官方仓库：`github.com/SysCV/ovtrack` ✅
- Commit：`e188b32e`；License：Apache-2.0
- 已读文件：README（VLM distillation + 数据幻觉；TETA 评估协议）
- Input：检测 + VLM 类别查询；Output：open-vocab boxes + IDs。
- 关联：两阶段（检测→关联），无学习型 regime。
- 碰撞：Claim 2 open-vocab 历史方法。

#### 5. Grounded SAM 2（IDEA-Research）

- 官方仓库：`github.com/IDEA-Research/grounded-sam-2` ✅
- Commit：`b7a9c29f`；License：Apache-2.0
- 已读文件：README（Grounding DINO/DINO-X/Florence-2 + SAM2 pipeline）
- 本质：**pipeline，不是单一模型**；无联合训练；MOT 身份仅靠
  SAM2 propagation。

#### 6. SAM2MOT（AAAI 2026，Huawei Cloud）

- 官方仓库：`github.com/TripleJoy/SAM2MOT` ✅（仅 README）
- Commit：`7bdae12c`；License：Apache-2.0
- **代码未发布**（README：Incoming）。只能 paper-guided，
  不得声称复用官方实现。

#### 7. STORM-Bench / STORM（2026，Amazon）

- 官方仓库：`github.com/amazon-science/storm-referring-multi-object-grounding` ✅
- Commit：`0d87c3ba`
- 内容：STORM-Bench（VidOR 派生，90,617 帧、29,933 tracks、
  30,700 expressions，1fps）；STORM 模型（end-to-end MLLM，
  Task-Composition Learning；HOTA 66.7 / IDF1 78.3）**模型代码不在该仓库**。
- Input：referring expression；Output：multi-object trajectories。
- 碰撞：Claim 2 referring MOT 方向（基准已发布，模型代码未见）。

#### 8. QTrack / RMOT26（2026，MIT License）

- 官方仓库：`github.com/gaash-lab/QTrack` ✅
- Commit：`bc746fe2`；License：MIT
- 已读文件：README（query-driven MOT；RMOT26 基准；TPA-PO 策略优化；
  RMOT26 0.30 MCP / 0.75 MOTP）
- Input：reference frame + 文本 query；Output：只跟踪 query 指定的目标。
- 关联：VLM 端到端推理 + 时间一致性；无 latent regime 条件化。
- 碰撞：Claim 2 referring/query-driven MOT 方向。

#### 9. AnyTrack（2026）

- 官方仓库：`github.com/IdolLab/AnyTrack` ✅
- Commit：`7d5ca454`
- 内容：SOT 的多模态（RGB-T/D/E）统一；非 MOT；作 B 的模态参考。

#### 10. 无已验证官方代码的 2026 方法（记录，不当作依据）

- GOVTrack（CVPR 2026，generative OVMOT）：未找到官方 repo。
- COVTrack（ICCV 2025）/ COVTrack++（2026）：adaptive multi-cue
  fusion OVMOT；官方 repo 未见发布（“code will be available”）。
- STORM（ICML 2026，6D tracking）：单目标 6D，与 MOT 无关。

#### 11. Conditional computation / MoE（结构参考）

- Unified Multimodal Visual Tracking with Dual MoE（ICML 2026）：
  SOT 多模态 T-MoE/M-MoE；无 MOT association。
- AHAT（2026）：adaptive hybrid association（复杂运动场景）——
  动态特征融合，非 latent regime。
- 3D MOT scene-adaptive learned thresholds（2026）：按 density/motion
  自动调阈值——与 regime 思想接近，但仅 3D 阈值，非共享条件化计算。
- 结论：condition-aware routing 在视觉任务有结构先例，但
  **没有在标准 box-MOT association 上做 latent regime 条件化的
  已验证官方实现**。

#### 12. 总表

| Method | Year/Venue | Official repo verified | Commit | License | MOT identity | 多目标 | Text prompt | Visual prompt | Open-vocab | Dataset-specific params | Multi-domain train | Conditional computation | Regime adaptation | Novelty collision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SAM 3/3.1 | 2026 Meta | ✅ | 96914d24 | SAM License | ✅ | ✅ | ✅ | ✅ | ✅ | 否 | 是（promptable 视频） | 否 | 否 | Claim 2 强 |
| GLEE | CVPR 2024 | ✅ | f36a49e8 | MIT | ✅ | ✅ | ✅ | 部分 | ✅ | 否 | 是（含 BDD/TAO） | 否 | 否 | Claim 1/2 部分 |
| OVTR | ICLR 2025 | ✅ | 500e72c1 | MIT | ✅ | ✅ | ✅ | 否 | ✅ | 否 | 否（TAO 训练） | 否 | 否 | Claim 2 |
| OVTrack | CVPR 2023 | ✅ | e188b32e | Apache-2.0 | ✅ | ✅ | ✅ | 否 | ✅ | 否 | 否 | 否 | 否 | Claim 2 |
| Grounded SAM 2 | 2024/25 | ✅ | b7a9c29f | Apache-2.0 | ✅ | ✅ | ✅ | ✅ | ✅ | 否 | 否（pipeline） | 否 | 否 | Claim 2 |
| SAM2MOT | AAAI 2026 | ✅（代码未发布） | 7bdae12c | Apache-2.0 | ✅ | ✅ | 否 | ✅ | 否 | 否 | 否 | 否 | 否 | 无 |
| STORM | 2026 | ✅（bench 仓库） | 0d87c3ba | — | ✅ | ✅ | ✅ | 否 | 部分 | 否 | 是 | 否 | 否 | Claim 2 referring |
| QTrack | 2026 | ✅ | bc746fe2 | MIT | ✅ | ✅ | ✅ | 否 | 部分 | 否 | 否 | 否 | 否 | Claim 2 referring |
| AnyTrack | 2026 | ✅ | 7d5ca454 | — | SOT | 否 | 部分 | 部分 | 否 | 否 | 否 | MoE | 否 | B 模态参考 |
| COVTrack/++ | ICCV25/26 | ❌ 无代码 | — | — | ✅ | ✅ | ✅ | 否 | ✅ | 否 | 否 | adaptive fusion | 部分 | Claim 2/3 部分 |
| GOVTrack | CVPR 2026 | ❌ 无代码 | — | — | ✅ | ✅ | ✅ | 否 | ✅ | 否 | 否 | 否 | 否 | Claim 2 |

#### 13. 审计结论

1. **Claim 2（统一 prompt 接口）已被 SAM3/GLEE 强碰撞**：text/point/box/
   mask prompt 统一本身不是 novelty；
2. **Claim 1（多域统一）被 GLEE 部分碰撞**：GLEE 联合训练 BDD/TAO 等，
   但不是在标准 box-MOT AC 协议下的 association 核心统一；
3. **Claim 3（latent regime 条件化 tracking computation）未发现
   已验证官方等价实现**：最接近的是 3D 场景自适应阈值 / 自适应融合，
   但都不是共享模型内部的 latent regime 条件化。


## 附录 C — Novelty Collision 审计

> 来源文件：`reports/l3_novelty_collision_audit.md`（已嵌入本报告）

### Stage L3 — Novelty Collision Audit（Gate 0）

日期：2026-08-10。

#### 1. 三个 Claim 候选

- Claim-Candidate 1：一个 checkpoint 统一 heterogeneous box-MOT
  domains（DanceTrack/MOT17/MOT20/BDD multi-class/TAO）。
- Claim-Candidate 2：同一个 tracking core 统一 class-agnostic /
  category / open-vocab / referring / visual prompt object specification。
- Claim-Candidate 3：模型用 latent tracking regime 条件化 tracking
  computation，而不是 dataset-specific specialization。

#### 2. 逐项碰撞结论

##### Claim 1：统一 box-MOT domains

已核实：

- GLEE（CVPR 2024）在**检测/分割 foundation** 层面联合训练多数据集
  （含 BDD multi-class、TAO），是部分碰撞；
- SAM 3.1 在 promptable video 域（BURST/YTVIS/OVIS/SA-V）统一多目标
  跟踪，但不是标准 box-MOT 的检测集固定 AC 协议；
- MOTIP/TDLP/CAMELTrack 等均为单域或两域 trained association，无
  四域统一 checkpoint 的主结果。

结论：**在标准 box-MOT Association-Controlled 协议下，未发现
“一个 checkpoint 统一 DanceTrack/MOT17/MOT20/BDD multi-class 且
无 dataset-specific 参数”的已验证官方实现**。但必须把“统一”定义为
标准 MOT 协议，不能与 GLEE 的检测 foundation 混为一谈。

##### Claim 2：统一 object specification 接口

已核实：

- SAM 3 / SAM 3.1：text/point/box/mask/exemplar 统一到图像+视频的
  检测-分割-跟踪，open-vocabulary；BURST HOTA 43.3（SAM3.1）；
- GLEE：text（CLIP）+ image/video + 多任务头，含 referring/grounding；
- OVTR/OVTrack：open-vocab MOT；
- STORM/QTrack：referring/query-driven MOT（2026）。

结论：**Claim 2 单独不成立（强碰撞）**。本项目不能把
“统一 prompt 接口”当核心 novelty；只能把 spec conditioning 当作
与 MOT identity/regime 正交的第二个轴，并在与 SAM3/GLEE 的
差异（标准 MOT 身份关联、AC 协议、regime 条件化）上建立边界。

##### Claim 3：latent tracking regime 条件化

已核实：

- 无已验证官方实现把“从 prediction-side 状态估计 latent regime，
  并条件化共享 association core”用于标准 box-MOT；
- 最近邻：3D MOT 场景自适应阈值（2026）、AHAT 自适应混合关联、
  COVTrack 自适应多 cue 融合——都不是共享模型内部 latent regime
  因子分解，且无官方代码；
- MoE/conditional routing 在 SOT 多模态（ICML 2026 Dual MoE）存在，
  但无 MOT association 版本。

结论：**Claim 3 是三个 claim 中新颖性最强、唯一未见直接等价实现的
候选**。但必须由实验证明：(a) naive shared 有负迁移；(b) regime
条件化减轻负迁移；(c) 不是 dataset router。

#### 3. Gate 0 判定

```text
NO_DIRECT_EQUIVALENT_VERIFIED_METHOD_FOUND
（对 Specification × Regime Factorization 在标准 box-MOT 的完整组合）
```

边界与风险：

1. Claim 2 有明确 collision（SAM3/GLEE/OVTR/QTrack/STORM），
   最终 claim 必须把“统一接口”降为第二个轴，不得作为独立 novelty；
2. 不得使用 first / foundation tracker / universal tracker 表述；
3. 若 pilot 无法证明 regime 条件化减轻负迁移，则回到
   `L3_REGIME_NOT_SUPPORTED`，不硬写 novelty。


## 附录 D — 协议矩阵

> 来源文件：`docs/l3_protocol_matrix.md`（已嵌入本报告）

### Stage L3 — 协议矩阵

日期：2026-08-10。

| Dataset | Regime | Classes | FPS/时间 | 标注密度 | Main metric | Prompt support | 是否可共同训练 | Train/Eval 角色 |
|---|---|---|---|---|---|---|---|---|
| DanceTrack | dense same-class，外观歧义高，非线性运动/交叉 | 1（person） | 30fps，帧间隔 1 | dense box+ID | HOTA/AssA/IDF1/IDSW（官方 TrackEval） | ALL / person text | ✅ | calib=train；val=主评估 |
| MOT17 | standard pedestrian | 1（person） | 30fps（采样） | dense box+ID | HOTA/AssA/IDF1/IDSW | ALL / person text | ✅ | train + 跨域评估 |
| MOT20 | extreme crowd/遮挡 | 1（person） | 25fps（采样） | dense box+ID | HOTA/AssA/IDF1/IDSW | ALL / person text | ✅ | train + 跨域评估 |
| BDD100K | multi-class driving，ego-motion，5fps | 11 类 | 5fps 采样 | dense box+ID（全类） | per-class AC + macro；官方 BDD eval（若可行） | ALL / category text / open-vocab | ✅ | train + 多类评估 |
| TAO / C-TAO | long-tail/open-world/sparse | 数百类（TAO categories） | 1fps 级别 | sparse federated | TETA / official TAO | ALL / category / open-vocab | 待 cache 修复 | 延迟纳入 |
| STORM-Bench | referring MOT（VidOR） | 80 类 | 1fps | referring expression→tracks | HOTA/IDF1（官方 bench） | referring text | 待数据下载 | 诊断级 |
| RMOT26 | query-driven MOT | — | — | grounded queries | MCP/MOTP（官方） | text query | 待数据下载 | 诊断级 |

#### 统一训练采样原则

- dataset-balanced + video-balanced + category/long-tail-balanced +
  regime-balanced + prompt-type-balanced；
- loss 按任务/域归一化；
- 禁止 raw concat；
- 禁止 dataset ID 作为输入。

#### AC 协议

- 所有主要对比使用 Association-Controlled：同一候选集
  （boxes/scores/features）、同帧数、同输出数量，只改 IDs；
- hash 校验候选集一致性；
- 主要指标 TrackEval（HOTA/AssA/IDF1/IDSW）。

#### 输出

`outputs/l3/manifests/`（构建脚本生成）。


## 附录 E — 数据集审计

> 来源文件：`docs/l3_dataset_audit.md`（已嵌入本报告）

### Stage L3 — 数据集审计

日期：2026-08-10。只审计本地实际存在的资源，不假设。

#### 1. DanceTrack

- 本地：`/data1/LWR/vranlee/DATASETS/JDE/dancetrack`（train/val GT）；
- manifest：`outputs/l1_c/fixed_candidate_manifest/dancetrack_{calibration,val,train}.jsonl`；
- cache：`/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1A/cache_dla`
  （DanceTrack 专用）；
- 规模：val raw pkl 25 视频 / 25,508 帧（val split 40 序列，
  其中 25 有 cache）；calibration 8 视频 / 8,024 帧；
- regime：密集同类、外观歧义高、非线性运动、交叉；
- 标注：box + GT ID，person only，dense；
- 角色：calibration=训练（oracle/U0/U1），val=主评估。

#### 2. MOT17

- manifest：`fixed_candidate_manifest/mot17_train.jsonl`；
- cache：`LocateMOT_L1B/cache_dla/mot17`；
- 规模：3 视频（MOT17-02/04/09-SDP），240 帧（采样）；
- regime：标准 pedestrian，中等密度；
- 标注：dense box + ID；
- 角色：训练 + 跨域评估。

#### 3. MOT20

- manifest：`fixed_candidate_manifest/mot20_train.jsonl`；
- cache：`LocateMOT_L1B/cache_dla/mot20`；
- 规模：2 视频，160 帧；
- regime：极端 crowd / 遮挡；
- 标注：dense box + ID；
- 角色：训练 + 跨域评估。

#### 4. BDD100K（多类）

- manifest：`fixed_candidate_manifest/bdd100k_train.jsonl`；
- cache：`LocateMOT_L1B/cache_dla/bdd100k`；
- 规模：200 视频 / 8,001 帧（5fps 采样）；
- **GT 已含 11 类**：bicycle/bus/car/motorcycle/other person/
  other vehicle/pedestrian/rider/trailer/train/truck；
- `matched` 字段含全类候选匹配（5,191/8,001 帧 ≥2 个匹配）；
- 现有 `outputs/l1_d/raw/bdd100k_train.pkl` 的 `cand_gt` 即全类；
- regime：多类驾驶、ego-motion、5fps 大时间间隔、尺寸分布广；
- 角色：多类训练 + 多类评估（按类过滤 + macro，或官方 BDD eval）。

#### 5. TAO / C-TAO

- manifest：`fixed_candidate_manifest/tao_amodal_train.jsonl`；
- 规模：105 视频 / 4,200 帧；有候选帧 2,256；
- cache：`LocateMOT_L1B/cache_dla/tao_amodal/`，但为**旧布局**
  （`train/{BDD,AVA,YFCC100M,HACS,LaSOT}/...`），与 manifest 的
  `cache_key` 不匹配；未发现 `.complete` 标记；
- `/data3/testdata/vranlee/.MOTSynth.partial/C-TAO/` 有 C-TAO
  base/novel 类别文件（清单级）；
- 结论：TAO 为 sparse/federated annotation，评测须用官方 TETA/TAO
  协议；**当前 cache 不可直接复用，Stage L3 先文档记录，延迟到
  A 的四域 pilot 之后**。

#### 6. 其他（本地状态）

- YTVIS/MOSE：cache_dla 下有缓存目录，但无 L3 manifest，未纳入；
- DAVIS/BURST：仅 GLEE_PMOT 项目的评估 CSV（其他项目只读参考），
  无本项目缓存；
- STORM-Bench / RMOT26：官方基准已 clone（`references/l3/`），
  数据未下载（体积大），referring 仅作诊断级；
- MOTSynth：禁止使用。

#### 7. 结论

L3 主实验最小集合：

- DanceTrack calibration（训练）+ val（评估）；
- MOT17 / MOT20（训练 + 评估）；
- BDD multi-class（训练 + 多类评估）；
- TAO：cache 修复后纳入 open-world 证据。


## 附录 F — Naive Shared 负迁移审计

> 来源文件：`reports/l3_negative_transfer_audit.md`（已嵌入本报告）

### Stage L3 — Naive Shared 负迁移审计

日期：2026-08-10。

#### 1. 定义

naive shared = 用同一固定规则/同一 checkpoint 跨全部域。若每个域的最优
规则不同，则任何单一 shared 规则必然在某域负迁移。

#### 2. 每域最优方法（官方 TrackEval AC，L2 baseline 矩阵）

| Domain | 最优 | AssA | 次优 | AssA | 差距 |
|---|---:|---:|---:|---:|---:|
| DanceTrack val | C1（Motion） | 0.4193 | L1DK | 0.4165 | +0.28pp |
| MOT17 | L1DK | 0.6010 | C1 | 0.5530 | +4.80pp |
| MOT20 | L1DK_d03（EGRA） | 0.2864 | C1 | 0.2869 | +0.05pp |
| BDD | L1DK | 0.3292 | C1 | 0.3019 | +2.73pp |

单一 shared 选择（L1DK）相对 per-domain 最优的负迁移：

- DanceTrack −0.28pp；MOT17 0；MOT20 −0.85pp；BDD 0；
- macro 负迁移 ≈ −0.28pp（若 per-domain 最优为 C1/EGRA/L1DK/L1DK）。

结论：**负迁移存在但幅度小**；更强的是“不同域/regime 下方法偏好
确实不同”（见 `reports/l3_regime_signal.md`）。

#### 3. 为什么幅度小

L1DK 的 0.4 IoU + 0.2 PBD + 0.4 Kalman-motion 线性融合是强先验，
在任何单域都不是最差；这既是优点（共享稳定），也意味着
“固定规则”的失败不是灾难性的，需要更细粒度（regime 内）的证据
来支撑条件化动机。

#### 4. 与 L2 oracle 的关系

L2 证明：即使未来 oracle 也不能在 AC 协议内大幅提升 L1DK 的整视频
AssA。因此 L3 的 U1 目标不是“在 L1DK 上再涨 2pp”，而是：

- 统一模型在 4 域同时达到 per-domain 强基线的水平（消除负迁移）；
- 多类 BDD 与 spec/prompt 接口的统一能力；
- 跨域 regime 条件化的机制可解释性。


## 附录 G — Regime 信号审计

> 来源文件：`reports/l3_regime_signal.md`（已嵌入本报告）

### Stage L3 — Latent Regime 信号审计

日期：2026-08-10。

#### 1. 方法

用 prediction-side 状态特征（候选数、IoU 歧义、PBD 歧义、运动代理、
尺寸、时间间隔、语义多样性）把帧切成 regime 桶；对每个桶，用
保存的 AC tracker 输出计算 H=16 windowed AssA（与官方 TrackEval
公式一致），比较 IoU/Motion/L1DK/EGRA 的偏好。

数据：DanceTrack val 3,000 窗口、MOT17 30、MOT20 20、BDD 693。
（MOT17/MOT20 窗口少，结论以 DanceTrack/BDD + MOT17 大差距桶为准。）

#### 2. 关键结果（每轴 marginal：lo/hi 桶内平均 windowed AssA）

##### DanceTrack val

| Regime | best | AssA | L1DK | C1 | EGRA |
|---|---:|---:|---:|---:|---:|
| n_cand=lo | EGRA | 0.9592 | 0.9590 | 0.9558 | 0.9592 |
| n_cand=hi | L1DK | 0.9474 | 0.9474 | 0.9466 | 0.9472 |
| iou_amb=hi | EGRA | 0.9361 | 0.9358 | 0.9331 | 0.9361 |
| pbd_amb=hi | EGRA | 0.9696 | 0.9680 | 0.9660 | 0.9696 |
| motion=hi | EGRA | 0.9405 | 0.9393 | 0.9371 | 0.9405 |

DanceTrack 上差异小（0.1–0.5pp），L1DK/EGRA 稳定占优。

##### MOT17（30 窗口，小样本但有强反差）

| Regime | best | AssA | L1DK | C1 | EGRA |
|---|---:|---:|---:|---:|---:|
| motion=hi | **C1** | 0.6400 | 0.6296 | 0.6400 | 0.6215 |
| motion=lo | L1DK | 0.8526 | 0.8526 | 0.8138 | 0.8461 |
| iou_amb=hi | **C1** | 0.6519 | 0.5853 | 0.6519 | 0.5388 |
| n_cand=lo | EGRA | 0.6182 | 0.5726 | 0.6081 | 0.6182 |

MOT17 的 regime 反差最明显：高运动/高 IoU 歧义时 Motion 规则胜
（最多 +6.7pp），低运动时 L1DK 胜，低密度时 EGRA 胜。

##### BDD（693 窗口）

| Regime | best | AssA | L1DK | C1 | EGRA |
|---|---:|---:|---:|---:|---:|
| n_cand=hi | **C0** | 0.5572 | 0.5534 | 0.5562 | 0.5428 |
| n_cand=lo | L1DK | 0.5396 | 0.5396 | 0.5199 | 0.5351 |
| pbd_amb=hi | EGRA | 0.5152 | 0.5141 | 0.4995 | 0.5152 |
| size=hi | L1DK | 0.6787 | 0.6787 | 0.6646 | 0.6682 |
| size=lo | L1DK | 0.4422 | 0.4422 | 0.4356 | 0.4376 |

BDD 上 L1DK 多数桶最优，但高密度桶 C0（纯 IoU）超过 L1DK，
高 PBD 歧义桶 EGRA 略优。

#### 3. 结论

**REGIME_SPECIALIZATION_SIGNAL_SUPPORTED（弱到中等）**：

1. 不同 regime 桶的最优方法确实不同（MOT17 反差最大，BDD/DanceTrack
   差异较小）；
2. 不存在一个固定规则在所有 regime 桶都严格占优；
3. 但 L1DK 是强共享先验，负迁移绝对值不大 → regime 条件化必须
   在“保持强先验 + 按 regime 微调证据权重”的设定下验证，而不是
   推翻基座。

风险：MOT17/MOT20 窗口样本少，正式结论以 DanceTrack/BDD 为主；
U1 pilot 的收益预期应设为“消除 MOT17 高运动桶的 0.5–6pp 差距 +
BDD 高密度桶差距”，而不是全域大幅提升。


## 附录 H — Spec Backbone 审计

> 来源文件：`reports/l3_spec_backbone_audit.md`（已嵌入本报告）

### Stage L3 — Object Specification Backbone 审计

日期：2026-08-10。

#### 1. 候选

##### S0 — LocateAnything / PBD（现有）

- 优点：现有 PBD cache/manifest 工程成熟（L0–L2 全部基于它）；
- 缺点：text 提示非原生；普通 grounding LoRA 已失败
  （LORA_PBD_DEGRADED）；视觉 prompt（box/point/mask）无原生编码；
- 定位：继续作为 object/track token 表示，spec 用额外轻量编码。

##### S1 — SAM3（Meta 2026）

- 能力：text/point/box/mask/exemplar 统一编码，open-vocab；
- 限制：SAM License（需确认条款）、HF 登录、CUDA 12.6 + torch 2.10
  + flash-attn-3；本环境 torch 2.x/CUDA 12.1 不匹配；
- 定位：B 的强 comparison；不作为默认（工程成本高）。

##### S2 — GLEE（CVPR 2024，MIT）

- 能力：text（CLIP 类）+ image/video 多任务；
- 限制：完整模型大；env 无 CLIP；且 GLEE 是检测 foundation，
  与 AC association 核心叠加成本高；
- 定位：comparison/消融。

#### 2. Pilot 采用

本阶段 B 未进入训练（A Gate 未过）。若继续，最小实现：

- spec = learned embedding（ALL + BDD 11 类 + person + OPEN，
  见 `locatemot/models/l3_unified.py::SPECS`）；
- 候选类别兼容输入：`cand_spec_compat`（0/1）附加到 candidate
  features（类别来自检测侧；评估时可用候选 GT 类别过滤计算指标）；
- open-vocab：transformers 提供 frozen text encoder 作为扩展，
  不阻塞。

#### 3. 结论

- S0 作为主 object token；spec 编码器从轻量 learned embedding 起步；
- S1/S2 仅在需要“open-vocab text / visual prompt”强证据时评估，
  且需处理 License/环境限制；
- 当前不因 SAM3/GLEE 存在就放弃 B，但 B 的 novelty 边界已收缩
  （见 `reports/l3_novelty_collision_audit.md`）。


## 附录 I — U0 Shared Baseline

> 来源文件：`reports/l3_u0_shared_baseline.md`（已嵌入本报告）

### Stage L3 — U0：Naive Shared 学习型关联核心

日期：2026-08-10。

#### 1. 定义

U0 = L1DAssociator（set-level transformer + 有界残差，与 L1-D EGRA
同架构），在 DanceTrack calibration + BDD(11 类) + MOT17 + MOT20
联合数据上训练 30 epochs（~6,300 步，batch 64），无 dataset-specific
参数，推理用同一 checkpoint。

#### 2. 结果（四域 AC，统一 fresh per-video 协议，官方 TrackEval）

| Domain | AssA | IDF1 | IDSW |
|---|---:|---:|---:|
| DanceTrack val | 0.4169 | 0.5694 | 2,588 |
| MOT17 | 0.6050 | 0.5825 | 259 |
| MOT20 | 0.2950 | 0.4012 | 2,406 |
| BDD（11 类 GT） | 0.2881 | 0.2923 | 11,042 |

#### 3. 与强基座对比（同协议）

| Domain | L1DK AssA | U0 AssA | Δ | L1DK IDSW | U0 IDSW |
|---|---:|---:|---:|---:|---:|
| DanceTrack | 0.4165 | 0.4169 | +0.04pp | 2,558 | 2,588 |
| MOT17 | 0.5883 | 0.6050 | **+1.67pp** | 280 | **259** |
| MOT20 | 0.2778 | 0.2950 | **+1.72pp** | 2,603 | **2,406** |
| BDD | 0.2951 | 0.2881 | −0.70pp | 12,405 | **11,042** |

Macro AssA：U0 0.4013 vs L1DK 0.3944（+0.69pp）。

#### 4. 结论

1. **naive shared 学习型核心（U0）已超过 L1DK 固定规则**：
   MOT17/MOT20 明显正向，BDD 略降，DanceTrack 持平；
2. 说明“负迁移”不是灾难性的：共享学习本身能吸收多域数据；
3. U0 是 L3 真正的 shared dense baseline，U1 必须在此基础上证明
   regime 条件化增益。

#### 5. 协议说明

本表使用 per-video fresh OnlineTracker（L1 协议），与 L2 报告中
MOT17/MOT20/BDD 的 L1DK 数字（旧 shared-tracker 输出）不完全一致；
差异源于旧输出跨视频共享 tracker 状态，TrackEval 对 ID 重标号后
AssA 仍受关联历史影响。L3 所有方法在同一 fresh 协议下比较。


## 附录 J — U1 Conditional Pilot

> 来源文件：`reports/l3_u1_conditional_pilot.md`（已嵌入本报告）

### Stage L3 — U1：Regime-Conditioned 条件化核心（Pilot）

日期：2026-08-10。

#### 1. 定义

U1 = L3Associator：U0 同架构 + RegimeEncoder
（prediction-side 统计 → z_regime 32 维）+ FiLM 条件化
（track/cand token 与 encoder 输出）+ z 注入 pair head。
训练数据/步数与 U0 完全一致（30 epochs，seed 20260806）。

Regime 输入（causal）：候选数、IoU/PBD 歧义、运动代理、gap、
track age/hits、margin、base 竞争统计。无 dataset ID。

#### 2. 结果（四域 AC，同协议）

| Domain | U0 AssA | U1 AssA | Δ | U0 IDF1 | U1 IDF1 | U0 IDSW | U1 IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|
| DanceTrack | 0.4169 | 0.4050 | −1.19pp | 0.5694 | 0.5618 | 2,588 | **2,528** |
| MOT17 | 0.6050 | 0.5859 | −1.91pp | 0.5825 | 0.5726 | **259** | 274 |
| MOT20 | 0.2950 | 0.2958 | +0.08pp | 0.4012 | 0.3971 | **2,406** | 2,436 |
| BDD | 0.2881 | 0.2792 | −0.89pp | 0.2923 | 0.2861 | **11,042** | 11,027 |

Macro AssA：U1 0.3915 vs U0 0.4013（−0.98pp）。

#### 3. Gate 判定

任务书 Gate A：U1 在 ≥3 个 heterogeneous domains 总体优于 U0。

**不满足**：U1 只在 MOT20 微正（+0.08pp），DanceTrack/MOT17/BDD
均下降。

#### 4. 原因分析

1. z_regime 学成了 dataset shortcut：域分类器在 z 上的准确率 96.6%
   （随机 25%），域质心距离远大于域内标准差（见
   `reports/l3_shortcut_audit.md`）；
2. Regime 特征（density/gap 等）与 dataset 天然相关（BDD 5fps 大 gap、
   DanceTrack 30fps），FiLM 条件化退化为 dataset-conditional 偏置；
3. 即使作为 dataset 偏置，也没有带来跨域收益——说明该 pilot 的
   regime 条件化对 association 学习无正信息。

#### 5. 结论

```text
L3_REGIME_NOT_SUPPORTED（pilot）
REGIME_ROUTER_DATASET_SHORTCUT（z 与 dataset 强相关）
```

不进入正式训练；不堆容量/不换 MoE 强行挽救。


## 附录 K — Regime Token Shortcut 审计

> 来源文件：`reports/l3_shortcut_audit.md`（已嵌入本报告）

### Stage L3 — Regime Token Shortcut 审计

日期：2026-08-10。

#### 1. 方法

对 U1 的 z_regime（32 维）做三类检查：

1. 域质心距离 vs 域内标准差（dataset separation）；
2. z 与 prediction-side density 的最大相关；
3. 在 z 上训练 domain 分类器（80/20 split，Logistic Regression）。

样本：每域 2,000 个训练样本的 collated batch 前向。

#### 2. 结果

| Domain | n | z norm | intra std(mean) | max|corr| with density |
|---|---:|---:|---:|---:|
| dancetrack | 2,000 | 2.19 | 0.261 | 0.294 |
| bdd | 2,000 | 3.24 | 0.258 | 0.470 |
| mot17 | 2,000 | 3.64 | 0.293 | 0.515 |
| mot20 | 2,000 | 0.76 | 0.259 | 0.751 |

- 域质心两两距离：0.76–3.64（最大 MOT17↔BDD 3.64）；
- 域内标准差均值 ≈ 0.26–0.29，远小于域间距离；
- **domain classifier accuracy = 96.6%**（随机 25%）。

#### 3. 结论

```text
REGIME_ROUTER_DATASET_SHORTCUT CONFIRMED
```

1. z_regime 主要编码 dataset 身份（regime 特征与 dataset 天然相关：
   BDD 5fps 大 gap、DanceTrack/MOT17/MOT20 密集）；
2. 该 shortcut 未带来任何跨域收益（U1 < U0）；
3. 即使增加 anti-shortcut 正则，pilot 也已证明 regime 条件化在当前
   association 任务上没有可测量正信号；
4. 结论：L3 主方法（latent regime conditioning）**在本次协议下不成立**，
   不得把 dataset-correlated z 当作 regime 泛化证据。


## 附录 L — 失败分析

> 来源文件：`reports/l3_failure_analysis.md`（已嵌入本报告）

### Stage L3 — Failure Analysis

日期：2026-08-10。

#### 1. 失败结论

```text
L3_REGIME_NOT_SUPPORTED（pilot）
REGIME_ROUTER_DATASET_SHORTCUT
```

Stage L3 主方法（latent tracking regime 条件化 shared core）在
四域 AC pilot 上未通过 Gate A：U1 相对 U0 只在 MOT20 微正
（+0.08pp），DanceTrack −1.19pp、MOT17 −1.91pp、BDD −0.89pp；
macro AssA −0.98pp。

#### 2. 证据链

1. Regime 信号审计（`reports/l3_regime_signal.md`）：
   per-regime 方法偏好确实变化（MOT17 高运动桶 C1 胜 +6.7pp、
   BDD 高密度桶 C0 胜、低密度桶 EGRA 胜），但幅度小、样本少；
2. U0（naive shared learned）已超 L1DK：macro AssA 0.4013 vs 0.3944
   （MOT17 +1.67pp、MOT20 +1.72pp、BDD −0.70pp、DanceTrack +0.04pp）；
3. U1（regime 条件化）反而低于 U0：macro 0.3915；
4. Routing shortcut 审计：z_regime 的 domain classifier 准确率 96.6%，
   域质心距离远大于域内标准差 → z 主要编码 dataset 身份；
5. B（spec/prompt）未接入训练：因为 A 的 Gate 未通过，且 SAM3/GLEE
   已覆盖 prompt 接口统一（`reports/l3_novelty_collision_audit.md`）。

#### 3. 根因

##### 3.1 regime 特征与 dataset 天然共线

BDD 5fps → gap 大；DanceTrack/MOT17/MOT20 → gap 1。density/PBD 歧义
也与 benchmark 强相关。用这些统计学 z，FiLM 必然学到 dataset 偏置。

##### 3.2 association 任务的可条件化空间小

L2 oracle 已证明 L1DK 的整视频 AC headroom < 0.1pp；U0 又吸收了
多域数据的大部分可学信号。剩下的 regime 差异（MOT17 高运动桶）幅度
大但窗口样本极少（30 窗口），不足以驱动端到端训练。

##### 3.3 训练目标仍是 local correctness

U0/U1 都用 row/col CE（local）。L2 已证明 local 与 trajectory utility
不同构；因此 regime 条件化即使学到了，也不能保证改善 TrackEval。

#### 4. 为什么不做的事

- 不换 MoE / 24 层 / 500M（任务书禁止失败后堆容量）；
- 不加入 dataset 平衡重采样后重试 U1（shortcut 已确认，重采样不能
  消除 regime 特征与 dataset 共线）；
- 不把 dataset-specific adapter 作为主结果；
- 不把 prompt 接口硬塞进已失败的核心。

#### 5. 可保留的科学产出

1. **U0 是有效的 shared learned baseline**：一个 checkpoint 在
   DanceTrack/MOT17/MOT20/BDD（多类 GT）上达到/超过 L1DK
   （macro AssA +0.69pp），且无 dataset-specific 参数；
2. 负迁移审计：naive shared 的负迁移小（per-domain 最优差 ≤1pp），
   说明 L3 的“latent regime 必要性”在当前 AC 协议下证据不足；
3. 多类 BDD 协议：现有 manifest 已含 11 类 GT，可直接多类评估；
4. 2025/2026 审计：Claim 3（latent regime）未见等价实现，但 pilot
   证明其在本协议下无正收益——novelty 空洞不等于方法有效。

#### 6. 下一步建议（单一）

若继续 Unified MOT 主线，应先回答“U0 的 BDD −0.7pp 与 DanceTrack
IDSW +30 的来源”，用 **类别/密度感知的 spec-conditioned U0**
（B 轴，不依赖 latent regime）验证统一接口；否则回到 U0 作为
统一 checkpoint 的工程收口（full tracker + LODO 基线）。


## 附录 M — 实现证据

> 来源文件：`docs/l3_implementation_evidence.md`（已嵌入本报告）

### Stage L3 — Implementation Evidence

日期：2026-08-10。

#### 1. U0：共享 dense association core

- Module：`locatemot/models/l1d_association.py::L1DAssociator`
- Scientific purpose：多域共享 set-level 关联基线（local CE）。
- Reference：L1-D EGRA（本项目自研，基于 CAMELTrack/TDLP set-level
  竞争设计，commit 46a74bb / 50344b92 已审计）。
- Files inspected：`tools/train_l3.py`、`locatemot/models/l1d_association.py`
- Mechanism adopted：row/col CE + reliability BCE + base preservation。
- Mechanism changed：无（复用）。
- Why：U0 必须与 L1-D EGRA 可比。

#### 2. U1：RegimeEncoder + FiLM 条件化

- Module：`locatemot/models/l3_unified.py`
  - `RegimeEncoder`：48 维 prediction-side 统计 → 32 维 z_regime；
  - `L3Associator`：FiLM（track/cand token、encoder 输出）+ z 注入
    pair head。
- Scientific purpose：验证“how to track”可由 latent regime 条件化。
- Reference（结构）：condition-aware routing / FiLM 常见于
  conditional vision（ICML 2026 Dual MoE 等，仅结构参考；
  无 MOT association 等价实现，见 `docs/l3_reference_audit.md`）。
- Files inspected：`locatemot/models/l3_unified.py`、
  `tools/analyze_l3_routing.py`
- Mechanism adopted：prediction-side stats（density/IoU/PBD ambiguity/
  motion/gap/age/hits/margin/competition），无 GT/future/dataset ID。
- Mechanism changed：无（首版即 FiLM；未做 MoE/hypernetwork，因
  pilot 已无正信号）。
- Why change：n/a。
- License：clean reimplementation（无外部代码复制）。

#### 3. 评估管线

- `tools/eval_l3.py`：OnlineTracker AC shell + L1DK base 权重
  （0.4/0.2/0.4，thr 0.25，delta 0.3），输出同候选集只改 ID。
- `tools/run_l1d_trackeval.py`：官方 TrackEval。
- 协议：per-video fresh OnlineTracker（L1 定义），所有方法一致。

#### 4. 关键实现决策记录

1. Regime 输入全部 causal：候选统计、历史统计、base 竞争；无未来。
2. 禁止 dataset ID：训练/推理均无 dataset 输入。
3. U1 与 U0 同数据、同步数、同 seed，保证对比公平。
4. 结果：U1 未过 Gate；z_regime 呈 dataset shortcut
   （domain classifier 96.6%），详见 `reports/l3_shortcut_audit.md`。

---
（本报告为自包含版本：附录 A–M 为各产物完整原文。）
