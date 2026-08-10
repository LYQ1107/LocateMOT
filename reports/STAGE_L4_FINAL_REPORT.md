# Stage L4 Final Report

日期：2026-08-11。项目：LocateMOT（`/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`）。

## 1. Executive Summary

Stage L4 把 Unified MOT 重新定义为两轴合成问题：

- Axis A：一个 checkpoint 统一异质 box-MOT 域
  （DanceTrack / MOT17 / MOT20 / BDD 11-class / TAO）；
- Axis B：同一个 shared identity process 统一 object specification
  （ALL / category / instance；box / point / visual / referring 因
  缺少 verified 官方实现而 NOT_EXECUTED）。

科学问题：

```
T_theta(R_s(X)) ≈ R_s(T_theta(X))   （common objects 上，permutation-invariant）
```

关键结果：

1. **Problem Signal 成立**：frozen U0 在 BDD category 上的
   P0-vs-P1 identity drift 为 33–67%，DanceTrack person/instance 为
   32%/31%，TAO 为 14–24%；pre-filter（P1）通常显著降低 IDSW
   （Dance instance AssA 0.559→0.841，IDSW 799→72）；
2. **文献/官方代码审计**：未发现直接等价方法
   （`NO_DIRECTLY_EQUIVALENT_VERIFIED_METHOD_FOUND`）；
3. **Pilot 失败**：A2（spec 条件化）、A5（row/col KL + state
   consistency）、A5p（partition co-assignment MSE，一次最小修正）
   都没有降低 cross-spec drift，全部 20 epochs、U0 初始化、paired
   views（15,851 pairs）；
4. ALL 模式官方 TrackEval 保持 U0 水平（macro AssA 0.4013），
   说明失败不是「模型崩了」，而是**机制没有学到时间级身份等价**；
5. TAO cache 已恢复（105 视频全部可读）。

Stage Decision：

```text
L4_SPEC_RESTRICTION_SIGNAL_SUPPORTED（问题真实存在）
L4_PILOT_GATE_FAIL
L4_NOT_SUPPORTED（pilot mechanism）
ICLR readiness：NOT_READY
```

## 2. Why L3 Regime Routing Was Closed

Stage L3 的 U1（latent regime 条件化）macro AssA 0.3915，低于 U0
0.4013（−0.98pp）；z_regime 的 domain classifier 达 96.6%，
即学成了 dataset shortcut。Stage L4 不再继续 U1/MoE/anti-domain loss，
也不再堆 prompt 接口。

## 3. Why U0 Is the New Starting Point

U0（L1DAssociator，0.49M）是 L3 的 shared learned dense baseline，
四域 fresh per-video AC：

| Domain | AssA | IDF1 | IDSW |
|---|---:|---:|---:|
| DanceTrack val | 0.4169 | 0.5694 | 2,588 |
| MOT17 | 0.6050 | 0.5825 | 259 |
| MOT20 | 0.2950 | 0.4012 | 2,406 |
| BDD 11-class | 0.2881 | 0.2923 | 11,042 |
| Macro | 0.4013 | — | — |

## 4. Unified MOT: Axis A × Axis B

- Axis A：标准 box-MOT association-controlled 协议（candidates 固定，
  只改 IDs）；
- Axis B：specification = set-restriction operator，决定 WHAT to
  track；shared identity core 决定 HOW to track；
- 同一个真实对象不应因为 specification 表达形式或候选子集变化而任意
  改变 persistent identity trajectory。

## 5. Scientific Question

`T_theta(R_s(X)) ≈ R_s(T_theta(X))`。等价性在 common objects 上使用
permutation-invariant co-identity agreement 度量（不比较 raw track
integer ID）。

## 6. Formal Definition of Specification Restriction

见附录 C（`docs/l4_specification_task_definition.md`）。

已实现：

| spec | 定义 | 类别来源 |
|---|---|---|
| ALL | 保留全部候选 | 显式 ALL |
| cat:name | 保留 GT 匹配候选的类别为 name | BDD 11 类 canonical names；单类域 fallback person |
| inst:auto | 每视频 top-k 最长 GT 轨迹 | GT oracle |

全部标注 `PRIVILEGED_SPEC_ORACLE`（诊断），主推理不使用 GT membership。

## 7. Restriction Equivariance / Equivalent-Spec Consistency

- `T(R_s(X)) ≈ R_s(T(X))`：P0 vs P1 审计；
- 语义等价 spec（同一对象集合）的一致性：instance subset 对
  DanceTrack 已测，box/point/visual 未执行。

## 8. 2025–2026 Literature Audit

完整表见附录 A（`docs/l4_reference_audit.md`）。要点：

| 方法 | 是否等价 |
|---|---|
| SAM3/SAM3.1、GLEE、OVTR、OVTrack、Grounded SAM2 | 否：prompt 接口统一，无 restriction equivariance |
| STORM、QTrack、TempRMOT、TellTrack、EPIPTrack | 否：referring/query 内部一致性，不跨 spec |
| NOVA、GOVTrack、COVTrack/++ | 否：open-vocab，无 subset 等价 |
| V2-SAM、ViewSAM | 否：cross-view 一致性，非候选子集限制 |
| Path Consistency、UniTrack、TDLP、SambaMOTR | 否：时间/路径一致性，非 spec 限制 |

## 9. Novelty Collision Audit

见附录 B（`reports/l4_novelty_collision_audit.md`）：
`NO_DIRECTLY_EQUIVALENT_VERIFIED_METHOD_FOUND`。这是「未发现」，
不是「first」。

## 10. Why SAM3/GLEE Do Not Automatically Solve This Question

SAM3/GLEE 的统一在于「多 prompt 输入 + 检测/分割/跟踪」；它们不要求
同一视频换一种 specification 后身份 partition 不变，也不把
specification 定义为 set-restriction operator。L4 的 novelty 边界在
restriction-equivariant identity learning，而不是 prompt fusion。

## 11. Dataset / Spec Matrix

见附录 D（`reports/l4_spec_view_dataset.md`）与附录 E
（`docs/l4_tao_cache_recovery_plan.md`）。

| Domain | 视频/帧 | specs 已运行 |
|---|---:|---|
| BDD100K train | 200 / 8,001 | ALL + 11 categories |
| DanceTrack val | 25 / ~24k | ALL + person + inst:auto |
| TAO train | 105 / 4,200（2,256 有候选） | ALL + baby/car/dog/cat + inst:auto |
| DanceTrack calib / MOT17 / MOT20 | 训练 pairs | ALL + inst:auto |

## 12. BDD 11-Class Setup

manifest `outputs/l1_c/fixed_candidate_manifest/bdd100k_train.jsonl`
已含 11 类 GT canonical names（bicycle, bus, car, motorcycle,
other person, other vehicle, pedestrian, rider, trailer, train,
truck）；不捏造 supercategory。

## 13. TAO Recovery

缓存实际位于 `cache_dla/tao_amodal/train/<SOURCE>/<video>/<frame>/pilot.*`，
manifest key 少了 `train/<SOURCE>/` 一层。通过 `cache_key` 覆盖修复
（`tools/fix_tao_manifest.py`），不改共享缓存；105 视频全部可读。

## 14. Cross-Spec Consistency Metric Audit

见附录 F（`docs/l4_consistency_metric_audit.md`）。采用：

- 最优 ID 映射后的 pairwise co-identity agreement（主诊断）；
- per-GT drift；
- 每视图 windowed AssA/IDF1/IDSW（与官方 TrackEval 公式一致）；
- toy cases 验证（merge 0.5、split 0.667、全局置换 1.0）。

## 15. U0 Track-All Baseline / P0 / P1

见附录 G（`reports/l4_u0_restriction_audit.md`）。

## 16. Specification Inconsistency Audit

见附录 H（`reports/l4_cross_spec_inconsistency.md`）与附录 I
（`reports/l4_bdd_multispec.md`）、附录 J（`reports/l4_tao_openworld.md`）。

## 17. Does Candidate Restriction Change Identity?

**是**：

| Domain/Spec | drift |
|---|---:|
| BDD car | 32.9% |
| BDD pedestrian | 48.8% |
| BDD truck | 39.7% |
| BDD bus | 42.6% |
| DanceTrack person | 32.2% |
| DanceTrack inst:auto | 31.1% |
| TAO car_(automobile) | 24.1% |
| TAO inst:auto | 14.3% |

不是 evaluation bug：ALL vs ALL agree=1.0，toy-case 指标验证通过。

## 18. Domain-wise Restriction Sensitivity

- BDD（多类、低帧率 5fps）：category restriction 影响最大
  （33–67%）；
- DanceTrack（密集同类）：instance restriction 影响大（31%）；
- TAO（稀疏长尾）：影响较小但仍显著（14–24%）。

## 19. Category / Instance / Box / Point / Visual / Referring

- Category：已运行（11 类 + person）；
- Instance：已运行（top-2 GT oracle）；
- Box / Point / Visual exemplar：**NOT_EXECUTED**（无 verified 官方
  prompt encoder 的最小合法实现，不伪造）；
- Referring：**NOT_EXECUTED**（无官方 benchmark 数据接入；
  STORM/QTrack 仅记录）。

## 20. Architecture

`locatemot/models/l4_spec_eq.py`：

- 共享核心 = U0（EGRA：set-level transformer + bounded residual +
  reliability gate）；
- spec 条件 = type-level embedding（ALL/category/instance），只注入
  set-encoder token，不注入 pair head、不按 dataset 区分；
- 0.488M 参数；U0 checkpoint 初始化（`strict=False`，仅新增
  spec_embed/spec_proj）。

## 21. Multi-Spec Paired Training

`tools/build_l4_pairs.py` + `tools/train_l4.py`：

- 同一帧同一视频生成 full/restricted 两视图（L1DK base tracker
  分别推进）；
- 对齐：common candidates（索引子集）+ common tracks（birth-GT）；
- 15,851 pairs（BDD 7,645 / Dance 8,006 / MOT17 180 / MOT20 20）；
- 目标：local CE（每视图）+ assignment consistency + state
  consistency。

## 22. Assignment / State / Partition Consistency

- A5：row/col 对称 KL（common tracks/candidates）+ track-token cosine；
- A5p：partition-level co-assignment MSE（`S_full vs S_rest`，
  permutation-invariant）+ state cosine；
- 实现证据见附录 K（`docs/l4_implementation_evidence.md`）。

## 23. Training Objective

```
L = L_CE(full) + L_CE(rest) + w_a * L_assign + w_s * L_state
w_a = 1.0, w_s = 0.1（A2 时 w_a=w_s=0）
```

## 24. Pilot Results

见附录 L（`reports/l4_pilot.md`）。核心数字：

| 变体 | Dance inst drift | BDD car drift | TAO inst drift |
|---|---:|---:|---:|
| U0 | 0.3112 | 0.3291 | 0.1430 |
| A2 | 0.3272 | 0.3502 | 0.1865 |
| A5 | 0.3168 | 0.3262 | 0.1955 |
| A5p | 0.3314 | 0.3398 | 0.1677 |

## 25. Pilot Gate

| Gate | 结果 |
|---|---|
| A. Cross-spec consistency 显著下降 | **FAIL**（全部变体未下降） |
| B. Selected-object tracking ≥ naive pre-filter | 基本持平/略降 |
| C. ALL preservation | 官方 TrackEval 通过（macro AssA 0.4013）；audit 均值轻微下降 |
| D. Cross-domain | FAIL（TAO drift 上升） |

## 26. Full Multi-Domain Training

**NOT_EXECUTED**（pilot gate 未过，任务书规定早期失败直接收尾）。
见附录 M（`reports/l4_full_training.md`）。

## 27. One-Checkpoint Verification（ALL 模式，官方 TrackEval）

见附录 N（`reports/l4_ac_results.md`）。A2/A5/A5p 与 U0 完全一致：

| Domain | AssA | IDF1 | IDSW |
|---|---:|---:|---:|
| DanceTrack val | 0.4169 | 0.5694 | 2,588 |
| MOT17 | 0.6050 | 0.5825 | 259 |
| MOT20 | 0.2950 | 0.4012 | 2,406 |
| BDD | 0.2881 | 0.2923 | 11,042 |

## 28. Cross-Spec Identity Consistency Result

未被修复：A2/A5/A5p 的 drift 均 ≥ U0 水平（多数类别变差）。

## 29. Restriction Equivariance Result

`T(R_s(X)) ≈ R_s(T(X))` 在 U0 上不成立（drift 31–67%）；提出的
paired-consistency 训练未能建立等价性。机制层面
`L4_NOT_SUPPORTED`。

## 30. Track-All-Then-Filter Comparison

- P0 一致性最好（同一模型只跑一次）但受限对象 IDSW 高；
- P1 受限对象指标最好（Dance inst AssA 0.8406 / IDSW 72）但身份与
  ALL 不一致；
- L4 目标是合并两者优势，未达成。

## 31. Candidate/Token Efficiency

见附录 O（`reports/l4_efficiency.md`）。L4 训练 ~58 min（1 GPU），
推理审计 BDD ~160s / Dance ~9 min / TAO ~35s。

## 32. LODO

**NOT_EXECUTED**（pilot gate 未过）。见附录 P
（`reports/l4_lodo.md`）。

## 33. Leave-One-Spec-Type-Out

**NOT_EXECUTED**（pilot gate 未过）。见附录 Q
（`reports/l4_loso_spec.md`）。

## 34. Unseen Category / Open-Vocab

**NOT_EXECUTED**（无官方 open-vocab split；不伪造）。

## 35. Ablations

见附录 R（`reports/l4_ablation.md`）：

| tag | Dance inst drift | ALL mean AssA (Dance/BDD) |
|---|---:|---:|
| U0 | 0.3112 | 0.4514 / 0.3518 |
| A2 | 0.3272 | 0.4535 / 0.3277 |
| A5 | 0.3168 | 0.4422 / 0.3351 |
| A5p | 0.3314 | 0.4457 / 0.3364 |

A3/A4 未执行（A5 已失败，不细分）。

## 36. Where to Inject Specification

- Late selection（P0）：候选后过滤；
- Early conditioning（P1）：候选前过滤；
- Proposed：shared core + paired consistency（当前失败）；
结论：specification 的注入位置不是瓶颈，瓶颈是没有时间级
identity 约束。

## 37. Failure Cases

未做图像级可视化（无渲染管线）。数值级失败案例：

- BDD trailer：U0 drift 66.7%，A5 恶化至 73.3%；
- DanceTrack inst:auto：U0 drift 31.1%，三个变体全部 ≥31.7%；
- TAO dog：U0 drift 11.5%，A5p 恶化至 17.5%；
- 共性：crowd/crossing 时两个视图在不同帧发生 switch，单帧
  consistency 无法约束。

## 38. Why Not Regime Routing

U1 已证明 latent regime 是 dataset shortcut（96.6%），L4 不使用。

## 39. Why Not Universal ReID Embedding

L1-B 已失败；L4 不回到 embedding cosine 路线。

## 40. Why Not Prompt Interface Alone

SAM3/GLEE/OVTR 等已覆盖「支持多种 prompt」；这不是 novelty。

## 41. Why Not SAM3 / GLEE Alone

它们不解决 restriction-equivariant identity，且不是标准 box-MOT
association-controlled 协议下的 shared identity core。

## 42. Why Not Post-Filter Only

P0 是必须击败的 baseline；审计证明 P0 的 identity 与 P1 不一致，
且受限对象指标差，post-filter 不能称为解决方案。

## 43. Unified Claim Boundary

本阶段可以主张：

1. **诊断**：specification（候选子集限制）真实改变 persistent
   identity（三域、多 spec 类型）；
2. **审计**：未发现直接等价方法；
3. **负面结果**：逐帧 paired assignment/state/partition consistency
   不能修复该问题。

不可以主张：

- 已实现 restriction-equivariant tracking；
- ICLR-ready 方法创新；
- open-vocabulary / referring 支持。

## 44. ICLR Readiness Audit

| 条件 | 状态 |
|---|---|
| A. restriction 导致 identity inconsistency | ✅ PASS |
| B. proposed 显著减少 inconsistency | ❌ FAIL |
| C. 标准 MOT AssA/IDF1 不被牺牲 | ✅ PASS（官方 TrackEval 持平） |
| D. 一个 checkpoint 跨 ≥4 域 | ✅ PASS（ALL 模式） |
| E. BDD full multi-class | ✅ PASS（诊断） |
| F. TAO/open-world 证据 | ✅ PASS（诊断） |
| G. ≥3 类 spec 真实运行 | ✅ PASS（ALL/category/instance） |
| H. LODO | ❌ NOT_EXECUTED |
| I. unseen-spec-type | ❌ NOT_EXECUTED |
| J. novelty 边界清楚 | ✅ PASS |
| K. Track-All-Then-Filter 被解释/超越 | ⚠️ 被解释，未超越 |
| L. 真实 prompt/open-world benchmark | ❌ NOT_EXECUTED |

结论：**NOT_READY**。

## 45. Stage Decision

```text
L4_SPEC_RESTRICTION_SIGNAL_SUPPORTED
L4_PILOT_GATE_FAIL
L4_NOT_SUPPORTED（pilot mechanism）
ICLR readiness: NOT_READY
```

## 46. Next Single Recommendation

重设计为 **trajectory-level consistency**：配对数据从帧对改为 clip，
用可微 track-state 传播（或 Path-Consistency 式多路径一致）约束同一
对象的跨帧身份轨迹在两种 spec 视图下等价；在获得 pilot 正信号前，
不再训练任何逐帧 paired-consistency 变体。

## 47. Important Paths

- 最终报告：`reports/STAGE_L4_FINAL_REPORT.md`
- GPT handoff：`reports/STAGE_L4_GPT_HANDOFF.md`、
  `reports/LATEST_GPT_HANDOFF.md`
- 审计 JSON：`outputs/l4/audit_*.json`
- 模型：`outputs/l4/checkpoints/{a2,a5,a5p}/final.pt`
- 配对数据：`outputs/l4/data/*.pkl`
- TAO manifest：`outputs/l4/manifests/tao_amodal_train_l4.jsonl`
- 研究日志：`research_log.md`

## 48. Git Commit

`Stage L4 complete: specification-equivariant unified MOT`
（commit 见 git log，最终提交后记录）。

## 附录 A — 2025–2026 官方代码审计

# Stage L4 — 2025/2026 Specification/Identity-Consistency 官方实现审计

日期：2026-08-10（Asia/Shanghai）。
原则：只记录实际 clone + 阅读的官方代码，或明确标注「paper-only / 未公开
代码 / 未 clone」。commit 以 `git rev-parse HEAD` 为准；不根据摘要或
博客转述实现细节。

## 0. 审计问题

Stage L4 要回答的核心问题：

1. 是否存在已公开方法明确研究「same video + different specification →
   same persistent identity」；
2. 是否存在 MOT 方法要求 `Track(Restrict(X)) ≈ Restrict(Track(X))`
   （候选子集限制下的身份等价）；
3. 是否只是 SAM3/GLEE 式「多种 prompt 输入」的重述；
4. 是否只是 Track-All-Then-Filter；
5. 是否只是 Path Consistency 的 prompt 版本。

## 1. 2026 新检索与 clone

### 1.1 NOVA（IROS 2026，3D open-vocabulary MOT）

- 论文：NOVA: Next-step Open-Vocabulary Autoregression for 3D Multi-Object
  Tracking in Autonomous Driving（arXiv 2603.06254）。
- 官方仓库：`github.com/xifen523/NOVA`
- Commit：main `1bd3ff18`；**release/v0.1.0 `4358a627`（含完整代码）**
- License：Apache-2.0（LICENSE + NOTICE）。
- 已读文件：
  - `README.md`（release 分支：hybrid prompting、base/novel、
    `p_yes`、Hungarian、lifecycle）；
  - `nova/models/association_model.py`（geometry 注入 + causal LM +
    yes/no logprob → association cost）；
  - `nova/data/class_split.py`（canonical Base/Novel、alias、
    `AS_UNKNOWN` / `REJECT` 策略、authoritative class map 门）；
  - `nova/data/sample_builder.py`、`nova/preprocessing/common.py`
    （`build_semantic_prompt`、prompt policy）。
- 与 L4 关系：open-vocab class 用 base 名 / “Unknown” 的 hybrid prompt
  处理 Novel；但**没有**「同一视频、不同候选子集、身份应一致」的
  限制等价目标；3D 任务、无标准 2D box-MOT AC 协议。可借鉴
  class-split 的严谨性（禁止自造 taxonomy），不采用其 LM 关联核心。

### 1.2 V²-SAM（CVPR 2026，cross-view object correspondence）

- 论文：Marrying SAM2 with Multi-Prompt Experts for Cross-View Object
  Correspondence（arXiv 2511.20886）。
- 官方仓库：`github.com/jaychempan/V2-SAM`
- Commit：`31c3babf`；License：README 标注 MIT，**仓库根目录未见 LICENSE
  文件**（复制代码需作者确认）。
- 已读文件：
  - `projects/v2sam_visual/models/vp_matcher.py`（VPFeatureMatcher：
    prompt 视觉 embedding + mask 几何编码 + QKV cross-attention +
    spatial gate + FiLM 条件注入 + mask→mask decoder）；
  - `projects/v2sam_visual/models/v2sam.py`（RegionPooling、
    `get_contr_loss` 双向对比损失、`constr_prompt_fcs` 构造 SAM2 条件）；
  - `projects/v2sam_visual/models/sam2.py`（`inject_language_embd`）。
- 与 L4 关系：多 prompt expert + contrastive 表示对齐是「视觉 prompt
  一致性」参考；PCCS（post-hoc cyclic consistency selector）在 release
  代码中未检出明确实现。任务为 **cross-view correspondence / SOT**，
  不是同一视频上候选子集限制下的 MOT 身份等价。

### 1.3 DOVTrack（NeurIPS 2025，data-efficient OVMOT）

- 官方仓库：`github.com/zekunqian/DOVTrack`
- Commit：`5748236a`；内容：仅 README（Coming Soon），无代码/license。
- 与 L4 关系：open-vocab MOT 数据效率方向；未提供可审计实现。

### 1.4 TempRMOT（arXiv 2406.05039，referring MOT）

- 官方仓库：`github.com/zyn213/TempRMOT`
- Commit：`6a65640d`；License：**仓库无 LICENSE 文件**。
- 已读文件：`models/transrmot_pro.py`（ClipMatcher、`loss_refers`
  focal/CE）、`models/qim_pro.py`（QueryInteractionModule 轨迹 query
  更新）、`models/memory_bank.py`。
- 与 L4 关系：referring query 驱动 MOT，身份一致性只针对同一 query
  集合；不研究「不同 query/spec 下同一真实对象身份是否保持」。

### 1.5 TellTrack（“Tell Me What to Track”，2024/2026）

- 官方仓库：`github.com/Durablion/telltrack`
- Commit：HEAD（shallow clone）；内容：只有 index.html / privacy policy，
  **无模型代码**。
- 与 L4 关系：referring MOT 论文；无官方实现可审计。

### 1.6 未找到官方实现的 2026 方法（paper-only，不当作依据）

- **EPIPTrack**（arXiv 2510.13235）：explicit + implicit prompts 的
  multimodal MOT；GitHub API 搜索 0 结果，arXiv 页无 code 链接。
- **GOVTrack**（CVPR 2026）：generative OVMOT；未找到官方 repo。
- **ViewSAM**（arXiv 2605.02638）：weakly supervised cross-view
  referring MOT；未找到官方 repo。
- **Robust Exemplar Prompt Learning for MOT**（ACM MM 2026）：未找到
  official repo。
- **COVTrack / COVTrack++**（ICCV 2025 / 2026）：未发布代码。

## 2. 既有 L3/L2/L1 已 clone 审计复核（L4 视角）

以下仓库已在 `docs/l3_reference_audit.md`、`docs/l2_reference_audit.md`、
`docs/l1_d_reference_audit.md` 逐文件阅读，此处只做 L4 相关结论复核。

| Method | Year/Venue | Official repo | Commit | License | 多域 | 多 spec 输入 | Persistent ID | 跨 spec ID 一致性 | Restriction equivariance | 与 L4 关系 |
|---|---|---|---|---|---|---|---|---|---|---|
| SAM3/3.1 | 2026 Meta | sam3 | 96914d24 | SAM License | promptable 视频 | text/box/point/mask/exemplar | 是 | 否 | 否 | prompt 接口碰撞，非核心问题 |
| GLEE | CVPR 2024 | GLEE | f36a49e8 | MIT | 是（BDD/TAO） | text/query | 是 | 否 | 否 | 多域统一部分碰撞 |
| OVTR | ICLR 2025 | OVTR | 500e72c1 | MIT | 否 | text category | 是 | 否 | 否 | open-vocab prompt 历史 |
| OVTrack | CVPR 2023 | ovtrack | e188b32e | Apache-2.0 | 否 | text category | 是 | 否 | 否 | open-vocab prompt 历史 |
| Grounded SAM2 | 2024/25 | grounded-sam-2 | b7a9c29f | Apache-2.0 | 否（pipeline） | text/box/point | 部分 | 否 | 否 | pipeline 无学习型一致性 |
| SAM2MOT | AAAI 2026 | SAM2MOT | 7bdae12c | Apache-2.0 | 否 | 否 | 是 | 否 | 否 | 代码未发布 |
| STORM | CVPR 2026 | STORM-Bench | 0d87c3ba | — | 否 | referring | 是 | 否 | 否 | referring 基准可用 |
| QTrack/RMOT26 | 2026 | QTrack | bc746fe2 | MIT | 否 | query | 是 | 否 | 否 | query-driven MOT |
| AnyTrack | 2026 | AnyTrack | 7d5ca454 | — | SOT | 模态 prompt | SOT | 否 | 否 | 模态统一，非 subset 等价 |
| TDLP | 2025/26 | TDLP | 50344b92 | MIT | 否 | 无 | 是 | 否 | 否 | 下一帧 link prediction |
| Path Consistency | CVPR 2024 | path-consistency | f4b7d26d | Apache-2.0 | 否 | 无 | 是 | 多路径 | 路径级非 spec 级 | 一致性数学框架参考 |
| UniTrack | ICLR 2026 | UniTrack | afdd9869 | README MIT（无 LICENSE 文件） | 是 | 无 | 是 | 轨迹平滑 | 否 | 轨迹正则参考 |
| MOTIP/MOTIP-2 | CVPR 2025 | MOTIP / MOTIP-2 | ffc0e905 / 012856c1 | Apache-2.0 | 否 | 无 | ID 预测 | 否 | 否 | ID decoder 历史 |
| CAMELTrack | 2025 | CAMELTrack | 46a74bb | 见仓库 | 否 | 无 | 是 | 否 | 否 | set-level 竞争 + InfoNCE |
| FDTA | CVPR 2026 | FDTA | b3b3b778 | 有 | 否 | 无 | 是 | 否 | 否 | 判别 embedding |
| TRACT | ICCV 2025 | TRACT | 19f01d72 | 无 | 否 | text | 是 | 轨迹内 | 否 | 轨迹一致性表示 |
| LG-Track / LLTrack | 2023/25 | LG-Track / LLTrack | 432a467 / 2ab7994 | MIT/有 | 否 | 类别互斥 | 是 | 否 | 否 | 语义互斥 cue |
| MeMOTR | ECCV 2022 | MeMOTR | HEAD | 有 | 否 | 无 | 是 | 否 | 否 | memory 参考 |
| NOVA | IROS 2026 | NOVA | 4358a627 | Apache-2.0 | 3D | hybrid base/novel | 是 | 否 | 否 | open-vocab class 严谨 split |
| V²-SAM | CVPR 2026 | V2-SAM | 31c3babf | README MIT（无 LICENSE） | 否 | visual prompt | SOT | cross-view 表示 | 否 | 表示对齐/对比损失参考 |
| DOVTrack | NeurIPS 2025 | DOVTrack | 5748236a | 无 | 否 | text | 是 | 否 | 否 | 代码未发布 |
| TempRMOT | 2024/26 | TempRMOT | 6a65640d | 无 | 否 | referring | 是 | 否 | 否 | RMOT query 更新 |
| TellTrack | 2024/26 | telltrack | HEAD | 无 | 否 | referring | 是 | 否 | 否 | 无代码 |
| EPIPTrack | 2025 | — | — | — | 否 | explicit/implicit prompt | 是 | 否 | 否 | paper-only |
| GOVTrack | CVPR 2026 | — | — | — | 否 | text | 是 | 否 | 否 | paper-only |
| ViewSAM | 2026 | — | — | — | 否 | referring | 跨视角 | 跨视角 | 否 | paper-only |
| COVTrack/++ | 2025/26 | — | — | — | 否 | text | 是 | 否 | 否 | paper-only |

## 3. 非 MOT 的结构参考（只作理论，不冒充 MOT baseline）

- Multiset/Set-equivariant set prediction（ICLR 2022）：集合预测的
  permutation 结构先例；不涉及 persistent identity。
- V²-SAM PCCS 思路（论文，代码未检出）：multi-expert cyclic
  consistency 选择；跨视角 correspondence，不是候选子集限制。
- Path Consistency：多观测路径的关联一致性监督，是 L4 assignment
  consistency loss 的数学近亲；但路径来自时间子采样，不是 spec 候选子集。

## 4. 审计结论

1. **未发现**任何已公开 MOT 方法明确要求
   `Track(Restrict_s(X)) ≈ Restrict_s(Track(X))` 或研究
   「same video + different specification → same persistent identity」；
2. 已公开的 prompt/query 类 MOT（SAM3、GLEE、QTrack、STORM、
   EPIPTrack、TempRMOT、TellTrack）都把 spec 作为输入接口，身份一致性
   只在**同一 spec 内部**维持；
3. Path Consistency 有「一致性」之名，但路径来自观测子采样，不是
   specification 诱导的候选集限制；
4. 因此 L4 的候选子集限制等价性在本次审计范围内无直接等价方法。

## 附录 B — Novelty Collision Audit

# Stage L4 — Novelty Collision Audit

日期：2026-08-10。

## 1. 核心主张

> MOT identity should be stable to how the target object set is specified.
> We formulate specification as a set-restriction operator and enforce
> restriction-equivariant identity tracking (one shared tracker across
> heterogeneous MOT domains and specification interfaces).

等价形式：

```
T_theta(R_s(X)) ≈ R_s(T_theta(X))   （common objects 上，permutation-invariant）
s_a ~ s_b 语义等价 → T(R_sa(X)) ≈ T(R_sb(X))
```

## 2. 逐项回答

### 2.1 是否已有「same video + different specification → same persistent identity」

否。已核实 SAM3/GLEE/OVTR/OVTrack/Grounded-SAM2/SAM2MOT/STORM/QTrack/
EPIPTrack/TempRMOT/TellTrack/NOVA/COVTrack/GOVTrack/ViewSAM 等：
所有方法在同一查询/spec 内维护身份，不评估、不优化「同一个视频换一种
specification 后身份是否保持」。V²-SAM/ViewSAM 的一致性跨**视角**，
不是候选子集限制。

### 2.2 是否已有 MOT 方法要求 Track(Restrict(X)) ≈ Restrict(Track(X))

否。Path Consistency 是最接近的「一致性」方法，但其路径来自观测子集
（时间采样），不是 specification 诱导的候选对象限制；且没有
Track-All-Then-Filter vs Pre-Filter 的对偶审计。

### 2.3 是否只是 SAM3/GLEE 的重述

不是。SAM3/GLEE 的贡献是「多种 prompt 输入接口 + 统一检测/分割/跟踪」；
它们不要求不同候选子集下身份等价，也不把 specification 定义为
set-restriction operator。L4 不新增 prompt 编码器，核心是
restriction-equivariant identity learning。

### 2.4 是否只是 post-filter

不是。post-filter（P0）是必须击败/解释的强 baseline；Stage L4-A 的
paired audit 正是为了证明 P0 与 P1 在 common objects 上身份不一致，
从而说明「单纯 post-filter 不保证身份稳定」。

### 2.5 是否只是 Path Consistency 的 prompt 版本

不是。Path Consistency 在无 GT 身份监督下约束时间子采样路径的关联一致；
L4 使用 privileged GT 配对视图（诊断 + 训练）、TrackEval 主指标和
candidate-subset restriction，不是路径采样的直接推广。Path Consistency
仅作为 assignment-consistency loss 的数学参考。

## 3. 结论

**NO_DIRECTLY_EQUIVALENT_VERIFIED_METHOD_FOUND**

注意：

1. 这是「在 2026-08-10 可核实范围内未发现」，不是「first」；
2. 检索盲区包括付费期刊、未公开代码、非英语来源；最终稿前需再次检索；
3. 若发现等价方法，必须修改 claim 并明确差异。

## 附录 C — Specification / Restriction 任务定义

# Stage L4 — Specification / Restriction 任务定义

日期：2026-08-10。

## 1. Specification 是 set-restriction operator

给定视频的候选流

```
X = {x_{t,i}}   （候选 box + 特征流）
```

specification `s` 定义候选保留集合：

```
R_s(X) = {x_{t,i} ∈ X : x_{t,i} 满足 s}
```

本阶段已实现（PRIVILEGED_SPEC_ORACLE，训练/诊断用途）：

| spec | 定义 | 类别来源 |
|---|---|---|
| `ALL` | 保留全部候选 | 显式 ALL |
| `cat:<name>` | 保留 GT 匹配候选的类别为 name | BDD manifest 真实 11 类 canonical names；DanceTrack/MOT 单类 fallback `person` |
| `inst:<gid,...>` / `inst:auto` | 保留指定 GT 身份（auto=每视频最长 top-k 轨迹） | GT 身份（诊断 oracle） |

未实现（本阶段不伪造）：

- box / point / visual exemplar / referring：需要官方 prompt encoder 与
  真实 benchmark 数据；Stage L4-A 没有合法可复用实现，标注
  `NOT_EXECUTED`，不把 synthetic prompt 当作主结果。

## 2. 等价性定义

`≈` 只在 common objects 上比较，使用 permutation-invariant
co-identity agreement（见 `docs/l4_consistency_metric_audit.md`）：

```
T_theta(R_s(X)) ≈ R_s(T_theta(X))
```

语义等价的 `s_a ~ s_b`（指向同一对象集合）还要求：

```
T(R_sa(X)) ≈ T(R_sb(X))
```

## 3. Dataset / Spec 矩阵

| Domain | 候选来源 | 已运行 specs | 说明 |
|---|---|---|---|
| BDD100K train（200 视频，8001 帧，11 类 GT） | LocateAnything cache | ALL + 11 category | 主证据 |
| DanceTrack val（25 视频） | LocateAnything cache | ALL + person + inst:auto | 第二证据（instance） |
| DanceTrack calibration（8 视频） | 同 cache | ALL + inst:auto（训练 pairs） | 训练 |
| MOT17 / MOT20 train | 同 cache | ALL + inst:auto（训练 pairs） | 训练/评估 |

BDD category 用官方 11 类 canonical names（bicycle, bus, car, motorcycle,
other person, other vehicle, pedestrian, rider, trailer, train, truck）；
不捏造 supercategory 层级。

## 4. 禁止事项

- 主推理结果不允许用 GT membership 过滤候选（本阶段所有 category/
  instance 结果均标注 PRIVILEGED_SPEC_ORACLE）；
- 不声称 open-vocabulary / referring（无官方 split / benchmark）；
- 不把 P0（post-filter）包装成创新。

## 附录 D — Specification Paired-View Dataset

# Stage L4 — Specification Paired-View Dataset

日期：2026-08-10。

## 1. 构造

`tools/build_l4_pairs.py` 对每个视频帧同时推进两个 base tracker
（L1DK：0.4 IoU + 0.2 PBD + 0.4 Kalman，thr 0.25）：

- full view：全部候选（spec=ALL）；
- restricted view：spec 保留的候选（category / instance）。

每对样本包含：

- full/rest 各自的 EGRA 特征（pair/track/cand/base）；
- 各自的 GT row/col label 与 base_correct；
- `common_cand`：(full_idx, rest_idx) 候选对齐（受限候选天然是
  full 候选的子集）；
- `common_track`：(full_track_idx, rest_track_idx) 轨迹对齐
  （按 birth 时 privileged GT 身份匹配）。

spec 类型编码（共享、非 dataset-specific）：

| idx | 类型 |
|---|---:|
| 0 | ALL |
| 1 | category |
| 2 | instance |

## 2. 规模

| Domain | Pairs | Spec 构成 |
|---|---:|---|
| BDD100K train | 7,645 | car 4,755 / pedestrian 1,274 / truck 953 / bus 340 / other vehicle 164 / bicycle 89 / motorcycle 21 / trailer 18 / rider 16 / other person 15 |
| DanceTrack calibration | 8,006 | inst:auto（top-2 GT） |
| MOT17 train | 180 | inst:auto |
| MOT20 train | 20 | inst:auto |
| **合计** | **15,851** | |

## 3. 用途

- 训练 A2（spec-conditioned、无一致性）与 A5（+ assignment/state
  consistency）；
- 可扩展用于 A3/A4 消融；
- 全部为 PRIVILEGED_SPEC_ORACLE（GT 身份只用于训练对齐，推理不使用
  GT membership）。

## 附录 E — TAO Cache Recovery Plan

# Stage L4 — TAO Cache Recovery Plan

日期：2026-08-10。

## 1. 现状

- Manifest：`outputs/l1_c/fixed_candidate_manifest/tao_amodal_train.jsonl`
  （105 videos / 4200 frames；2256 帧有候选，1944 帧无候选）。
- 缓存：`/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1B/
  cache_dla/tao_amodal/train/<SOURCE>/<video_id>/<frame>/pilot.*`
  （source ∈ {BDD, AVA, YFCC100M, HACS, LaSOT}）。
- 4200 个 `.complete` 全部存在；safetensors/meta 同目录。
- 根因：manifest 的读取 key 是
  `tao_amodal/<video_id>/<frame>/pilot`，而真实路径多了一层
  `train/<SOURCE>/`。缓存本身没有损坏。

## 2. 恢复方案（已执行）

**不修改共享缓存、不重跑 LocateAnything、不复制数据。**

1. 新增 `tools/fix_tao_manifest.py`：对每个 video_id 在
   `cache_dla/tao_amodal/train/<SOURCE>/` 下定位一次来源，写入
   `outputs/l4/manifests/tao_amodal_train_l4.jsonl`，每行增加
   `cache_key = tao_amodal/train/<SOURCE>/<video_id>/<frame>/pilot`；
2. `build_candidates`（`tools/l4_restriction_audit.py`、
   `tools/eval_l3.py`）优先使用 `entry["cache_key"]`；
3. 验证：U0 audit 在 TAO ALL 上成功运行 105 视频，
   pairs=7,522（公共候选观测），ALL vs ALL agree=1.0。

## 3. 结果（frozen U0，PRIVILEGED_SPEC_ORACLE）

| Spec | Pairs | drift | P0 AssA | P1 AssA | P0 IDSW | P1 IDSW |
|---|---:|---:|---:|---:|---:|---:|
| ALL | 7,522 | 0.0000 | 0.4174 | 0.4174 | 870 | 870 |
| baby | 578 | 0.0675 | 0.2501 | 0.2625 | 83 | 69 |
| car_(automobile) | 798 | 0.2406 | 0.0963 | 0.1315 | 164 | 70 |
| dog | 217 | 0.1152 | 0.0225 | 0.0285 | 32 | 25 |
| cat | 168 | 0.0119 | 0.0431 | 0.0436 | 5 | 2 |
| inst:auto | 2,230 | 0.1430 | 0.4044 | 0.4616 | 359 | 217 |

结论：TAO（open-world long-tail）同样表现出 restriction 敏感性
（car 24%、instance 14%）与 P1 改善；作为第三域证据加入
Problem Signal。

## 4. 遗留

- 无候选帧（1944/4200）在 AC 协议下只能作为空帧；open-world
  detection 不是本项目 AC 范围；
- 若后续做 TETA/TAO 官方协议，需要额外 detection 输出与
  TrackEval TETA 适配（不在 Stage L4 主 AC 范围内）。

## 附录 F — Cross-Spec Consistency Metric Audit

# Stage L4 — Cross-Spec Consistency Metric Audit

日期：2026-08-10。

## 1. 要求

比较两个 spec 视图（例如 ALL vs category）的身份一致性时：

- **permutation invariant**：不比较 raw track integer ID；全局一致重标号
  必须视为一致；
- 只比较 **common detections / common identities**（两视图都出现的
  候选与轨迹）；
- 对 **merge / split / switch** 敏感；
- 有 toy-case 验证；
- 与 GT-based AssA/IDF1 做 sanity check（TrackEval 仍是主指标）。

## 2. 已检索的候选指标

| 指标 | 优点 | 缺点 / 为何不作为主指标 |
|---|---|---|
| 直接比较 track ID | 简单 | 违反 permutation-invariant，全局重标号即误报 |
| HOTA-style AssA（per view） | 官方主指标 | 度量每个视图自身的关联质量，不直接度量两视图一致 |
| Pairwise co-identity agreement（最优 ID 映射后） | permutation-invariant，对 merge/split/switch 敏感 | 需要一个对齐步骤（匈牙利），对齐本身是诊断的一部分 |
| Partition F1 / Adjusted Rand Index | 成熟的聚类一致性 | 对候选集合差异敏感，需要额外规定 common set；可作诊断 |
| Trajectory partition consistency | 直观 | 与 pairwise co-identity 等价但实现更绕 |

结论：**采用「最优 ID 映射后的 pairwise co-identity agreement」作为
主诊断指标**，并同时报告：

- per-GT identity drift（每个 GT 身份在两视图间不一致的比例）；
- 每个视图自己的 TrackEval-consistent windowed AssA/IDF1/IDSW。

## 3. 实现

`tools/l4_restriction_audit.py`：

1. P0（Track-All-Then-Filter）：对全候选流跑 frozen U0，再按 spec 过滤
   轨迹；
2. P1（Pre-Filter）：只对 spec 候选流跑同一个 U0；
3. 在公共候选帧上收集 `(frame, tid_P0, tid_P1, gid, cat)`；
4. 用 Hungarian 在 `(tid_P0, tid_P1)` 共现计数矩阵上求最优 ID 映射；
5. `agree_rate` = 映射后一致的比例；`drift_rate = 1 - agree_rate`；
6. `per_gt_drift`：按 GT id 聚合的一致率；
7. 每视图用 `windowed_metrics`（AssA/IDF1/IDSW，公式与官方 TrackEval
   一致）计算。

## 4. Toy-case 验证（2026-08-10 实际运行）

| Case | agree_rate | 判断 |
|---|---:|---|
| 全局一致重标号（A→B 每帧同一映射） | 1.0000 | 正确：partition 未变 |
| 两个身份在 B 中合并 | 0.5000 | 正确：merge 被捕获 |
| 一个身份在 B 中分裂 | 0.6667 | 正确：split 被捕获 |
| 5 个身份全局一致置换 | 1.0000 | 正确：permutation-invariant |
| 全视频一致 switch（1↔2 互换） | 1.0000 | 正确：partition 等价 |

真实数据中的 drift（BDD 33–67%、DanceTrack 31–32%）不是全局置换，
而是随时间不一致的 merge/split/switch，因此被该指标捕获。

## 5. 与 TrackEval 的 sanity

同一代码路径的 `windowed_metrics` 已在 Stage L2 与官方 TrackEval
整视频数值对齐（`docs/l2_trackeval_objective_audit.md`）。本审计中
`ALL vs ALL` 的自检为 `agree_rate = 1.0` 且 P0/P1 指标完全相同，
排除实现层面的系统性偏差。

## 6. 不做的事

- 不比较 raw track integer ID；
- 不把「自定义 consistency」当主结果替代 TrackEval；
- 不使用 partition F1/ARI 作为主指标（两视图候选集合大小不同时
  需要额外规定 common set；可留作论文诊断，本阶段未实现）。

## 附录 G — U0 Restriction Audit (P0 vs P1)

# Stage L4-A — U0 Restriction Audit：P0 (Track-All-Then-Filter) vs P1 (Pre-Filter)

日期：2026-08-10。
协议：frozen U0（L1DAssociator，`outputs/l3/checkpoints/u0/final.pt`），
OnlineTracker L1DK shell（weights 0.4/0.2/0.4，thr 0.25，delta 0.3），
per-video fresh tracker，`output_all_candidates=True`。

## 1. BDD100K（200 视频 / 8001 帧 / 11 类 GT，PRIVILEGED_SPEC_ORACLE）

windowed AssA/IDF1/IDSW 按视频均值聚合（IDSW 为总和）；ALL 自检
agree=1.0 且 P0=P1，验证管线。

| Spec | Pairs | agree | drift | P0 AssA | P1 AssA | P0 IDSW | P1 IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL | 56,013 | 1.0000 | 0.0000 | 0.3518 | 0.3518 | 16,603 | 16,603 |
| car | 32,059 | 0.6709 | 0.3291 | 0.3950 | 0.3801 | 9,511 | 8,527 |
| bus | 634 | 0.5741 | 0.4259 | 0.2037 | 0.2656 | 231 | 42 |
| truck | 2,044 | 0.6032 | 0.3968 | 0.3438 | 0.4404 | 688 | 236 |
| pedestrian | 6,097 | 0.5122 | 0.4878 | 0.3151 | 0.3462 | 2,486 | 2,053 |
| rider | 59 | 0.5424 | 0.4576 | 0.0873 | 0.0927 | 20 | 3 |
| motorcycle | 39 | 0.5128 | 0.4872 | 0.0356 | 0.0400 | 15 | 3 |
| bicycle | 194 | 0.5155 | 0.4845 | 0.1070 | 0.1200 | 76 | 31 |
| train | 1 | 1.0000 | 0.0000 | 0.0050 | 0.0050 | 0 | 0 |
| trailer | 30 | 0.3333 | 0.6667 | 0.0022 | 0.0052 | 12 | 3 |
| other vehicle | 312 | 0.5256 | 0.4744 | 0.1057 | 0.1318 | 136 | 31 |
| other person | 62 | 0.5968 | 0.4032 | 0.0426 | 0.0446 | 25 | 20 |

除 car 外，P1 的 AssA 均 ≥ P0；所有样本充足的类别 P1 的 IDSW 都显著
更低（bus −82%、truck −66%、rider −85%、motorcycle −80%、bicycle −59%、
other vehicle −77%、trailer −75%）。

## 2. DanceTrack val（25 视频）

| Spec | Pairs | agree | drift | P0 AssA | P1 AssA | P0 IDSW | P1 IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL | 218,580 | 1.0000 | 0.0000 | 0.4514 | 0.4514 | 5,858 | 5,858 |
| person | 213,113 | 0.6775 | 0.3225 | 0.4505 | 0.4597 | 4,464 | 3,947 |
| inst:auto (top-2 GT) | 48,586 | 0.6888 | 0.3112 | 0.5592 | 0.8406 | 799 | 72 |

## 3. 结论

1. **候选集限制真实改变 persistent identity**：BDD category 的
   drift 33–67%，DanceTrack instance 31%；
2. 限制方向通常**改善**受限视图的关联质量（IDSW 大幅下降，
   多数 AssA 上升）——说明 distractor competition / set context
   是身份漂移的来源；
3. `L4_SPEC_RESTRICTION_SIGNAL_SUPPORTED`（BDD category 主证据 +
   DanceTrack instance 第二证据，跨两个 domain/spec 类型）。

## 附录 H — Cross-Spec Identity Inconsistency

# Stage L4 — Cross-Spec Identity Inconsistency

日期：2026-08-10。

## 1. 问题

同一个 frozen U0，对同一视频跑 ALL 与受限候选流，公共对象上的
track identity 会因候选子集变化而改变：

- BDD：car 33%、truck 40%、bus 43%、pedestrian 49%、
  trailer 67% 的 common-object 观测在最优 ID 映射后仍不一致；
- DanceTrack：person 32%、top-2 instance 31% 不一致。

## 2. 为什么身份会漂移（机制判断）

1. **Hungarian 竞争**：全候选流中不同类别的 distractor 占用轨迹槽，
   受限流去掉了竞争后匹配结果不同；
2. **set context**：EGRA 的 pair/track/cand 特征包含
   `log_n_cand / margins / top1` 等集合级特征，候选集合变化直接改变
   模型输入；
3. **track-state 更新**：两视图的轨迹生命周期（birth/lost/terminate）
   不同步，公共对象在其中一个视图可能被错误匹配后污染后续状态；
4. **P0 过滤后产生缺失帧**：Track-All-Then-Filter 输出在受限对象上
   有 gap，TrackEval 的 IDSW 会把这些 gap 记为 switch。

## 3. P0 vs P1 方向

- P1 通常改善受限视图的 AssA/IDSW（尤其 instance subset）；
- 但 P1 的 identity 与 ALL 不一致，意味着「用户换一种询问方式」会得到
  一套不同的轨迹编号语义；
- 因此需要的不是选 P0 或 P1，而是让 shared identity process 对候选
  子集稳定（restriction-equivariant）。

## 4. 训练必要性

该不一致不是 evaluation bug（ALL vs ALL 自检一致、toy-case 指标验证
通过），因此进入 Stage L4-B：paired-view spec-equivariant training。

## 附录 I — BDD100K Multi-Spec Results

# Stage L4 — BDD100K Multi-Spec Results

日期：2026-08-10。200 视频 / 8001 帧 / 11 类 GT；
PRIVILEGED_SPEC_ORACLE；windowed metrics 按视频均值，IDSW 为总和。

## 1. U0（frozen）

| Spec | drift | P0 AssA | P1 AssA | P0 IDSW | P1 IDSW |
|---|---:|---:|---:|---:|---:|
| ALL | 0.0000 | 0.3518 | 0.3518 | 16,603 | 16,603 |
| car | 0.3291 | 0.3950 | 0.3801 | 9,511 | 8,527 |
| bus | 0.4259 | 0.2037 | 0.2656 | 231 | 42 |
| truck | 0.3968 | 0.3438 | 0.4404 | 688 | 236 |
| pedestrian | 0.4878 | 0.3151 | 0.3462 | 2,486 | 2,053 |
| rider | 0.4576 | 0.0873 | 0.0927 | 20 | 3 |
| motorcycle | 0.4872 | 0.0356 | 0.0400 | 15 | 3 |
| bicycle | 0.4845 | 0.1070 | 0.1200 | 76 | 31 |
| train | 0.0000 | 0.0050 | 0.0050 | 0 | 0 |
| trailer | 0.6667 | 0.0022 | 0.0052 | 12 | 3 |
| other vehicle | 0.4744 | 0.1057 | 0.1318 | 136 | 31 |
| other person | 0.4032 | 0.0426 | 0.0446 | 25 | 20 |

## 2. A2（spec-conditioned，无一致性）

| Spec | drift | P0 AssA | P1 AssA | P0 IDSW | P1 IDSW |
|---|---:|---:|---:|---:|---:|
| ALL | 0.0000 | 0.3277 | 0.3277 | 16,798 | 16,798 |
| car | 0.3502 | 0.3746 | 0.3629 | 9,698 | 8,696 |
| bus | 0.4590 | 0.2042 | 0.2628 | 237 | 33 |
| truck | 0.4178 | 0.3334 | 0.4331 | 692 | 206 |
| pedestrian | 0.5048 | 0.3099 | 0.3293 | 2,494 | 2,039 |
| rider | 0.4237 | 0.0869 | 0.0901 | 22 | 5 |
| motorcycle | 0.4359 | 0.0367 | 0.0408 | 13 | 1 |
| bicycle | 0.5103 | 0.1049 | 0.1121 | 74 | 31 |
| train | 0.0000 | 0.0050 | 0.0050 | 0 | 0 |
| trailer | 0.7333 | 0.0021 | 0.0049 | 15 | 2 |
| other vehicle | 0.4744 | 0.1049 | 0.1287 | 130 | 27 |
| other person | 0.4355 | 0.0416 | 0.0443 | 26 | 19 |

## 3. A5（+ row/col KL + state consistency）

| Spec | drift | P0 AssA | P1 AssA | P0 IDSW | P1 IDSW |
|---|---:|---:|---:|---:|---:|
| ALL | 0.0000 | 0.3351 | 0.3351 | 16,660 | 16,660 |
| car | 0.3262 | 0.3808 | 0.3777 | 9,573 | 8,543 |
| bus | 0.4495 | 0.2054 | 0.2636 | 235 | 32 |
| truck | 0.4022 | 0.3298 | 0.4446 | 684 | 195 |
| pedestrian | 0.4793 | 0.3134 | 0.3429 | 2,467 | 2,049 |
| rider | 0.4237 | 0.0863 | 0.0911 | 21 | 4 |
| motorcycle | 0.4615 | 0.0365 | 0.0409 | 15 | 2 |
| bicycle | 0.4948 | 0.1080 | 0.1172 | 75 | 28 |
| train | 0.0000 | 0.0050 | 0.0050 | 0 | 0 |
| trailer | 0.7333 | 0.0021 | 0.0055 | 14 | 1 |
| other vehicle | 0.4391 | 0.1063 | 0.1316 | 129 | 22 |
| other person | 0.3387 | 0.0430 | 0.0480 | 25 | 19 |

## 4. 结论

1. A5 的 consistency 收益不系统：5/11 类别 drift 小幅下降，
   6/11 持平或变差；
2. A5 的 P1 IDSW 相对 U0 无一致改善（bus −10、truck −41、pedestrian
   −4，但 car +16、other vehicle −9、rider +1）；
3. ALL 模式 AssA 下降 1.67pp（U0 0.3518 → A5 0.3351），超过
   0.3–0.5pp 警报线。

## 附录 J — TAO / Open-World Results

# Stage L4 — TAO / Open-World Results

日期：2026-08-10。105 视频 / 4200 帧（2256 帧有候选）；
cache 已通过 `cache_key` 覆盖恢复（`docs/l4_tao_cache_recovery_plan.md`）。
PRIVILEGED_SPEC_ORACLE；windowed metrics 按视频均值，IDSW 为总和。

| Spec | U0 drift | A2 drift | A5 drift | U0 P1 AssA | A5 P1 AssA | U0 P1 IDSW | A5 P1 IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL | 0.0000 | 0.0000 | 0.0000 | 0.4174 | 0.4376 | 870 | 860 |
| baby | 0.0675 | 0.1142 | 0.1003 | 0.2625 | 0.2768 | 69 | 53 |
| car_(automobile) | 0.2406 | 0.2293 | 0.2481 | 0.1315 | 0.1318 | 70 | 62 |
| dog | 0.1152 | 0.2166 | 0.1198 | 0.0285 | 0.0299 | 25 | 26 |
| cat | 0.0119 | 0.0119 | 0.0119 | 0.0436 | 0.0436 | 2 | 2 |
| inst:auto | 0.1430 | 0.1865 | 0.1955 | 0.4616 | 0.4532 | 217 | 266 |

结论：

1. TAO 同样证明 restriction 会改变身份（U0 car 24%、inst 14%）；
2. A5 一致性训练未降低 drift（inst 14.3% → 19.6%），P1 指标混合；
3. A5 的 ALL 模式 AssA 略升（0.4174 → 0.4376），但不足以抵消
   BDD/DanceTrack 的退化与 drift 恶化。

## 附录 K — Implementation Evidence

# Stage L4 — Implementation Evidence

日期：2026-08-10。每个核心模块的官方参考、实际阅读文件、采用/不采用
理由，按证据要求记录。

## Module: Restriction Audit（P0 vs P1）

- Scientific purpose：证明 specification/candidate-set restriction
  真实改变 persistent identity，且不是 evaluation bug。
- Official references inspected：TrackEval formulas（
  `references/TrackEval-official`，commit 12c8791b）、MOTIP ID prediction
  （`references/identity_decoding/MOTIP`，ffc0e905）、Path Consistency
  （`references/association_2025_2026/PathConsistency`，f4b7d26d）。
- Repository commits：见上。
- Files inspected：`tools/run_l2_oracle.py`（windowed_metrics）、
  `tools/build_l1d_dataset.py`（base simulator）、`tools/eval_l3.py`、
  `locatemot/tracking/online_tracker.py`。
- Observed implementation：U0 在同一 AC shell 上对全候选/受限候选分别
  推理；最优 ID 映射（Hungarian on co-occurrence）后比较 common
  objects 的 co-identity agreement。
- Parts adopted：windowed AssA/IDF1/IDSW；per-video fresh tracker；
  threshold 0.25 / delta 0.3。
- Parts intentionally not adopted：不比较 raw track ID；不用 GT
  membership 过滤主推理结果（全部标 PRIVILEGED_SPEC_ORACLE）。
- Reason for final design：需要 permutation-invariant 且对
  merge/split/switch 敏感的诊断指标；TrackEval 仍是主结果。

## Module: Paired Spec Views

- Scientific purpose：构造 `T(R_s(X))` 与 `R_s(T(X))` 的可训练配对。
- Official references inspected：TDLP clip 构造
  （`references/association_2025_2026/TDLP`，50344b92）、Path
  Consistency 路径构造（f4b7d26d）、CAMELTrack 轨迹状态采样
  （`references/l1_d/CAMELTrack`，46a74bb）。
- Observed implementation：同一 L1DK base 对 full/restricted 候选流
  各自推进 tracker；按 frame 对齐候选，按 birth GT 对齐轨迹。
- Parts adopted：base simulator（L1DK Kalman）、EGRA 特征、
  Hungarian+threshold、birth/lifecycle。
- Parts intentionally not adopted：不用未来帧；不把 restricted view
  的 GT 身份当推理输入。

## Module: L4SpecEqAssociator（shared identity core + spec conditioning）

- Scientific purpose：一个 checkpoint 统一多域 + 多 spec；spec 只决定
  WHAT to track，shared core 决定 HOW to track。
- Official references inspected：CAMELTrack GAFFE set-level interaction
  （46a74bb）；V2-SAM visual prompt matcher + contrastive alignment
  （`references/l4/v2-sam`，31c3babf）；NOVA class split / hybrid prompt
  （`references/l4/nova`，4358a627）。
- Observed implementation：U0（L1DAssociator）核心不变；type-level
  spec embedding（ALL/category/instance）只注入 set-encoder token，
  不改变 pair head 结构；U0 权重完整初始化。
- Parts adopted：EGRA set transformer + bounded residual + reliability
  gate；spec 作为共享、非 dataset-specific、有界条件。
- Parts intentionally not adopted：不用 category one-hot 进 pair MLP；
  不做 dataset-specific MoE/router；不引入大 VLM。

## Module: Assignment / State Consistency Loss

- Scientific purpose：让 common objects 在两视图的 identity
  decision/state 一致（permutation-invariant）。
- Official references inspected：Path Consistency loss
  （f4b7d26d）；V2-SAM `get_contr_loss` 双向对比（31c3babf）；
  UniTrack 轨迹一致正则（afdd9869）。
- Observed implementation：row/col softmax 在 common candidates /
  common tracks 上的对称 KL；common track token 的 cosine 一致性。
- Parts adopted：permutation-invariant 的对齐方式（common set 内
  重归一化）；对称 KL；state cosine 作为轻正则（lambda=0.1）。
- Parts intentionally not adopted：不做 MSE 两个不同大小矩阵；
  不要求 raw ID 相等；不用 teacher-student 的 stop-grad（symmetric）。

## Module: TAO Cache Recovery

- Scientific purpose：恢复 open-world 长尾域证据，不重跑
  LocateAnything。
- Observed implementation：manifest key 与 cache 路径差一层
  `train/<SOURCE>/`；通过 `cache_key` 覆盖修复。
- Parts adopted：`tools/fix_tao_manifest.py` + `build_candidates`
  cache_key 优先。
- Parts intentionally not adopted：不改共享缓存；不复制 safetensors；
  不使用 `.broken` 帧。

## 附录 L — Pilot Results

# Stage L4 — Specification-Equivariant Training Pilot

日期：2026-08-10。

## 1. 设定

- 共享核心：U0（L1DAssociator，0.49M），U0 checkpoint 初始化；
- 结构：`L4SpecEqAssociator` = U0 core + type-level spec embedding
  （ALL/category/instance，只注入 set-encoder token，非 dataset-specific）；
- 配对数据：15,851 pairs（BDD 7,645 + DanceTrack calib 8,006 +
  MOT17 180 + MOT20 20）；
- 训练：20 epochs，batch 64，1 GPU，lr 3e-4，OneCycle；
  lambda_assign=1.0，lambda_state=0.1。

变体：

| tag | 一致性损失 |
|---|---|
| A2 | 无（naive spec-conditioned） |
| A5 | row/col 对称 KL（birth-GT 轨迹对齐）+ track-state cosine |
| A5p（一次最小修正） | partition-level co-assignment MSE + state cosine |

## 2. 结果：Cross-Spec Drift（P0 vs P1，最优 ID 映射后）

### BDD100K（200 视频，均值聚合）

| Spec | U0 drift | A2 drift | A5 drift | A5p drift |
|---|---:|---:|---:|---:|
| car | 0.3291 | 0.3502 | 0.3262 | 0.3398 |
| bus | 0.4259 | 0.4590 | 0.4495 | 0.4479 |
| truck | 0.3968 | 0.4178 | 0.4022 | 0.4242 |
| pedestrian | 0.4878 | 0.5048 | 0.4793 | 0.4996 |
| bicycle | 0.4845 | 0.5103 | 0.4948 | 0.5309 |
| other vehicle | 0.4744 | 0.4744 | 0.4391 | 0.4519 |

结论：A2 全面变差；A5 只在少量类别小幅改善；A5p 全面接近或差于 U0。
ALL 模式 audit 均值：U0 0.3518 → A2 0.3277 / A5 0.3351 / A5p 0.3364
（官方 pooled TrackEval 三者与 U0 完全一致，见 `reports/l4_ac_results.md`）。

### DanceTrack val（25 视频）

| Spec | U0 drift | A2 drift | A5 drift | A5p drift |
|---|---:|---:|---:|---:|
| person | 0.3225 | 0.3367 | 0.3394 | 0.3346 |
| inst:auto | 0.3112 | 0.3272 | 0.3168 | 0.3314 |

A2/A5/A5p 均未降低 drift；ALL audit 均值：U0 0.4514 → A2 0.4535 /
A5 0.4422 / A5p 0.4457。

### TAO（105 视频，open-world）

| Spec | U0 drift | A2 drift | A5 drift | A5p drift |
|---|---:|---:|---:|---:|
| baby | 0.0675 | 0.1142 | 0.1003 | 0.1073 |
| car_(automobile) | 0.2406 | 0.2293 | 0.2481 | 0.2456 |
| dog | 0.1152 | 0.2166 | 0.1198 | 0.1751 |
| inst:auto | 0.1430 | 0.1865 | 0.1955 | 0.1677 |

## 3. Pilot Gate 判定

| Gate | 结果 |
|---|---|
| A. Cross-spec consistency 显著下降（≥25–30% relative） | FAIL：A5 多数类别无改善甚至变差 |
| B. Selected-object tracking ≥ naive pre-filter | 基本持平/略降（Dance inst P1 AssA 0.8300 vs U0 P1 0.8406） |
| C. ALL preservation（≤0.3–0.5pp 下降） | FAIL：BDD −1.67pp，Dance −0.92pp |
| D. Cross-domain | FAIL：TAO drift 反而上升 |

## 4. Stage Decision

```text
L4_PILOT_GATE_FAIL
（逐帧 assignment/state consistency 不足以消除 temporal identity drift）
```

一次最小修正（A5p：partition-level co-assignment consistency）正在
训练验证后同样失败（Dance inst drift 0.3314 > U0 0.3112；
BDD car 0.3398 > U0 0.3291）。按任务书停止堆模型，进入
failure analysis。

## 附录 M — Full Multi-Domain Training

# Stage L4 — Full Multi-Domain Multi-Spec Training

状态：**NOT_EXECUTED**。

原因：Pilot Gate 未通过（`L4_PILOT_GATE_FAIL`）。任务书规定早期失败
直接进入 failure analysis + final report，不机械执行正式 multi-domain
multi-spec training。

现有 A2/A5 训练本身已覆盖 BDD + DanceTrack + MOT17 + MOT20 的
paired views（15,851 pairs，20 epochs，1 GPU each），可视为
pilot-scale 的多域多 spec 训练；其结论见 `reports/l4_pilot.md`。

## 附录 N — Association-Controlled Results

# Stage L4 — Association-Controlled Results（官方 TrackEval，ALL 模式）

日期：2026-08-10。协议与 L3 完全一致（fresh per-video OnlineTracker，
L1DK shell，thr 0.25 / delta 0.3；`tools/eval_l3.py` +
`tools/run_l1d_trackeval.py`，官方 TrackEval）。

| Domain | U0 AssA | U0 IDF1 | U0 IDSW | A2/A5/A5p AssA | A2/A5/A5p IDF1 | A2/A5/A5p IDSW |
|---|---:|---:|---:|---:|---:|---:|
| DanceTrack val | 0.4169 | 0.5694 | 2,588 | 0.4169 | 0.5694 | 2,588 |
| MOT17 | 0.6050 | 0.5825 | 259 | 0.6050 | 0.5825 | 259 |
| MOT20 | 0.2950 | 0.4012 | 2,406 | 0.2950 | 0.4012 | 2,406 |
| BDD 11-class | 0.2881 | 0.2923 | 11,042 | 0.2881 | 0.2923 | 11,042 |
| Macro AssA | 0.4013 | — | — | 0.4013 | — | — |

说明：

1. 官方 TrackEval 的 ALL 模式数值在 U0/A2/A5/A5p 间完全一致（4 位小数）；
2. `l4_restriction_audit` 的 per-video 均值 ALL 指标有微小差异
   （BDD 0.3518 → A5 0.3351、Dance 0.4514 → A5 0.4422），属于
   均值聚合对小变化的放大，官方 pooled 指标不受影响；
3. 因此 **A2/A5/A5p 保持了 ALL 模式的标准 TrackEval**（Gate C 在官方
   指标上通过），但 cross-spec consistency 未改善（Gate A 失败）。

主结果仍以官方 TrackEval 为准；audit 均值只作诊断。

## 附录 O — Efficiency

# Stage L4 — Efficiency Notes

日期：2026-08-10。

## 1. 模型

- L4SpecEqAssociator：0.488M 可训练参数（U0 core 0.49M + type-level
  spec embedding/projection ≈ 2K）。
- 无大 VLM / 无新增 prompt encoder；spec 只作为 token 级有界条件。

## 2. 训练

- 15,851 paired samples（BDD 7,645 / Dance 8,006 / MOT17 180 / MOT20 20）；
- batch 64，20 epochs，U0 初始化，1 GPU；
- 实测 ≈ 3.4 min/epoch（2 views forward/backward），20 epochs ≈ 58 min
  （A2/A5 各 1 GPU）。

## 3. 推理审计（frozen U0，同一脚本）

| Domain | 规模 | 耗时 |
|---|---:|---:|
| BDD（200 视频 × 12 specs） | 2,400 tracker runs | ~160 s |
| DanceTrack val（25 视频 × 3 specs） | 75 runs | ~9 min |
| TAO（105 视频 × 6 specs） | 630 runs | ~35 s |

## 4. 与 post-filter 对比

- P0（Track-All-Then-Filter）：每帧必须处理全部候选，之后丢弃非 spec
  轨迹；
- P1 / L4：每帧只处理 spec 候选（候选/token 数下降），同时 L4 训练
  一致性以保持身份稳定；
- 本项目 AC 协议固定候选，token 节省不是主 claim；只在报告记录。

## 附录 P — LODO

# Stage L4 — LODO（Leave-One-Domain-Out）

状态：**NOT_EXECUTED**。

原因：Pilot Gate 未通过（`L4_PILOT_GATE_FAIL`）。按任务书，早期失败
直接进入 failure analysis + final report，不机械执行后续 LODO。

若后续重设计 trajectory-level consistency 并通过 pilot，LODO 计划：

- Leave-BDD-Out：训练不含 BDD，测试 BDD multi-class/category；
- Leave-DanceTrack-Out：训练其他域，测试 DanceTrack dense same-class；
- held-out 域不参与 training / calibration / threshold 选择。

## 附录 Q — Leave-One-Spec-Type-Out

# Stage L4 — Leave-One-Spec-Type-Out

状态：**NOT_EXECUTED**。

原因：Pilot Gate 未通过（`L4_PILOT_GATE_FAIL`），不机械执行。

若后续通过 pilot，计划：训练 ALL + category + box + visual，
测试 point；或训练 ALL + category + point + box，测试 visual
exemplar，验证不是 prompt-type lookup。

## 附录 R — Ablations

# Stage L4 — Ablations

日期：2026-08-10。

## 1. 已执行

| tag | 定义 | Cross-spec drift（Dance inst） | ALL AssA（Dance/BDD） |
|---|---:|---:|---:|
| A0 = U0 | frozen shared core | 0.3112 | 0.4514 / 0.3518 |
| A1 = P1 | U0 pre-filter（评估方式，非独立模型） | — | — |
| A2 | spec-conditioned，无一致性 | 0.3272 | 0.4535 / 0.3277 |
| A5 | + row/col KL + state cosine | 0.3168 | 0.4422 / 0.3351 |
| A5p | + partition co-assignment MSE（一次最小修正） | 0.3314 | 0.4457 / 0.3364 |

## 2. Where to Inject Specification

- Late selection（P0，Track-All-Then-Filter）：一致性最好（同一模型
  只跑一次），但受限对象指标差、IDSW 高；
- Early conditioning（P1，pre-filter）：受限对象指标最好
  （Dance inst AssA 0.8406 / IDSW 72），但身份与 ALL 不一致；
- Proposed（共享 core + paired consistency）：目标是把 P1 的
  selected-object 质量与 P0 的稳定性合并；当前机制未达成。

## 3. 未执行

- A3（仅 assignment consistency）与 A4（仅 state consistency）：
  因 A5 已失败，按任务书不再细分；如需论文对照可在后续重设计后补。

## 附录 S — Failure Analysis

# Stage L4 — Failure Analysis

日期：2026-08-10。

## 1. 结论

Stage L4 pilot 未通过：paired-view 的逐帧 assignment/state consistency
（A5）与 partition-level co-assignment consistency（A5p，一次最小修正）
都没有降低跨 spec identity drift；官方 TrackEval ALL 模式保持不变，
audit 均值 ALL 轻微下降。

## 2. 为什么一致性训练无效

### 2.1 身份漂移是时间现象，不是单帧分配现象

审计指标里的 drift 是「同一 GT 对象在两个视图中的长期轨迹 partition
不一致」，主要由不同时刻的 merge/split/switch 造成。A2/A5 的
consistency loss 只约束单帧的 assignment 分布（row/col KL）或单帧
co-assignment 矩阵（partition MSE），没有约束「跨时间身份轨迹」，
因此不能降低 drift。

### 2.2 轨迹对齐噪声

A5 用 birth-GT 对齐两视图轨迹。当 base tracker 发生身份错误时
（track 的 birth 身份 ≠ 当前匹配对象的身份），这种对齐把「错误身份」
当共同身份来压一致，反而把错误固化。A5p 改为 partition-level 对齐
（co-assignment 矩阵 MSE），绕开轨迹对齐，但该 loss 数值接近 0
（~1e-4），仍只覆盖单帧 co-assignment，无法约束跨帧 ID 迁移。

### 2.3 base affinity 主导 + 集合级特征

EGRA 的 final = base + gated residual；base 本身包含
`log_n_cand/margins/top1` 等集合级特征，候选子集变化会同时改变
base 与上下文 token。0.49M 模型在 20 epochs 内没有学到能抵消
set-context 变化的残差。

### 2.4 训练-评估协议不匹配

训练时每帧 paired softmax 被一致性正则化；评估时 OnlineTracker 在
每视图独立做 Hungarian + 生命周期。softmax 的一致不保证 Hungarian
分配与跨帧 ID 迁移一致。

## 3. 为什么不继续

- 任务书允许「一次最小机制修正」，已执行（A5p）；
- 不允许无限调 lambda / 堆容量；
- 当前证据指向「逐帧 association-level consistency」不是正确的
  mechanism，需要 trajectory-level / differentiable-tracking 层面
  的重设计，超出本阶段时间预算。

## 4. Stage Decision

```text
L4_PILOT_GATE_FAIL
L4_NOT_SUPPORTED（pilot mechanism）
Problem Signal：L4_SPEC_RESTRICTION_SIGNAL_SUPPORTED（真实存在）
ICLR readiness：NOT_READY
```

## 5. 保留资产

- P0/P1 审计证明 specification restriction 真实改变身份（U0）；
- 配对数据、指标（co-identity agreement）、TAO 恢复；
- A2/A5/A5p 训练管线可复用；
- 失败模式记录：单帧 consistency ≠ temporal identity consistency。
