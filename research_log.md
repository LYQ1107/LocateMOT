# Research Log

原则：只记录实验假设、失败现象、原因判断、修改内容、结果变化、是否保留。

## 2026-08-08 — Stage L1-B 启动

- 假设：LocateAnything ObjectToken 经统一 Identity Adapter 可成为跨类别、跨数据集的 persistent identity token（替代 L1-A 失败的时间/trajectory 路线）。
- 决策：先做 L1-B0 数据集审计 + 2025–2026 方法审计；禁止 MOTSynth；禁止 dataset-specific fine-tune。
- 修改：AGENTS.md 增加“先确认项目身份（LocateMOT）”“先读代码→短计划→实现/实验”“research_log”原则。
- 结果：待 L1-B0 输出（dataset_statistics.json / l1_b_dataset_identity_audit.md）。
- 保留：是。

## 2026-08-08 — L1-B0 完成

- 假设：ObjectToken → Identity Adapter 可跨类别跨数据集成立。
- 结果：
  - 方法审计：OG-ReID 官方仓库未公开（NO_VERIFIED_OFFICIAL_IMPLEMENTATION）；
    VICP/UPCL/UniTrack 已克隆固定 commit；FDTA/MOTIP 复用 L1-A 审计。
  - 数据集审计：DanceTrack/MOT17/MOT20/YT-VOS/MOSE train/C-TAO 可用；
    BDD100K 本地仅检测标签（不可做身份训练）；TAO 官方缺失；MOTSynth 禁用；
    MOSE valid 标注隐藏。
- 修改：新增 tools/l1_b_dataset_audit.py、docs/l1_b_*、reports/l1_b_storage_plan.md。
- 保留：是（审计结果作为 pilot 数据选择依据）。

## 2026-08-09 — L1-B pilot 完成（IDENTITY_SIGNAL_NOT_SUPPORTED）

- 假设：ObjectToken 经轻量 Identity Adapter（InfoNCE）可在 Same-Category
  Retrieval 上跨数据集超过 raw PBD。
- 失败现象：R4 full 在全部数据集不优于 raw；R4 pbd-only 只在 DanceTrack
  +4.9pp（0.919→0.968），MOT17/TAO/YT-VOS/MOSE 不提升或下降。
- 原因判断：raw PBD 在 dense 场景已强（接近协议上限）；region 输入稀释；
  704 身份小样本不足以跨数据集迁移；deformable/sparse 目标未改善。
- 修改：full→pbd-only 消融；cache/评估/训练基建全部保留。
- 结果：pilot gate 未通过 → L1_B_IDENTITY_SIGNAL_NOT_SUPPORTED。
- 保留：基建保留；“adapter 已成功”结论不保留。

## 2026-08-09 — v2：加入 BDD100K tracking 重测

- 修改：修正 BDD100K 审计（masa box_track_20 本地可用）；pilot 增加
  BDD 240 帧；缓存超集 2,932 帧；训练 1,064 身份（full+pbd，30 epochs）。
- 结果：full adapter：BDD +3.0pp / TAO +8.7pp / DanceTrack +6.5pp；
  MOT17 -3.6 / MOT20 -4.7 / YT-VOS -1.3 / MOSE -1.7；macro +1.0pp。
- 原因判断：数据规模部分改善 multi-class road；dense/deformable 仍不迁移。
- 保留：是（作为后续 multi-class 方向证据）；阶段决策仍
  L1_B_IDENTITY_SIGNAL_NOT_SUPPORTED。

## 2026-08-09 — Road LODO（BDD/TAO 12k 帧，7.8k 身份）

- 假设：数据规模不足导致 adapter 不迁移；扩大 BDD/TAO 并用严格 LODO 可验证
  multi-class road 方向。
- 失败现象：in-domain A_bdd 在 BDD −15.0pp；LODO A_tao→BDD −15.9pp、
  A_bdd→TAO −7.5pp；仅 A_road 在 TAO +4.0pp。
- 原因判断：adapter 学习 dataset-specific shortcut，非 universal
  identity representation；“数据规模不足”假设被排除。
- 修改：缓存 BDD 8,001 + TAO 4,200 帧；训练 A_bdd/A_tao/A_road；
  修复迁移后 ACL 不可读缓存（删除 272 个残缺帧并重建）。
- 结果：road LODO 未通过；阶段决策维持 L1_B_IDENTITY_SIGNAL_NOT_SUPPORTED。
- 保留：是（作为“representation 路线关闭，转向 detection/association”证据）。

## 2026-08-09 — Stage L1-C 启动

- 假设（H1–H3）：LocateAnything 冻结（UAF）或轻量 LoRA 适配（UAL）下，
  一个可训练的 set-level contextual Association Decoder 能学习跨数据集共享
  association rule；不再走 universal identity vector。
- 计划：先审计 2025–2026 官方实现（Eagle/LoRA、MOTIP、FDTA、HATReID-MOT、
  OVTR、COVTrack++、GOVTrack、SAM2-OV）→ 数据/协议审计（复用 L1-A/L1-B
  cache）→ association-controlled 协议与 fixed manifest → 基线 C0–C4 →
  UA decoder → UAF 训练/评估 → LoRA smoke/训练 → UAL → pilot gate →
  （通过才）full/LODO/ambiguity/why-not-IoU → 最终报告与 commit。
- 约束：seed=20260806；禁止 MOTSynth；禁止 dataset-specific head；
  UAF/UAL 同构；LoRA 先审计官方 Eagle 训练代码。
- 保留：是。

## 2026-08-08 — TrackOCD 误执行记录

- 现象：上一轮按错误附件执行了 OCD_OVMOT/TrackOCD Phase 4M，生成了与 LocateMOT 无关的报告。
- 原因判断：交接摘要与附件指向 TrackOCD，未先向用户确认项目身份。
- 修改：该轮全部工作隔离在 OCD_OVMOT 目录，未污染 LocateMOT；AGENTS.md 增加项目身份确认规则。
- 结果：LocateMOT 无改动；后续任务先确认项目。
- 保留：否（TrackOCD 产物不进入 LocateMOT）。

## 2026-08-09 — Stage L1-C 执行（审计 + 协议 + UAF）

- 假设（H1）：冻结 LocateAnything 下，set-level contextual Association
  Decoder 可学习跨数据集共享 association rule（不再走 universal identity）。
- 修改：
  - 完成 2025–2026 官方审计（MOTIP/FDTA/HATReID/OVTR/COVTrack/HNCD-MOTR；
    COVTrack++/GOVTrack/SAM2-OV 无官方代码）；Eagle LoRA 官方审计（A100
    需 sdpa）。
  - 完成数据/协议审计：L1A DanceTrack 67,578 帧 + L1B 多数据集 14,651 帧；
    fixed candidate manifest（hash 冻结）。
  - 实现 UA decoder（candidate self-attn + track history encoder +
    relation-bias cross-attn + K+1 assignment）与 clip 训练管线。
  - 修复 association-controlled 协议：AC 模式所有候选必须输出，
    min_hits 门控会破坏 DetA 一致性。
- 结果：
  - UAF smoke 30 步 loss 有限（1.8→3.3，随机初始），梯度正常；
    正式训练 50k 步已在 GPU1 运行。
  - C0/C1/C2/C3 AC 基线已完成，C4(b6) AC 运行中；TrackEval 待汇总。
- 保留：是（审计/协议/UA 基建全部保留）。

## 2026-08-09 — UAF 早期评估失败：NEW 类过拟合

- 假设：K+1 CE 可直接训练 association。
- 失败现象：step5000 检查点在 DanceTrack val AC 评估 AssA=0.0026、
  IDF1=0.009、IDSW=186k——几乎所有候选都被判为 NEW。
- 原因判断：BDD/TAO 未匹配候选占多数，CE 被 NEW 主导；new_head 学到
  大正 logit，推理时 NEW 恒赢。
- 修改：NEW 类 CE 权重 0.2；每帧保留全部 matched 候选 + 最多 8 个
  unmatched 候选（按 gen score），防止 NEW 主导；从 step5000 断点续训。
- 结果：待验证（step10000 检查点将重新跑 UA val）。
- 保留：训练损失/采样修改保留；未保留“NEW 过拟合”模型行为。

## 2026-08-09 — UAF step10000 复查

- 结果：step10000（修复后训练 5k 步）val AssA=0.127、IDF1=0.233、
  IDSW=50,846（vs step5000 AssA=0.003/IDSW=186k，大幅改善但仍低于
  C0/C1）。DetA=0.947 协议有效。
- 原因判断：模型仍偏向 NEW——单帧诊断 scores∈[-1.6,3.0]（mean .42），
  new_logits≈.68–1.24，仅 39% (candidate,track) 对分数超过 new。
- 修改：暂无（继续训练至 50k）；评估后若仍偏 NEW，在 calibration split
  上校准 new_logit margin（shared calibration，非 dataset-specific）。
- 保留：继续训练。

## 2026-08-10 — UAF 最终结果（pilot gate 不通过）

- 结果：50k 步 final checkpoint，NEW-margin 校准（calibration 8 视频，
  最优 margin 3.5）后 val：AssA=0.133、IDF1=0.270、IDSW=26,804、
  HOTA=0.355；均未超过 C4（0.155/0.308/16,456）。
- 原因判断：训练损失稳定但 association 决策仍偏 NEW/ID 不稳定；
  简单加大 margin 只能部分补偿，无法达到 IoU/motion 水平；
  当前 contextual association 设计在 DanceTrack AC 协议下不成立。
- 修改：无（不堆容量）；margin=3.5 作为 shared calibration 保留。
- 结果：UAF pilot gate 不通过；Route B（LoRA）继续作为诊断。
- 保留：UA 基建/协议保留；不保留“UAF 已成功”结论。

## 2026-08-10 — LoRA smoke 通过（Route B 环境打通）

- 假设：官方 Eagle LoRA 训练入口可在 A100 上运行（magi→sdpa/eager）。
- 失败/修改：
  - PyPI 镜像缺 deepspeed 0.15.4/liger 0.3.1 → GitHub 源码引入；
  - 官方 vision attn 硬编码 flash_attention_2 → eager（sdpa 路径有
    mask bug，eager 偶发 driver softmax error → CPU softmax 兜底）；
  - 官方 shell 需 LAUNCHER=pytorch + RANK/LOCAL_RANK/WORLD_SIZE。
- 结果：5 步 smoke loss=3.15，trainable=119.7M（LLM LoRA rank 64 +
  MLP），save→load→generate 正常。
- 修改记录：docs/official_code_modifications.md。
- 保留：是（作为 UAL 的训练基座）。

## 2026-08-10 — LoRA 训练完成；特征提取阻塞

- 结果：LoRA grounding 300 步训练完成（train_loss=1.27，7.3 min；
  checkpoints/lora/checkpoint-300 加载+生成验证通过）。
- 失败现象：tools/cache_l1c_lora.py（LoRA merge_and_unload +
  ObjectTokenExtractor）单帧 10 分钟未完成；直接 model.generate 快速
  可用，说明 instrumented _generate_loop 与 merged 模型/旧驱动组合不兼容
  或极慢。
- 原因判断：待调试 MTP 快速解码路径（n_future_tokens/hooks）或改用官方
  batch inference 后再接 hidden 提取。
- 保留：LoRA checkpoint 与提取工具保留；不保留“UAL 已完成”结论。

## 2026-08-10 — L1-D：EGRA 架构实现与第一轮结果

- 假设：保留强 base affinity（IoU+PBD+运动融合）+ set-level 有界残差 +
  reliability gate，可修复 base 的 set-level ID 错误而不破坏先验。
- 修改：实现 `locatemot/models/l1d_association.py`（pair/track/cand 特征、
  2 层 set transformer、0.6*tanh 残差、track 级 gate；row+col CE 主损失 +
  gate BCE + 保留正则）；离线 base 模拟器（`tools/build_l1d_dataset.py`）
  验证 C3 等价（calib AssA 0.384/IDSW 573 与 L1-C 审计一致）。
- 结果（DanceTrack val AC）：L1D(0.6 delta) AssA 0.378/IDSW 3053 < base C3
  0.393/2981；calibration 上 delta=0.3 时 AssA 0.3825/IDSW 565（≈base，
  IDSW 略降）；correction audit（帧间连续性）：helpful 5252 vs harmful 597、
  precision 0.898、coverage 0.780、preservation 0.990——但官方 TrackEval
  AssA/IDSW 不奖励该连续性改善。
- 原因判断：base 的身份正确率仅 44.5%（大量 swap 状态），模型无法用
  有界残差恢复（需 delta>0.6 占多数）；学习到的“连续性保持”与 TrackEval
  AssA 不对齐。
- 修改：base 运动 cue 从常速度外推升级为 Kalman（C1 同款），重新校准：
  (0.4,0.2,0.4,t0.25) calib AssA 0.424/IDSW 512（超过 C1 的 0.419/2916
  val 水平）。重新训练 L1D-K（40 epochs）。
- 保留：EGRA 框架保留；旧 constant-velocity base 结果保留为对照。

## 2026-08-10 — L1-D Kalman base 训练（进行中）

- 假设：更强 base（Kalman motion）上 residual 更易正向。
- 修改：`compute_affinity_features` 支持 Kalman pred boxes；simulator 与
  OnlineTracker L1D 均按 C1 生命周期维护 Kalman。
- 结果：待评估。

## 2026-08-10 — L1-D Kalman base 训练完成 + 评估

- 假设：更强 base（Kalman motion）上 residual 更易正向。
- 修改：base = 0.4 IoU + 0.2 PBD + 0.4 Kalman-motion（thr 0.25，
  calibration 网格）；EGRA 40 epochs 8,360 步。
- 结果（DanceTrack val AC）：L1DK base AssA 0.4165/IDF1 0.563/IDSW 2558
  （vs C1 0.4193/0.566/2916，C3 0.3934/0.5367/2981）；L1DK_d03
  0.3993/0.5503/2579 → residual 在 calibration +1.9pp、val −1.7pp。
- 跨域：MOT20 强正（IDSW −35.5%）、MOT17 持平、BDD AssA −4.5pp/
  IDSW −8.2%；macro AssA −1.6pp / IDF1 +0.7pp / IDSW rel −10.9%。
- 原因判断：residual 学到帧间连续性（val continuity 0.841→0.951，
  helpful/harmful=27,993/3,187），但 TrackEval AssA/IDSW 不奖励该
  连续性；base 身份正确率仅 44.5%，多数错误需 delta>0.6 无法修复。
- 修改：无（不堆容量；不 tune val）。
- 结论：L1_D_PARTIAL；采用 L1DK base，residual 不部署。
- 保留：L1DK base（新最强 AC 基座）；EGRA 作为消融。

## 2026-08-10 — Stage L2 启动：baseline 矩阵 + 文献审计

- 假设：L1DK base 是最强统一基座；local correctness 与 future
  trajectory utility 不同构。
- 实验：四域 AC 矩阵（C0/C1/C2/C3/L1DK/L1DK_d03）在 DanceTrack val、
  MOT17、MOT20、BDD（同一固定候选 manifest）全部重跑。
- 结果：L1DK base macro AssA 0.4062 最高，DanceTrack/MOT17/BDD 三域
  最优；MOT20 由 L1DK_d03 略优。BEST_STRONG_BASE = L1DK base。
- 文献：TDLP（下一帧 link prediction）、SambaMOTR（自回归 query）、
  TRACT（轨迹感知）、UniTrack（轨迹平滑 hinge）、Path Consistency、
  QuoVadis、FDTA、HATReID-MOT、HNCD-MOTR 全部实际 clone 阅读；
  无直接等价“counterfactual future utility”方法。
- 决策：进入 oracle headroom 实验（进行中）。

## 2026-08-10 — Counterfactual Oracle（进行中）

- 假设：存在“局部正确但未来差 / 局部错但未来好”的决策事件，且 oracle
  best action 相对 base 有可学习 headroom。
- 实验：L1DK base 精确重放（与 baseline 100% 一致），对冲突组件枚举
  6-8 个候选 action，冻结 base policy rollout H∈{4,8,16,32} 帧，
  用 TrackEval 同款公式算 windowed AssA/IDF1/IDSW。
- 当前状态：DanceTrack val 25 视频 oracle 运行中；BDD/MOT17/MOT20 待跑。

## 2026-08-10 — Oracle 完成：Gate 1 判定 LOW（Stage L2 停止大训练）

- 假设：oracle future-best action 相对 L1DK base 有可学习整视频 headroom。
- 实验：单事件窗口（H4–H32）+ 端到端 greedy oracle（privileged）。
- 结果：
  - 单事件：DanceTrack H32 mean gain +0.74pp（frac 21.9%）、
    BDD H16 +1.01pp（frac 61.7%）；
  - 端到端：DanceTrack +0.02/+0.06pp、BDD 均值 −0.88pp、
    MOT17 −2.32pp；IDSW 全部变差；
  - mismatch：DanceTrack H32 219/1000 future-best≠base
    （local_correct_future_bad 128 / local_wrong_future_good 60）；
    BDD H16 460/745（173/110）。
- 原因判断：base 短窗口已接近最优；窗口效用与全局 ID 统计不同构；
  动作经 base 再优化后趋同；历史污染不可在短窗口修复。
- 修改：无（不训练 TUM，按任务书停止条件）。
- 结论：`L2_ORACLE_HEADROOM_LOW`；进入失败分析与最终报告。
- 保留：oracle 工具链、windowed AssA 校验、污染审计、文献审计。

## 2026-08-10 — Stage L3：审计 + U0/U1 pilot

- 假设：latent regime 条件化共享核心能减轻多域负迁移。
- 审计：SAM3/SAM3.1、GLEE、OVTR、OVTrack、grounded-sam-2、SAM2MOT、
  STORM-Bench、QTrack、AnyTrack 全部 clone 阅读；Claim 2 被
  SAM3/GLEE 强碰撞；BDD manifest 已含 11 类 GT。
- 修改：`locatemot/models/l3_unified.py`（RegimeEncoder + FiLM）；
  `tools/train_l3.py`、`tools/eval_l3.py`、
  `tools/l3_regime_diagnostics.py`、`tools/analyze_l3_routing.py`。
- 结果（四域 AC fresh 协议）：
  - U0 macro AssA 0.4013 > L1DK 0.3944（MOT17 +1.67pp、
    MOT20 +1.72pp、BDD −0.70pp、DanceTrack +0.04pp）；
  - U1 macro 0.3915（仅 MOT20 +0.08pp；DanceTrack −1.19pp、
    MOT17 −1.91pp、BDD −0.89pp）；
  - z_regime domain classifier 96.6% → dataset shortcut。
- 原因判断：regime 特征与 dataset 共线（BDD 5fps gap 等）；local CE
  目标与轨迹级指标不同构；association 可条件化空间小。
- 结论：`L3_REGIME_NOT_SUPPORTED + REGIME_ROUTER_DATASET_SHORTCUT`；
  不训练 B/不堆容量。
- 保留：U0（shared learned baseline）、多类 BDD 协议、全套审计。

## 2026-08-10 — Stage L4：Restriction audit（P0 vs P1）

- 假设：specification（候选集限制）会真实改变 U0 的 persistent
  identity，且 pre-filter（P1）与 track-all-then-filter（P0）在
  common objects 上不一致。
- 实验：frozen U0 在 BDD 200 视频 11 类、DanceTrack val 25 视频、
  TAO 105 视频上跑 ALL 与受限候选流；用最优 ID 映射后的
  co-identity agreement（permutation-invariant）度量。
- 结果：
  - BDD category drift 33–67%（car 33%、pedestrian 49%、truck 40%、
    bus 43%、trailer 67%）；P1 的 IDSW 几乎全部显著下降
    （bus −82%、truck −66%），多数 AssA 上升；
  - DanceTrack person drift 32%、top-2 instance drift 31%；
    instance P1 AssA +28.1pp（0.5592→0.8406）、IDSW 799→72；
  - TAO car drift 24%、instance drift 14%，P1 改善；
  - ALL vs ALL 自检 agree=1.0，toy-case 指标验证通过。
- 原因判断：Hungarian 竞争、set-level 特征（log_n_cand/margins）、
  track-state 更新、P0 过滤产生 gap 共同导致身份漂移。
- 决策：`L4_SPEC_RESTRICTION_SIGNAL_SUPPORTED` → 进入 paired-view
  spec-equivariant training pilot（A2 naive vs A5 full）。
- 保留：审计工具链、TAO cache_key 修复 manifest。

## 2026-08-10 — Stage L4：Paired-view pilot 训练（进行中）

- 假设：paired full/restricted views + assignment/state consistency
  能降低跨 spec 身份漂移，同时保持 ALL 与受限视图的 TrackEval。
- 修改：`locatemot/models/l4_spec_eq.py`（U0 core + type-level spec
  embedding，仅注入 token）、`tools/build_l4_pairs.py`（15,851 pairs：
  BDD 7,645 + Dance calib 8,006 + MOT17 180 + MOT20 20）、
  `tools/train_l4.py`（local CE + row/col assignment KL + track-state
  cosine，lambda_assign=1.0 / lambda_state=0.1）。
- 状态：A2（无一致性）与 A5（完整一致性）各 20 epochs 训练中，
  U0 初始化，batch 64，1 GPU each。

## 2026-08-10 — Stage L4：Pilot 评估（A2/A5 失败 + 一次修正 A5p）

- 假设：paired-view assignment/state consistency 能降低跨 spec drift。
- 结果（P0 vs P1 最优 ID 映射）：
  - A2：BDD 多数类别 drift 变差（car 0.329→0.350、pedestrian
    0.488→0.505）；Dance inst 0.311→0.327；TAO inst 0.143→0.187；
  - A5：BDD car 0.329→0.326、pedestrian 0.488→0.479（小幅改善），
    bus/trailer 变差；Dance inst 0.311→0.317；TAO inst 0.143→0.196；
  - 官方 TrackEval ALL：A2/A5 与 U0 完全一致（4 位小数），
    说明 ALL 未退化，但一致性也未改善；
  - A5 的 audit 均值 ALL 有微小下降（BDD 0.3518→0.3351），
    官方 pooled 指标不受影响。
- 原因判断：身份漂移是**时间/轨迹级**现象，单帧 assignment 或
  partition 一致性无法约束跨帧 ID 迁移；birth-GT 对齐在身份错误
  时会把错误固化。
- 修改（一次最小修正）：A5p = partition-level co-assignment MSE +
  state cosine（permutation-invariant、不对齐轨迹），20 epochs 训练。
- 结果（A5p 评估）：Dance inst drift 0.3314（> U0 0.3112）、
  BDD car 0.3398（> U0 0.3291）、TAO inst 0.1677（> U0 0.1430）；
  part loss 训练中仅 ~1e-4；官方 TrackEval ALL 与 U0 一致。
- 决策：`L4_PILOT_GATE_FAIL + L4_NOT_SUPPORTED`；Problem Signal
  （`L4_SPEC_RESTRICTION_SIGNAL_SUPPORTED`）真实存在但当前机制无法
  修复；停止训练，进入 failure analysis + final report。
