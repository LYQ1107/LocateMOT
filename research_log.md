# Research Log

## 2026-08-11 Stage L5（夜间无人值守）

### L4 评估完整性审计
- 假设：L4 官方 TrackEval 与 U0 完全一致是 bug。
- 实验：逐帧重放 + per-tag 数据目录重跑官方 AC。
- 结果：确认 bug（build_data 只在目录不存在时复制，L4 复用旧 L3 目录）；
  重跑后 A2/A5/A5p 与 U0 确实不同但差异小（AssA ±0.005–0.025），
  无模型跨域一致改善。
- 解释：L4 的 0.488M/20ep frame-level consistency 失败不是评估假象，
  但也不是路线判决依据。
- 保留：reports/l5_evaluation_integrity_audit.md；eval_l4_ac.py 修复。

### Phase 2 文献审计
- 检索 2025/2026：MambaTrack/GatedTemporalFusion/AssociaTR/NOOUGAT/
  DualPathDecoder/DecoderTracker/SOTFormer 等均无直接等价工作；
  新 clone SOTFormer（MIT, bb28e62）、MO-YOLO（AGPL，仅阅读）。
- 保留：docs/l5_reference_audit.md、reports/l5_novelty_collision_audit.md。

### Phase 3 Clip 数据
- 构建 gt（GT-anchored 轨迹）与 u0（U0 rollout）双源、双视图
  （ALL + cat/inst）clip 数据，H=16，PBD float16。
- baseline drift（u0, val 小集）：BDD 46.5%、Dance 68.1%、合计 65.0%；
  Type2（P0 wrong/P1 correct）1554/2598。
- 保留：docs/l5_clip_dataset.md、reports/l5_headroom_analysis.md、
  tools/build_l5_clips.py、tools/l5_drift_eval.py。

### Route A 实现与 overfit
- 假设：缺少 persistent temporal state 是 drift 主因；
  GT-anchored 监督 + cross-spec relation consistency。
- 实现：L5TemporalAssociator（temporal causal encoder + set decoder +
  bounded residual），接入 OnlineTracker variant L5。
- 发现 bug 并修复：collate padding 参与 CE/argmax → 掩码后重启训练。
- 发现 bug 并修复：temporal encoder 的 3D mask + -inf padding 在 torch 2.5
  下产生 NaN（-inf + -inf）；改为 causal 2D mask + zero-padding，验证无 NaN。
- 结果1（gt-only，epoch5 Small）：val_row_acc 0.94（≈base），
  u0 drift 66.0%（baseline 65.0%）→ 未降。
- 解释：gt-clean 轨迹训练不迁移到 u0 含错轨迹；target 虽 GT-anchored，
  但输入分布不一致。
- 修改：训练源改为 gt+u0 混合（target 均为 GT-anchored），spec-weight
  0.2，rel-weight 0，batch 16，pct_start 0.05。
- 发现并修复：Base 因 T×T relation 矩阵 OOM（rel_weight=0 却仍计算）→
  移除 forward 中 rel_mat；collate 大 batch 出现 T=160 的 BDD 样本导致
  显存激增 → Dataset 按样本 cap T≤48（保留有监督 track 优先）。
- 结果2（gt+u0 mixed，ep10 Small）：u0 rowacc 几乎不变
  （BDD train 0.457→0.475、Dance train 0.690→0.690），drift 65% 不变。
- 解释：gt 源稀释 u0 信号；preservation loss 使 delta 太小
  （correct 0.016 / wrong 0.089），模型几乎等于 base；
  cross-spec 目标权重不足。
- 修改：u0-only 训练，pres-weight=0，delta-scale=1.0，
  spec-weight=10（assignment-level KL），T≤48 cap。
- 发现指标错误并修正：u0 target 用 history 多数 GT 会被早期 switch
  污染；改为当前帧 GT 框 IoU 锚定（track_cur_gt）。cur_gt 下 per-frame
  base rowacc 达 0.95-0.98，说明单帧关联基本一致；
  真正的问题是轨迹级 identity chain 漂移。
- 新主指标：视频级全局最优 ID 对齐后的 track-ID 分歧率
  （L4 一致）。baseline：BDD val 53.2% / Dance val 37.9%（在线 rollout）。
- 结果3（u0+cur_gt，delta=1.0，spec-w=10）：BDD 在线 drift ep5 5.6%、
  ep10 20.4%（不稳定）；Dance ep10 39.4%（≈U0 37.9%）。
- 解释：过强 delta 在 Dance（外观模糊）制造新 switch；BDD 有明显收益。
- 修改：delta-scale=0.6 + pres-weight=0.1 + spec-weight=2（温和修正），
  u0+cur_gt，运行中 GPU 6/7。

### Route B 试点（2026-08-11 早）
- 假设：sequence-local ID prediction（MOTIP 风格）直接监督跨 spec
  identity slot。
- 结果：train slot acc 0.93（Small/Base ep20），val 0.14；在线 drift
  BDD 69.3%（U0 53.2%）。
- 判断：小集不迁移 → NOT_SUPPORTED。

### Route A 最终结果（u0+cur_gt，delta=0.6，spec-w=2）
- 在线 drift（ep40）：BDD 28.7%（U0 53.2%，-46%）、
  Dance 29.3%（U0 37.9%，-23%）；ep20 后稳定。
- TrackEval（ep40）：Dance AssA 0.4182（+0.13pp）/IDSW 2558（-30）；
  BDD AssA 0.2951（+0.7pp）但 IDSW 12399（+1357）；
  MOT17 AssA 0.5914（-1.4pp）；MOT20 AssA 0.2763（-1.9pp）。
- 学习曲线：ep1-60 val_row_acc 恒定 0.98；Small==Base（容量饱和）。
- 判定：L5_ROUTE_A_PARTIAL（正机制信号，未过 full-scale）。
- Route C：NOT_EXECUTED（被 A 的 cross-spec KL 部分覆盖）。
- 最终判定：L5_PARTIAL / ICLR_NOT_READY。
- 交付：reports/STAGE_L5_FINAL_REPORT.md（自包含）。

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

## 2026-08-11 — Stage L6：UIDM 主模型（Learned Causal Identity Dynamics）

- 假设：MOT 域差异可被一个共享的、在交互轨迹集合上学习的因果身份
  动力学过程统一（持久记忆 + 集合交互 + 学习化转移 + 生命周期 +
  tracking-level loss + model-in-the-loop）。
- 设计：`locatemot/models/l6_uidm.py`（UIDM-Large ~15M: d=384/6层；
  GRU-like 记忆更新、anchor、set transformer、pair/no_match/new/alive/
  motion 头、soft-switch hinge loss）；训练脚本
  `tools/train_l6_uidm.py`（H=16、scheduled sampling、DDP）。
- 数据：`outputs/l6/data/*`（per-video 帧序列，BDD 200 + Dance 40 +
  MOT17 3 + MOT20 2 + TAO 105），无 MOTSynth，无 dataset ID。
- 实验：冒烟通过（loss/grad/rollout/TrackEval 管线）；第一次全量 DDP
  epoch4 崩溃（DDP unused params）→ `find_unused_parameters=True` 修复；
  NEW/no-match/pair 头零初始化偏置改进（防止推理早期碎片化）。
- 状态：全量训练重启中（3 GPU，4200 steps）；epoch1-3 曾显示 rowacc
  0.6→0.9，但 epoch1 MOT17 TrackEval 仍碎片化（AssA 0.027），判定为
  训练不足 + 头部初始化问题，继续训练后重新评估。

## 2026-08-11 — Stage L6：关键 bug 修复（births 立即死亡）

- 失败现象：训练 loss 下降、rowacc 0.6-0.9，但推理 TrackEval 严重
  碎片化（MOT17 IDSW ~4700、AssA 0.025）；逐帧 logits 显示 pair
  logits 全部为负、new≈1，LSA 全部走 NEW。
- 原因判断：训练 rollout 中 newborn slot 的 alive_logit=0.0，而
  active 判定为 alive_logit>0，导致**所有新生轨迹在同一帧内立即
  失活**——模型从未见过跨帧持久状态，只学到“一切皆 NEW”的退化解；
  推理 shell 却保留新生轨迹，造成 train/inference 状态语义不一致。
- 修改：birth alive=1.0（训练 rollout + 推理 birth 同步）；推理
  NEW margin 保留为可调参数（默认 0）。
- 结果：待验证（全量训练已用修复版重启，epoch1-3 后将重新 TrackEval）。

## 2026-08-13 — Stage L6：最终评估 + 消融（训练已完成）

- 全量训练完成：UIDM-Large 4200 步（epoch 18），最终 rowacc 0.976、
  loss 0.74（仍下降）。
- 主结果（fresh TrackEval）：Macro HOTA 0.5897（+5.2pp）、
  Macro AssA 0.4922（+9.1pp）、Macro IDF1 0.5199（+5.9pp）；
  BDD AssA 0.4866、MOT17 0.6991、MOT20 0.4584 大幅提升；
  **Dance AssA 0.3248 塌陷**（U0 0.4169）。
- 失败分析：Dance 12,543 switches 中 92% 为 gap=1 连续帧错误，
  平均 crowd IoU 0.42 —— PBD 外观证据过强、motion/competition 不足；
  BDD switches 主要发生在 6–10 帧检测缺失后。
- Cross-spec drift：BDD 53.2%→17.0%（大幅改善），Dance 37.9%→34.0%。
- Ablation（MOT17 AssA）：full 0.6991；no-trackloss 0.4678；
  no-memory 0.4724；no-interaction 0.4901；no-lifecycle 0.4876；
  small-3M 0.5458 —— 四项机制各贡献 ~21–23pp，容量贡献 ~15pp。
- 决策：`L6_PARTIAL / SUPPORTED with Dance collapse`；报告完成，
  下一步 cue-reliability 修复 + IDSW 校准 + long-gap memory。

## 2026-08-14 — Stage L7 开始：Unified MOT（ALL / OV / Referring）

- 假设：不同 WHAT-TO-TRACK specification（ALL、open-vocabulary category、
  referring language）可以共享同一个 HOW-TO-TRACK 因果身份动力学核心；
 统一模型 = Specification Encoder（选择目标）+ Shared UIDM（维护身份）。
- 文献审计（官方仓库，见 docs/l7_reference_audit.md）：
  OVTR(ICLR25)、OVTrack(CVPR23)、COVTrack(ICCV25)、QTrack(26)、
  TempRMOT(24)、STORM(26)、ReaMOT(25)、OVT-B(NeurIPS24) 均已阅读 README
  与关键模型/评估代码；无同时满足 closed-set+OVMOT+RMOT+共享身份核心的
  直接等价方法（NO_DIRECT_EQUIVALENT_VERIFIED）。
- 关键碰撞：COVTrack 已公开 association-embedding 级 adaptive
  appearance/motion/semantic 门控融合，因此 cue reliability 不能作为第一
  创新，只能作为 UIDM identity-transition decoder 内部组件并明确区分。
- 数据：TAO 官方 train/val/test 帧+标注本地完整（~354GB），LVIS v1 类别表
  在 masa 目录；Refer-KITTI-V2 在 MFT2025 目录（仅标注）；OVT-B/C-TAO
  不可用（C-TAO 在 .MOTSynth.partial，禁用）。
- 计划：一次 Dance 修复（decision-level cue reliability）→ 四域回归 →
  OVMOT（TAO 官方协议）→ RMOT（Refer-KITTI）→ joint unified checkpoint
  → 关键消融与报告。

## 2026-08-14 — Stage L7：Dance repair 训练 + OVMOT 管线实现

- 设计：UIDM Identity Transition Decoder 内新增 decision-level cue
  experts（motion/geometry/appearance/competition/memory）+ reliability
  router（softmax 权重 × cue score + full-evidence context head）；
  辅助损失 = GT 匹配行 soft-target CE（w=0.1）。第一版用 per-candidate
  BCE 导致 logit 发散（rel≈10），改为 GT 行 soft-target CE 后正常
  （rel≈0.26/frame）。与 COVTrack embedding 级 MCF 明确区分。
- 实验：4 卡 DDP 从 L6 uidm_full 微调 4200 步（lr 1.5e-4，
  teacher 800→0.3，六域同混合），运行中。
- OVMOT 协议：核对官方 TETA（run_ovmot.py）：Base=frequency!=r、
  Novel=r、TETA50 逐类均值；本地已有官方 v1 GT、Detic public dets、
  LVIS v1 CLIP 文本嵌入（与 "a {name}" 模板核对 mean cos 0.9999）。
- 实现：UIDM app_dim 参数化（PBD 2048 / CLIP 512）；TAO val 构建器
  （Detic dets + CLIP crops，批量化 cv2+fp16 后 2 视频 19s vs 旧版
  13min）；官方 TETA 包安装；closed-set CLIP 缓存器；
  `--app-key/--freeze-core` 训练支持。
- 失败/修正：TAO 文件名非标准（ArgoVerse side 相机 9512 帧）导致解析
  崩溃 → 用 stem/frame_index 双规则；检测文件命名 frameXXXX/原名两种。
- 验证：TAO val 完整构建（988 视频 / 36,375 帧 / 1.61M dets，0 空 CLIP）；
  closed-set CLIP 缓存（245 视频 / 428,648 crops，0 空）；OVMOT 评估管线
  端到端冒烟通过（官方 TETA，随机模型 30 视频：All TETA 22.98 /
  LocA 60.77 / AssocA 8.16 / ClsA 0；AVA 子集 Detic 类别本身对不上，
  ClsA=0 是数据特性；关联基线 ~8 为随机水平）。

## 2026-08-14 — Dance repair 结果（一次 iteration，失败）

- 实验：cuerel（cue experts + reliability router，4200 步微调）四域回归。
- 结果：Dance HOTA 0.4888 / AssA 0.2522 / IDSW 9251（L6 为
  0.5546 / 0.3248 / 5290，更差）；BDD AssA 0.5299（L6 0.4866，升）；
  MOT17 0.6973（≈）；MOT20 0.4626（微升）。Macro AssA 0.4855
  vs L6 0.4922。
- 定向回归：关闭 cue-mixture 只留 context head → Dance
  0.5466 / 0.3156 / IDSW 9204。说明伤害不仅来自 mixture，cuerel 核心
  微调本身让 Dance 漂移（IDSW 翻倍）。
- 判断：cue-reliability 作为 Dance 修复机制未达到“显著恢复”，且普通
  MOT 出现严重 IDSW collapse；按“只允许一次主要 iteration”冻结普通
  MOT，改用 L6 uidm_full 作为 OVMOT 共享核心（该核心 Dance 最优）。
- 决策：`L7_DANCE_REPAIR_FAIL`；cuerel 保留为负结果对照；OVMOT 探针
  从 L6 core + 新 CLIP 投影器（freeze-core）重启；普通 MOT 冻结。

## 2026-08-14 — Stage L7：OVMOT 正信号 + 消融 + 收口

- 实验：L6 UIDM core 冻结 + 0.69M CLIP 投影器（仅 closed-set CLIP
  校准，2000 步）→ TAO val 官方 TETA。
- 结果：All TETA 31.48 / AssocA 29.51；Base 29.54 ≈ Novel 29.31
  （随机基线 8.2）；ClsA 0.14（Detic label 差）。
- 归因（同 track_id 只换分类）：Detic ClsA 0.14 → CLIP 文本余弦 7.51
  → GT oracle 96.05；AssocA 恒 29.51。All TETA 31.48→33.94→63.45。
  → WHAT（分类/spec）与 HOW（身份）完全解耦；novel 无偏好。
- closed-set 回归（同一统一 checkpoint，CLIP 前端）：Dance
  0.5369/0.3045/6164、BDD 0.4317/0.4077/11430、MOT17
  0.6471/0.584/369、MOT20 0.5973/0.4196/1799；Macro AssA 0.4290
  （PBD 版 L6 0.4922，-6.3pp，统一语义前端代价）。
- 消融（stateless，训练+推理都无递归记忆 h，保留 anchor/ref 外观）：
  All AssocA 24.32（-5.19pp），Base 24.22 / Novel 25.10。
- 失败/修正：第一版 stateless 只在训练清零 h、推理仍持久 → 无效消融
  （29.81≈full）；改为推理也清零后得到真实差距。Detic SwinB 本地
  推理 roi_align OOM（51.7GB），官方 CDN 网络阻断 → TAO train dets
  未生成，joint OVMOT 训练 NOT_EXECUTED；RMOT 数据（KITTI 帧需登录）
  阻塞。
- 决策：`L7_OVMOT_SUPPORTED / RMOT_NOT_EXECUTED`；报告收口。

## 2026-08-14 — Stage L8：RMOT 协议审计 + Unified Observation + 训练

- 假设：MOT/OVMOT/RMOT 共享同一 identity-dynamics core；差异只在
  WHAT-specification。RMOT 由“语言决定选谁，UIDM 决定身份持续”。
- 数据：Refer-Dance 官方 zip（40 train / 25 test，40 个有 GT 的 query）；
  iKUN 论文官方基线 HOTA 29.06（ByteTrack+NKF）、TransRMOT 9.58。
  本地复用 JDE DanceTrack 图片（symlink）+ L6 PBD cache + L7 CLIP cache，
  不重复解压大图。
- 实现：UnifiedObservationAdapter（CLIP crop + spec → gated sem residue
  注入 UIDM 候选 token，PBD identity 流保留）+ relevance head；
  同一 UIDM core（large，L6 uidm_full 初始化，冻结）；pbd-dropout
  0.15 使同一 core 可处理无 PBD（TAO）的缺失身份证据。
- 失败/修正：relevance logit 阈值 0.0 导致零输出（200 步后 target
  -0.255 vs non-target -0.752，已学出分离但整体偏负）→ 需要 train-set
  F1 校准阈值。TrackEval 硬编码 KITTI 路径、numpy 1.x 别名、seq 长度
  目录层级三处 patch。
- 状态：2400 步四卡 joint（frozen core）训练中；smoke 全链路已验证。

## 2026-08-14 — Stage L8：统一 token 负结果 → identity-pure v2

- 冻结 core 联合训练 2387 步：RMOT 官方 40-query HOTA 34.12（iKUN 基线
  29.06，检测输入不同）；但普通 MOT Macro AssA 0.2798（L7 0.4290，
  L6 0.4922），MOT17/20 因域采样不均几乎塌掉。
- 域均衡采样 + 解冻 core 再训 3000 步（L8-B1）：RMOT HOTA 33.71，
  ordinary Macro AssA 0.2643（更差）。结论：**把 CLIP+spec 语义残差
  直接注入 UIDM 候选 token 破坏 PBD identity**（L7 的 PBD-vs-CLIP
  trade-off 的又一证据）。
- 修正假设（v2）：identity-pure 共享 UIDM（PBD 身份流，spec 不进
  core）+ unified semantic relevance（CLIP+spec 只负责 WHAT/选择），
  WHAT/HOW 解耦；PBD-dropout 让同一 core 可处理无 PBD 的 OVMOT。
- 实现 `--sem-in-core` 开关（默认 off）；hybrid init = L6 core +
  L8 adapter；2500 步四卡训练中。

- **重要更正**：所谓“sem-in-core 破坏身份”是 PBD eval key bug 的伪影
  （eval 误用 coord_mean，训练用 box_end）。修正后 L8-B1（sem-in-core）
  ordinary Macro AssA 0.5087、RMOT HOTA 37.88/AssA 31.02，均优于
  identity-pure v2（0.5045 / 35.20/28.63）。两个变体都是正结果：
  语义可以进 token 流，只要身份证据（pbd_be）正确。

## 2026-08-14 — Stage L8：v2 三任务正信号（PBD eval key 修复后）

- 关键 bug：L8 eval 误用 `pbd`（coord_mean）而 L6 训练/评测用
  `pbd_be`（box_end）→ 所有 L8 数字先作废；修复后 v2 四域 ordinary
  Macro AssA **0.5045**（L6 0.4922 / L7 0.4290）。
- v2 RMOT（40 GT queries，train 校准阈值 -0.1）：HOTA **35.20** /
  DetA 43.42 / AssA 28.63（iKUN 29.06/25.33/33.35，检测器不同）。
- v2 OVMOT（TAO val 官方 TETA，PBD-zero + pbd-dropout）：All
  **34.33** / AssocA **30.44** / ClsA 7.51；Base≈Novel（30.45/30.40）。
  同一 checkpoint 三任务全部为正信号。

## 2026-08-15 — Stage L9：Scaled Specification-Conditioned Unified MOT

### 假设

共享 UIDM identity-dynamics core 可同时支持 closed MOT / OVMOT / RMOT；
在 PBD box-end 身份证据 + CLIP/spec 语义的统一观测空间上，加一个
per-candidate learned gate（`z = z_id + gate*W(sem)`），让语义仅在需要
消歧时调制身份流，避免 L7/L8 的 identity-regression trade-off。

### 已完成

- L9-A：文献/code 审计（OVTR ICLR25、TRACT ICCV25、AED TIP25、QTrack
  2026）；无工作实现“one identity core + one shared ckpt + 三 formulation”。
- L9-B：`tools/cache_l9_tao_pbd.py`（crop 提取，~0.27s/crop，resume，
  write-through）；全图生成方案因 L1B 46% 空帧弃用；批量 vision encode
  尝试后因大 crop OOM/CUDA error 弃用。缓存已启动，`tools/monitor_l9.py`
  自动扩容/重启 worker。
- L9-D：`cond_gated` 实现（init ≡ L8-B1，sigmoid gate≈0.73，W=I），
  L8-B1 large ckpt 加载验证通过（仅新增参数缺失）。
- L9-C：`eval_l8_ovmot.py --pbd-cache` 接入 `pbd_box_end_last`。
- L9-E：`tools/train_l9_uidm.py`（3-way task balance、resume opt/sched、
  周期 ckpt、`--cond-gated`、可选 OVMOT 流）。

### 环境

- 08:45-09:25：SAM3_InterMOT（他人任务）占 ~112GB 主机内存；
  LocateMOT 先 1 worker 后自动扩到 2 worker；不干扰其进程。
- GPU 使用仅 1/2/4/6/7（避开 3/5/9）。

### 缓存性能修复（10:45）

- 现象：meta `seconds` 8-12s 但帧间隔 36s。
- 原因：`_ckpt_hash(MODEL_DIR)` 每帧重算，hash 整个 3B 模型文件
  （~6GB 磁盘读/帧）。
- 修改：启动时算一次并复用 → 帧间隔恢复到 ~13s（2 worker 约
  9.4 帧/分）。已提交。

### 下一步

- TAO val PBD cache 完成后跑 L8-B1/B2 full-observation TETA；
- DLA dets（TAO train）→ CLIP → PBD cache → L9 联合训练；
- 三任务正式评测 + 4 组消融 + novelty audit + 最终报告。

### L9 v1 训练回归（15:30 发现并修复）

- 现象：L9 main 10k steps（cond-gated, init L8-B1）在 ordinary MOT 上
  大幅回归（Dance AssA 0.3457→0.1135，Macro 0.5087→~0.42），RMOT
  AssA 31.02→10.58；L8-B1 对照评测正常（Dance AssA 0.3405）。
- 原因：`sem_transform` 初始化误用 `nn.init.ones_`（全 1 矩阵，退化
  投影），应为 `eye_`（单位阵）。L9 语义残差被压成常向量，破坏核心。
- 修复：`nn.init.eye_(sem_transform.weight)`；v1 权重保留在
  `outputs/l9/checkpoints/uidm_l9_main_v1_failed/` 作为失败证据。
- 重训：v2（6k steps, 2 GPU, 同样 init L8-B1）已启动，计划 3k 步时
  做 dance 快速评估监测是否再次漂移。

### v2/v3 仍回归 → 自训练漂移判定（16:45）

- v2（eye init, unfrozen, init L8-B1 final）：step1000 dance AssA
  0.196（B1 对照 0.3405）→ 即使 init 修复，从已收敛 final ckpt 继续
  自训练仍立即漂移。
- v3（eye init, freeze-core, 只训 adapter/gate）：step1000 dance AssA
  0.175 / IDSW 32477（更差）→ 冻结 core 只改输入流也不稳，说明
  core 对 cand_sem 分布敏感，且输入侧单边适配会 out-of-distribution。
- 判定：UIDM 自训练（teacher=student rollout）从收敛点继续必然漂移；
  “充分训练”不能通过延长已收敛 checkpoint 实现。
- 对策 v4：镜像 L8-B1 成功配方——init `uidm_l8_joint`（未收敛中间点）
  + cond-gated + eye init + 3000 步；与 B1 同轨迹地共同训练 gate 与
  core。已启动（17:22）。

### 真正根因：init 加载 bug（18:40 定位）

- v4/control 依旧高 loss（~110）而手动诊断正常（~7）→ 逐脚本对比
  定位：`train_l9_uidm.py` 的 `--init-ckpt` 手动过滤只匹配裸 key，
  而 `uidm_l8_joint` 的 164 个 key 全部带 `uidm.`/`adapter.` 前缀 →
  **core 从未加载，全程随机初始化**；adapter 加载成功造成假象。
- L8-B1 当年用的是旧版正确 init（其日志 missing=0）。
- 修复：改用 `load_l8_state`（兼容前缀/裸 key）；train_l8_uidm.py
  同步修复。v1-v4 与 control 均为 init bug 产物，不作为 gate 的
  科学负证据（保留为失败证据）。
- v5（修复后，uidm_l8_joint + cond-gated + 3k 步）已启动：
  step10 loss 7.5（对照 v4 的 105），核心恢复正确初始化。

### full-PBD OVMOT 负结果与 crop-PBD 训练适配（08-17）

- TAO val full-PBD 官方 TETA：L8-B2 All TETA 32.22 / AssocA 24.95；
  L8-B1 31.83 / 23.87；L9-v5 32.04 / 24.22 —— 全部低于 PBD-zero
  （34.33 / 30.44）。Base≈Novel 保持，判定为 crop-PBD 观测分布与
  训练分布不匹配，非开放词汇回归。
- 对策：用 L6 105 个 TAO train 视频构建 crop-PBD + CLIP + GT 的
  OVMOT 训练流（DLA dets 因 torchvision roi_align OOM 不可用）；
  4,200 帧 / 7,522 候选 / 86% GT 匹配。缓存完成并合并。
- L9-ovmot 训练（resume v5 + OVMOT 流, 6k 步, 4 GPU）进行中；完成后
  复测 full-PBD OVMOT 验证分布适配假设。
