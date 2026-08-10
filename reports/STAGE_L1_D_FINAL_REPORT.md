# Stage L1-D Final Report

项目：LocateMOT（Unified MOT）。日期：2026-08-10。
状态：`L1_D_PARTIAL`（残差修正部分方向有效，不作为统一部署模块；
证据驱动的 L1DK 基座被采用）。

## 1. Executive Summary

Stage L1-D 的目标是把 L1-C 的失败证据（raw PBD 判别强但 full-video
AssA 低；UAF from-scratch 破坏先验；失败模式可预测）转化为一个
evidence-driven 的统一关联方法。我们实现了：

1. **L1DK base**：校准后的 Kalman-motion + IoU + PBD 线性融合基座
   （单一 affinity 矩阵 + Hungarian + 共享阈值）。DanceTrack val
   Association-Controlled：**AssA 0.4165 / IDF1 0.563 / IDSW 2,558**
   （对比 C1 motion 0.4193/0.566/2,916，C3 0.3934/0.5367/2,981）。
2. **EGRA residual**：2 层 set-level transformer + 有界残差 +
   track 级 reliability gate（0.49M 参数，8,360 步，4.2 分钟训练）。
   它在 calibration 上提升 AssA +1.9pp，但在 DanceTrack val 不迁移
   （AssA 0.3993 vs base 0.4165）；跨域方向不一致（MOT20 显著提升、
   BDD/MOT17 下降）。

**阶段结论**：残差修正不是统一的正向机制（`L1_D_PARTIAL`），
不部署为统一模型；采用 L1DK base 作为当前最佳统一关联基座。
LoRA 主线维持 L1-C 结论（`LORA_PBD_DEGRADED`），不用于主线。

## 2. Unified MOT Objective

同一套核心模型/主要参数面向 DanceTrack、MOT17、MOT20、BDD100K
（TAO 缓存缺失，本阶段只记录）。本阶段在 Association-Controlled
协议下评估（相同 boxes/scores/frames，只改 ID），主指标 AssA/IDF1/IDSW，
辅以 HOTA/DetA/Frag。

## 3. L1-A Evidence

- T0–T6 full-video 评估完成：frozen B6 local kernel 与候选解析
  在 AC 下 DetA≈0.947；reactivation（id-kept≈6.3%）失败；
  TrackEval 协议与 per-seq 分析归档
  （`reports/STAGE_L1_A_FINAL_REPORT.md`）。

## 4. L1-B Evidence

- Universal Identity Adapter 失败：`L1_B_IDENTITY_SIGNAL_NOT_SUPPORTED`；
  Road LODO 7,841 identities；两个 LODO 方向均相对 raw PBD 退化。
- 因此不再走 ObjectToken → Universal Identity Vector → cosine。

## 5. L1-C UAF Negative Result

- UAF（frozen LocateAnything + UA decoder，7.9M 参数，50k 步，
  NEW-margin=3.5 校准）DanceTrack val AC：
  **AssA 0.133 / IDF1 0.270 / IDSW 26,804**，低于全部传统基线
  （C0–C4），pilot gate FAIL。
- 根因：K+1 CE 被 NEW 主导；from-scratch assignment 在 easy 区
  （IoU margin≥0.10，96.5% 事件）也低于 IoU（0.856 vs 0.960）。

## 6. Baseline Semantic Mapping Audit

以方法名为主键（`reports/l1_c_baseline_mapping_audit.md`）：

| 编号 | 实际方法 | DanceTrack val AssA / IDSW |
|---|---|---:|
| C0 | IoU Hungarian（thr 0.3） | 0.3899 / 3,554 |
| C1 | OC-SORT 风格 motion（Kalman+OCM） | 0.4193 / 2,916 |
| C2 | raw PBD cosine | 0.1555 / 15,616 |
| C3 | IoU+PBD 0.7/0.3（thr 0.3） | 0.3934 / 2,981 |
| C4 | frozen B6 | 0.1546 / 16,456 |
| UA | UAF（50k，margin 3.5） | 0.1329 / 26,804 |

关键澄清：AssA≈0.419 是 Motion，不是 raw PBD。

## 7. Why Raw PBD Matters

- Frozen PBD candidate selection：same-category R@1=0.922，
  mAP=0.944（DanceTrack calibration）；
- 但 full-video raw-PBD AC AssA=0.155（val）——判别强不等于
  set-level ID 分配强；PBD 必须与 geometry/motion 融合。

## 8. LocateAnything LoRA Engineering Audit

- 根因 1：`generation_trace.py` 对普通 Qwen2ForCausalLM 也有
  `base_model` 自引用导致 `merge_and_unload()` 后死循环 → 仅对
  PeftModelForCausalLM/LoraModel unwrap。
- 根因 2：LoRA JSONL 第一版写成 `(x,y,x,y)` 字面量；官方 `_BOX_RE`
  要求 `<box><x1><y1><x2><y2></box>` → 修正 `tools/build_l1c_lora_data.py`。
- A100 适配：vision eager + sdpa、序列上限 2048、packing_buffer=1、
  video_total_pixels=8192（`docs/official_code_modifications.md`）。
- 结果：300 步 LoRA（train_loss≈1.27），save→load→generate 通过；
  LoRA PBD cache 8,024 帧全部完成。

## 9. Frozen Extractor Equivalence

- 新 extractor vs 旧 extractor（Frozen）：dancetrack0004 frame1
  boxes IoU 0.998、hidden cosine 1.0、两次运行逐位一致 → PASS。
- accepted PBD 只取最终 accepted 路径 box-end hidden；
  rejected MTP partial 不进入特征。

## 10. Frozen PBD vs LoRA PBD

（DanceTrack calibration，8,016 帧，60,805 正样本对）

| 指标 | Frozen PBD | LoRA PBD (300 步) |
|---|---:|---:|
| same-ID cosine | 0.9775 | 0.8900 |
| diff-ID cosine | 0.9244 | 0.8533 |
| Same-Category R@1 | 0.9222 | 0.4349 |
| mAP | 0.9438 | 0.5800 |
| AC AssA / IDSW | 0.140 / 4,190 | 0.042 / 25,414 |

分类：`LORA_PBD_DEGRADED`（PBD 判别与 grounding 双降）；
主线用 Frozen。

## 11. Grounding Forgetting Audit

- LoRA 候选质量：Recall@0.5 0.977→0.803、candidate/frame 8.69→8.53、
  dup rate 0→0.10；DetA（AC）0.944→0.801。
- 结论：短程 LoRA grounding 适配破坏 LocateAnything 原能力，
  不符合 Unified MOT 要求。

## 12. IoU Ambiguity

- 96.5% 事件 IoU margin≥0.10（acc 0.86–0.96）；margin<0.05 仅 1.7%
  （acc 0.40–0.58）。
- IoU 是 DanceTrack 压倒性候选 cue；歧义区是所有方法共同难点。

## 13. PBD Ambiguity

- PBD margin≥0.01 时正确率≥0.91；margin<0.01 只占 9%（正确率 0.83）。
- raw PBD 差 AssA 不是 candidate-selection 问题，而是 full-video
  set-level ID 分配问题。

## 14. PBD-vs-IoU Disagreement Taxonomy

（225,071 events，DanceTrack val）

| 类别 | 数量 | 占比 |
|---|---:|---:|
| both_correct | 201,197 | 89.4% |
| iou_only_correct | 23,827 | 10.6% |
| pbd_only_correct | 25 | 0.01% |
| both_wrong | 22 | 0.01% |

## 15. PBD-Win Cases

- pbd_only 仅 25 例；PBD 不互补于 IoU 的候选选择。
- PBD 的价值在跨帧身份判别（R@1 0.922），不在 IoU 选错时补位。

## 16. IoU-Win Cases

- iou_only 23,827 例（10.6%）：PBD 选错而 IoU 选对——出现在
  同类别高相似候选竞争（margin 低的 PBD 行）。
- motion 在稳定速度/低加速度区有效；PBD 失效集中在
  IoU 歧义/crowding/位置交换。

## 17. Cue Complementarity

- IoU 与 PBD 互补性极弱（pbd_only 0.01%）；motion 与 IoU 互补性
  强（C1/C3 对比，val IDSW 2,916/2,981 vs IoU 3,554）。
- 因此 L1-D base 必须含 motion —— 这是本阶段最重要的数据驱动决策。

## 18. Learnability Probe

（logistic regression，calibration 训练 / val 评估）

| 目标 | AUROC | PR-AUC |
|---|---:|---:|
| PBD selection wrong | 0.933 | 0.632 |
| raw-PBD ID continuity wrong | 0.915 | 0.575 |

可预测 ≠ 可修正：L1-D 实验证明修复后的官方指标不提升（见 §30–34）。

## 19. Evidence-Based Architecture Decision

- 保留强 base affinity（IoU+PBD+motion）→ L1DK base；
- 学习器只做门控有界残差（EGRA），不 from-scratch（UAF 反证）；
- 无 NEW 类（CAMELTrack InfoNCE + 阈值证据）；
- 训练分布 = base 真实在线状态 + GT 监督（与推理一致）。

## 20. 2025–2026 Targeted GitHub Audit

新审计（`docs/l1_d_reference_audit.md`）：

| 仓库 | commit | 关键机制 |
|---|---|---|
| CAMELTrack | 46a74bb | GAFFE set-level；InfoNCE；Hungarian+阈值；hard sampling |
| LG-Track | 432a467 | 定位置信度乘性 cost；分阶段匹配 |
| LLTrack | 2ab7994 | IoU+embedding+角度联合 cost；appearance gate；类别互斥 |

复核 L1-C 已审计方法：MOTIP/FDTA（unknown self-attn + cross-attn，
K+1）、OVTR（persistent queries + Hungarian newborn）、COVTrack
（assoc_conf 门控残差）、HATReID（历史变换）、HNCD-MOTR（hard-negative）
——结论不变：必须 set-level；NEW 与关联解耦。

## 21. Proposed L1-D Architecture

EGRA：pair 特征（19 维：iou/pbd/motion/margin/gen/gap/scale/
anchor-cos 等）→ 2 层 set transformer（cand+track tokens）→
`A_final = A_base + r_i * 0.6*tanh(delta_ij)`。

## 22. Strong Base Affinity

L1DK base：`0.4*IoU(last) + 0.2*PBD_cos + 0.4*IoU(Kalman_pred)`，
thr=0.25（calibration 网格校准，val 未参与）。校准：
AssA 0.4241 / IDSW 512（DanceTrack calibration）。

## 23. Reliability / Residual Mechanism

- track 级 gate r_i：监督标签 = base 行是否身份选错（pos_weight=9）；
- delta 有界 ±0.6，选 calibration 最优 0.3 缩放；
- 保留正则：base-correct 行惩罚 |delta|（λ=0.1）。

## 24. Set-Level Competition

GAFFE 式 cand/track 全集合 self-attention（2 层，4 head），
行/列 CE 同时优化 track→candidate 与 candidate→track 的 ranking。

## 25. Motion / Semantic Cue

- 保留 motion：Kalman 预测框 IoU（训练/推理同公式）；
- 语义：本阶段数据协议全部为 person（DanceTrack/BDD/MOT17/MOT20
  均为 person 协议），无跨类候选，语义互斥未启用；
- 速度方向一致性作为 track 特征（velx/vely）进入模型。

## 26. NEW Handling

与 UAF 不同：不设 NEW 类。Hungarian 后低于共享阈值（0.25）的
候选按统一出生规则成为新 track；阈值只在校准集调整。

## 27. Training Objective

- 主：row-CE + col-CE（assignment ranking）；
- 辅 1：reliability BCE；辅 2：保留正则 |delta|。
- 无多个 metric loss 堆叠。

## 28. Multi-Dataset Training Protocol

DanceTrack calibration + BDD100K train + MOT17 train + MOT20 train
（13,405 帧 / 88,465 有监督事件）；50% easy + 50% hard 帧采样；
同一 checkpoint 评估 4 域。

## 29. Trainable Parameter Count / Compute

0.49M 可训练参数；8,360 步 / 40 epochs / batch 64 / 单卡 A100，
~255 秒（GPU 2；GPU 0/3 被其他项目占用，未共卡）。

## 30. Association-Controlled Main Results

### DanceTrack val（TrackEval official）

| method | HOTA | DetA | AssA | IDF1 | IDSW |
|---|---:|---:|---:|---:|---:|
| C0 IoU | 0.6078 | 0.9473 | 0.3899 | 0.5291 | 3,554 |
| C1 Motion | 0.6301 | 0.9470 | 0.4193 | 0.5660 | 2,916 |
| C2 raw PBD | 0.3836 | 0.9466 | 0.1555 | 0.3188 | 15,616 |
| C3 IoU+PBD | 0.6103 | 0.9469 | 0.3934 | 0.5367 | 2,981 |
| C4 B6 | 0.3827 | 0.9475 | 0.1546 | 0.3083 | 16,456 |
| UA (failed) | 0.3548 | 0.9472 | 0.1329 | 0.2704 | 26,804 |
| **L1DK base** | **0.6280** | 0.9470 | **0.4165** | **0.5630** | **2,558** |
| L1DK_d03 | 0.6149 | 0.9466 | 0.3993 | 0.5503 | 2,579 |

DetA 全方法≈0.947（AC 协议有效，boxes/score/count 一致，hash 校验）。

### 跨域（同一 checkpoint）

| 域 | method | AssA | IDF1 | IDSW |
|---|---|---:|---:|---:|
| BDD100K | L1DK base | 0.3292 | 0.3167 | 12,149 |
| BDD100K | L1DK_d03 | 0.2841 | 0.2889 | 11,151 |
| MOT17 | L1DK base | 0.6010 | 0.5784 | 276 |
| MOT17 | L1DK_d03 | 0.5922 | 0.5775 | 274 |
| MOT20 | L1DK base | 0.2779 | 0.3232 | 3,736 |
| MOT20 | L1DK_d03 | 0.2864 | 0.3916 | 2,408 |

Macro（4 域等权）：AssA base 0.4062 → L1D 0.3905（−1.6pp）；
IDF1 0.4453 → 0.4521（+0.7pp）；IDSW relative 均值 −10.9%。

## 31. Correction Precision

0.898（val）/ 0.886（calibration）——被采纳的修正中 ~90% 是
“帮助保持连续性”。

## 32. Correction Coverage

0.782（val）/ 0.786（calibration）——约 78% 的 base 连续性断裂被修复。

## 33. Helpful vs Harmful Corrections

val：helpful 27,993 / harmful 3,187（8.8:1）；
calibration：5,281 / 679（7.8:1）。

## 34. PBD Preservation Rate

0.983（val）/ 0.989（calibration）——98%+ 的 base-correct 事件保持不变。

## 35. IoU Ambiguity Results

歧义区（margin<0.05）占比 1.7%，所有方法 acc 0.40–0.58；
L1D 在该区无显著提升（连续性修正不改变候选选择）。

## 36. PBD Ambiguity Results

PBD margin<0.01 占 9%（acc 0.83）；修正收益主要来自该区，
但不足以改变官方 AssA。

## 37. Same-Category Competition

DanceTrack 全同类别（semantic-disabled stress test）：base 的身份
正确率仅 44.5%（大量历史 swap），是有界残差无法恢复的主因
（~50% 错误行需 delta>0.6）。

## 38. Multi-Class Semantic Contribution

未启用（本阶段协议为 person）；BDD 亦为 person 协议。

## 39. Crowd / Density

MOT20（crowd）：L1D 显著提升（AssA +0.9pp、IDF1 +6.8pp、
IDSW −35.5%）——residual 在拥挤域是正向的，但方向不统一。

## 40. Motion Regimes

Kalman motion 加入 base 后 val IDSW 2,558（比 C1 低 12.3%）；
残差在稳定运动区主要保持 base（preservation 0.983）。

## 41. DanceTrack

见 §30。L1DK base AssA 0.4165 / IDSW 2,558（AC）。

## 42. MOT17

base AssA 0.6010 / IDSW 276；L1D 基本持平（−0.9pp / −2）。

## 43. MOT20

base AssA 0.2779 / IDSW 3,736；L1D +0.9pp / −35.5% IDSW。

## 44. BDD100K

base AssA 0.3292 / IDSW 12,149（稀疏 5fps 协议）；L1D AssA −4.5pp
但 IDSW −8.2%。

## 45. TAO-compatible

TAO 缓存缺失（4,200 帧 cache 未生成），本阶段不评估；
记录在 `docs/future_rl_reference.md` 与 storage plan。

## 46. Macro Cross-Domain Result

见 §30。base 是更稳定的统一选择；L1D 方向不一致。

## 47. One-Checkpoint Verification

同一 `outputs/l1_d/checkpoints/l1d_k/final.pt` 在 4 域评估；
base 无训练参数（固定权重），同一公式。

## 48. Leave-One-Domain-Out

未执行。任务书 §55 规定 LODO 仅在 pilot 成功后执行；
pilot（residual < base on DanceTrack val）未通过。

## 49. Full Tracker Protocol

未执行（AC 未通过）；AC 是唯一合法归因协议。

## 50. Why Not IoU?

- IoU 单 cue：val AssA 0.3899 / IDSW 3,554；
- 加入 PBD（C3）：0.3934 / 2,981（IDSW −16%）；
- 再加入 Kalman motion（L1DK）：0.4165 / 2,558（IDSW −28% vs IoU）。
- 结论：IoU 是强 cue 但不是全部；motion 是 DanceTrack 上被低估的
  互补 cue。

## 51. Why Not Raw PBD Alone?

- raw PBD AC AssA 0.1555 / IDSW 15,616；
- 判别力强（R@1 0.922）但 set-level ID 分配弱；
- 必须融合 geometry/motion（L1DK 比 raw PBD AssA +26.1pp、
  IDSW −83.6%）。

## 52. Why Did UAF Fail?

K+1 CE 被 NEW 主导；from-scratch 破坏 easy 区先验
（easy 区 acc 0.856 vs IoU 0.960）；50k 步与 margin 校准不能补偿。

## 53. Why Did Universal Identity Adapter Fail?

L1-B 证据：universal identity vector 相对 raw PBD 双方向退化；
identity 信号不支持独立 cosine 关联。

## 54. Does LoRA Help Tracking?

- grounding：下降（Recall@0.5 −17.4pp）；
- PBD 表示：下降（R@1 −48.7pp）；
- association：下降（AC AssA 0.140→0.042）。
- 结论：当前 LoRA 配方不帮助 tracking（`LORA_PBD_DEGRADED`）。

## 55. What Is Actually Unified?

- 统一的是：一套 base affinity 公式 + 一个残差模型（消融）+ 共享
  NEW/lifecycle 规则 + 同一评估协议；
- 本阶段可部署的统一模块是 L1DK base（无训练参数，跨 4 域方向稳定）；
- EGRA 残差不是统一正向模块（MOT20 例外），保持为消融证据。

## 56. Failure Cases

- DanceTrack 位置交换/历史 swap（base 身份正确率 44.5%）；
- BDD 稀疏 5fps（gap 大，Kalman 单步预测与实际帧间隔不匹配）；
- MOT20 高密度遮挡（residual 有效但 gate 校准不能跨域共享）。

## 57. Resource Usage

- 训练：1×A100（GPU 2），0.49M 参数，~4.3 分钟；
- 评估：DanceTrack val 25 视频 ~3 分钟/方法；
- 数据：新增 4 个 raw/sim pkl（~1.5GB）；无新 GPU 占用冲突
  （GPU 0/3 为其他项目，未共用）。

## 58. Scientific Interpretation

1. 可预测的失败不一定可修正（learnability 0.93 → residual 无 val 收益）；
2. 帧间连续性 ≠ TrackEval AssA/IDSW（指标语义差异被实测确认）；
3. 强 base 融合（motion+appearance+geometry）比 learned residual
   更稳健——证据驱动设计的真正收益在 base，不在 decoder。

## 59. Claim Boundary

- 所有数字为 Association-Controlled TrackEval（或注明 custom MOT
  格式）；未做 full tracker、未做 TAO、未做 LODO；
- BDD/MOT17/MOT20 为训练域评估（方向检查），不是 unseen；
- 残差模型的校准仅用 DanceTrack calibration，val 未参与调参。

## 60. Stage Decision

**L1_D_PARTIAL**：
- 正向：L1DK base（采用）；MOT20 上 residual 正向（消融）；
- 负向：DanceTrack val 上 residual 不迁移；BDD/MOT17 AssA 下降；
  方向不一致 → 不部署 residual；
- LoRA 子状态：`LORA_PBD_EXTRACTION_SUPPORTED`（Frozen equivalence
  PASS、8,024 帧 LoRA PBD cache 完成）+ `LORA_PBD_DEGRADED`
  （判别/grounding/association 均下降，未进入 L1-D 主线）。

## 61. Next Recommended Stage

Stage L2：以 L1DK base 为关联主线，进入统一 full-tracker 协议
（同一 checkpoint 输出完整轨迹），并补齐 TAO-compatible cache 后
做多类（person + vehicle）扩展；residual 仅作为 crowd-domain
消融继续研究，不进入统一部署。

## 62. Important Paths

- 模型：`locatemot/models/l1d_association.py`
- 训练：`tools/train_l1d.py`；数据：`tools/build_l1d_dataset.py`
- Checkpoint：`outputs/l1_d/checkpoints/l1d_k/final.pt`
- 评估：`tools/run_l1c_tracker.py` / `tools/run_l1c_trackeval.py` /
  `tools/run_l1d_trackeval.py`
- 审计：`docs/l1_d_reference_audit.md`、
  `docs/l1_d_architecture_evidence.md`
- 结果：`outputs/l1_c/association_controlled_trackeval.json`、
  `outputs/l1_d/ac_bdd.json`、`ac_mot17.json`、`ac_mot20.json`
- 校正审计：`outputs/l1_d/correction_audit_val_k.json`

## 63. Git Commit

本报告对应的提交：`a11241321b462bbedb28e3b24f0bf09eacced7ac`
（提交信息：`Stage L1-D complete: evidence-driven unified association
after PBD and LoRA diagnostics`）。
