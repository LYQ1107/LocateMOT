# Stage L1-D Architecture Evidence

日期：2026-08-10。目的：把 L1-C 的真实实验结果与官方代码审计映射到
L1-D 结构，防止凭直觉设计。

## 1. 证据 -> 设计约束映射

| # | L1-C 实测证据（真实数字） | 官方代码依据 | 设计约束 |
|---|---|---|---|
| E1 | raw PBD R@1=0.922 / mAP=0.944，但 AC AssA=0.155 / IDSW=15,616 | — | PBD 判别强，但单独做 full-video 关联不足；必须融合几何/运动 |
| E2 | IoU+PBD 固定融合（C3，0.7/0.3，thr 0.3）AssA=0.393 / IDSW=2,981，接近 IoU（0.390/3,554） | LLTrack/LG-Track 的 IoU+embedding cost 融合 | base affinity 采用可校准的线性融合，保留强先验 |
| E3 | Motion（C1）AssA=0.419 / IDSW=2,916 是最强传统基线 | OC-SORT（已审计） | base affinity 加入 motion（预测框 IoU + 速度方向一致性） |
| E4 | UAF（from-scratch K+1 CE，50k 步）AssA=0.133 / IDSW=26,804，低于全部传统基线 | CAMELTrack InfoNCE + 阈值；CAMEL 无 NEW 类 | 学习目标用 assignment ranking（行/列 CE），不设 NEW 类；NEW 由 Hungarian 阈值 + 共享 shell 决定 |
| E5 | 96.5% 事件 IoU margin≥0.10，easy 区 UAF 0.856 < IoU 0.960（from-scratch 破坏先验） | CAMELTrack GAFFE 是 from-scratch embedding（对照） | 学习器输出“有界残差 + 可靠性门控”，在 base affinity 上修正，不替代 base |
| E6 | IoU margin<0.05 歧义区仅 1.7%，所有方法 acc 0.40–0.58 | CAMEL OcclusionSampler / SwapOccluded hard-negative | 训练保留 easy 样本 + 按 margin/冲突加权 hard 采样 |
| E7 | learnability probe：PBD selection wrong AUROC=0.933，ID-continuity wrong AUROC=0.915 | — | reliability gate 可监督（base 行是否选错），用 prediction-side 特征 |
| E8 | both_correct 89.4% / iou_only 10.6% / pbd_only 25 例 | CAMEL GAFFE set-level | 纠错必须主要是 set-level ID 连续性，不是 pairwise cue 二选一 |
| E9 | LoRA PBD degraded（R@1 0.435，AC AssA 0.042） | — | 主线用 Frozen PBD；LoRA 仅作为消融 |
| E10 | BDD/MOT17/MOT20 缓存可用，TAO 缓存缺失 | — | pilot 用 DanceTrack+BDD+MOT17+MOT20；TAO 只留文档 |

## 2. 选定结构：Evidence-Gated Set-Level Residual Association (EGRA)

### 2.1 Base affinity（不可学习，仅校准）

A_base[i,j] = w_i * IoU(last_i, j) + w_p * PBD_cos(ref_i, j)
              + w_m * IoU(motion_pred_i, j)

- w 在 calibration split 上用 AC AssA 校准（小网格，仅此一次）。
- motion_pred = 常速度外推：pred = last_box + (last_box − prev_box) * gap。
  训练与推理使用同一公式（避免 Kalman 的 train/test 不一致）。
- 阈值 thr 只决定 unmatched（NEW），对所有方法共享。

### 2.2 Pair / track / candidate 特征（全部 prediction-side）

- Pair（每 (i,j)）：iou、iou_pred、pbd_cos、center dist（last/pred）、
  log scale、cand gen、gap、log1p(候选数)、行/列 margin
  （iou/pbd/base 各自 top1−top2）、cand size、track age。
- Track：norm box、velocity、gap、age、hits、top1 iou/pbd/base、
  base margin、候选数。
- Candidate：norm box、gen、size、列 top1 iou/pbd、列 base margin。

### 2.3 可训练模块（轻量，~1–2M 参数）

1. TrackContextEncoder / CandidateContextEncoder：MLP 到 d_model=128。
2. Set-level Transformer：candidate tokens + track tokens 拼接，
   full self-attention（2 层，4 head，ffn 512）——GAFFE 式集合竞争。
3. Pair residual head：concat(track_out_i, cand_out_j, pair_feats)
   → MLP → delta_ij = 0.3*tanh(·)，有界修正。
4. Reliability head：track 级 r_i = sigmoid(MLP(track_out_i, row-level
   features))，监督标签 = “base 行 argmax 是否选错”。

A_final[i,j] = A_base[i,j] + r_i * delta_ij

### 2.4 训练目标（主 1 + 辅助 2）

- 主：assignment ranking —— 对每个有 GT 匹配的 track 行做 row-CE，
  对每个 GT track 在 active 集合中的 candidate 列做 col-CE；
  两者平均。不设 NEW 类。
- 辅助 1：reliability BCE（r_i vs base 行选错，pos_weight≈9）。
- 辅助 2：base 行正确时惩罚 |delta|（保留先验，λ=0.1）。

### 2.5 推理

A_final → Hungarian（一对一）→ 共享阈值 thr → unmatched 候选按统一
出生规则成为新 track。所有候选输出（AC 协议），与 L1-C 基线完全一致。

## 3. 训练数据构造（与推理分布一致）

- 用校准后的 base 在训练集上离线模拟完整在线 tracker
  （全部候选输出、unmatched 出生、track 连续、max_age 终止），
  得到每帧 active track 状态（last/prev box、gap、age、pbd ref、
  track id）。
- 每个 track 记录其当前 GT id（出生/匹配时从 manifest `matched`
  反向映射），从而得到逐帧监督：track 的 GT 在当前帧有 match →
  正候选 = matched[gt]；无 → 该行不参与 row-CE。
- 特征与标签全部 prediction-side（GT 只用于离线标签），推理不需要 GT。
- 采样：50% 均匀 + 50% 按 base margin 逆加权（hard 帧），保留 easy。

