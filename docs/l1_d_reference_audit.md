# Stage L1-D Reference Audit

日期：2026-08-10。范围：在 L1-C 结论（raw PBD 判别强但 full-video
AssA 低、UAF from-scratch K+1 CE 破坏强先验、失败模式可预测）之后，
针对“evidence-driven unified association”定向检索并实际阅读的官方实现。

原则：只记录实际阅读过的代码；不根据论文结构图或摘要写实现依据。

## 0. 新审计总表（Stage L1-D）

| Method | Year/Venue | Official repo | Local path | Commit | License |
|---|---|---|---|---|---|
| CAMELTrack | 2025 arXiv 2505.01257 | github.com/TrackingLaboratory/CAMELTrack | `references/l1_d/CAMELTrack` | 46a74bb | 见仓库 LICENSE |
| LG-Track | 2023 arXiv 2309.09765 | github.com/mengting2023/LG-Track | `references/l1_d/LG-Track` | 432a467 | MIT |
| LLTrack | 2025/26 | github.com/ljc4336/LLTrack（README 引用 holmescao/LLTrack） | `references/l1_d/LLTrack` | 2ab7994 | 见仓库 LICENSE |

clone 日期：2026-08-10（本阶段开始时）。

## 1. CAMELTrack — Context-Aware Multi-cue Exploitation

### 1.1 关键接口与结构（实际阅读）

- `cameltrack/camel.py`：CAMEL LightningModule。
  - 每个 cue 一个 `TemporalEncoder`（`temp_encs`，ModuleDict），
    每个 encoder 有独立 `det_tokenizer`（BBoxLinProj / KeypointsLinProj /
    PartsEmbeddingsLinProj）。
  - 前向：`tokenize -> merge -> gaffe -> similarity`。
  - `compute_loss`：对每个 batch 内 track/det 的有效 embedding 拼接，
    以 track target（GT ID）为 label，用 `NTXentLoss`（InfoNCE，
    cosine distance）训练。**不是 K+1 CE**。
  - 推理：`predict_step` 用相似度矩阵 + 关联策略（默认
    `hungarian_algorithm`）+ 阈值 `sim_threshold`。
- `architecture/gaffe.py`：GAFFE = 共享 TransformerEncoder，
  输入 `[det tokens, track tokens, cls]`，自注意力同时看到候选集合与
  轨迹集合（set-level 全局竞争），输出各自 embedding。
- `architecture/temporal_encoder.py`：每个 tracklet 的历史 observations
  经 tokenizer → positional encoding（按 age）→ TransformerEncoder →
  CLS token → linear_out，得到 track 级 token。
- `utils/assignment_strats.py`：Hungarian 先做一对一，再按阈值把
  低于阈值的配对剔除（回归 unmatched），`sim_threshold` 是唯一
  “NEW/不匹配”门。
- `utils/similarity_metrics.py`：norm_euclidean / cosine / IoU 等，
  可对不同 token 类型用各自默认距离再平均。
- `train/dataset.py` + `train/sampler.py`：
  - 训练数据来自预先跑好的 tracker states（每帧检测 + 轨迹），
    每个 sample 是“某条 GT 轨迹在某一帧”；同一视频同一帧取
    `num_samples` 条轨迹组成 batch；`track_targets/det_targets` 是
    GT track id，NaN 表示无效。
  - 采样器：`CAMELSampler`（随机帧）、`OcclusionSampler`（按遮挡
    密度加权）、`GapSampler`（按 missing 帧加权）——hard-case 采样
    同时保留普通样本。
  - `transforms/tracklet.py`：DropoutFeatures（每轨迹最多丢一个 cue，
    p=0.2）、DropoutSporadic（按 age 高斯丢历史观测）、
    SwapSporadic/SwapOccluded（交换 occluded 轨迹的历史 ID，
    增加 hard negative）、DropDets、MaxAge。
- `cameltrack.py`：tracker 生命周期（init→active→lost→dead，
  max_wo_hits=150，min_num_hits 可设 0）；未匹配检测按
  `min_init_det_conf` 出生；训练配置 `cameltrack_train.yaml`：
  state 文件 + `CAMELDataModule`，多数据集训练有开关。

### 1.2 与本阶段直接相关的机制

1. **Association-centric training**：不学检测，只学
   track↔detection 的匹配，且用 InfoNCE（对比轨迹 ID），避免
   “NEW 类主导”问题——这正是 L1-C UAF 失败点。
2. **Set-level 竞争**：GAFFE 让每个候选看到所有轨迹、每条轨迹看到
   所有候选，再输出相似度，替代手写 cost 融合。
3. **单一匹配阈值**：Hungarian 后按相似度阈值产生 unmatched，
   NEW 与匹配解耦。
4. **Hard negative 构造**：SwapSporadic/SwapOccluded 在训练时
   主动制造“外观相似但 ID 不同”的轨迹历史。

### 1.3 采用 / 不采用

- 采用思想：InfoNCE/行内 CE 式 assignment ranking（不设 NEW 类）；
  set-level 竞争；训练时保留易样本 + hard 采样；Hungarian +
  共享阈值。
- 不采用：从零学习 embedding 替换强 base affinity（L1-C 证明会破坏
  raw PBD/IoU 先验）；不引入 keypoints/part 等本协议没有的 cue。
- 本项目形态：**保留强 base affinity（IoU/PBD/motion），只学习
  set-level residual + reliability gate**；不是 CAMEL 的 from-scratch
  embedding。

## 2. LG-Track — Localization-Confidence-Guided Association

### 2.1 实际阅读

- `tracker/matching.py`：
  - `iou_distance(..., pos=True)`：IoU cost 乘以 detection
    localization confidence `pos`（det.pos）。
  - `embedding_distance(..., score=True)`：ReID 距离乘以 detection
    score。
  - `fuse_motion`：Mahalanobis gating（chi2 阈值）+ `lambda_` 融合
    appearance 与 gating distance。
  - `fuse_iou`：`reid_sim * (1+iou_sim)/2`。
  - `fuse_score`：`iou_sim * det_score`。
- `tracker/LG_Track.py`：多阶段匹配，根据检测的
  classification confidence 与 localization confidence 落在不同
  阈值区间（`match_thresh_a..d`、`new_track_thresh`）选择不同 cost
  矩阵与匹配轮次；低 localization confidence 的框用不同匹配策略。

### 2.2 与本阶段相关的机制

1. **Confidence-aware cost**：detection 的定位/分类置信度不是过滤
   而是进入 cost（乘性门控），避免“低分框直接丢弃”。
2. **分阶段匹配**：高置信先匹配、低置信再匹配，保持一对一。

### 2.3 采用 / 不采用

- 采用：把 candidate gen score / localization 置信度作为 pair 特征
  与门控输入（乘性/可学习），不直接丢弃低分候选（AC 协议本来就要求
  全部输出）。
- 不采用：手写阈值分段匹配；固定 Kalman+ReID cost 融合。

## 3. LLTrack — Locality-Aware Multi-Stage Association

### 3.1 实际阅读

- `trackers/ocsort_embedding/association_yolo.py`：
  - 多 cue cost：`iou_matrix + angle_diff_cost + emb_cost`，
    `final_cost = -(iou + angle_diff + emb)`，再
    `linear_assignment`（lap）。
  - `associate()`：第一轮 appearance+motion 同时参与；`two_round_off`
    控制是否先用 appearance gate（`iou_matrix *= cost_matrix`）再做
    motion 轮；第二轮/第三轮只对 unmatched 用 motion。
  - `_nn_res_recons_cosine_distance`：embedding 的 mutual-reconstruction
    affinity（aff_td/aff_dt softmax 重建再求 cosine）作为增强相似度。
  - `filter_pairs(cost_matrix, gate)`：appearance gate 阈值。
  - `associate_kitti`：类别不一致直接 cost=-1e6（semantic hard
    exclusion）。
- `trackers/ocsort_embedding/ocsort.py`：OC-SORT 式 Kalman +
  velocity direction consistency（`speed_direction`、`diff_angle`），
  轨迹观测缓冲 + `k_previous_obs`；`update()` 内分多轮
  （appearance 轮、motion 轮、motion third）。
- `lmf.py`：Focal + LDAM loss 是检测分类损失，不是关联损失
  （避免误引）。

### 3.2 与本阶段相关的机制

1. **多轮关联**：先“外观+运动”联合，unmatched 再纯运动轮；
   每轮匈牙利一对一。
2. **Semantic hard exclusion**：类别不同直接禁止配对
   （cost=-1e6）——BDD/TAO 多类场景可用。
3. **Velocity direction consistency**：轨迹运动方向与检测方向夹角
   进入 cost，作为 motion cue 的补充。

### 3.3 采用 / 不采用

- 采用：类别互斥（有类别标注时）作为 pair 特征/掩码；运动一致性
  （速度方向、预测框 IoU）作为 base affinity 的组成；多轮关联思想
  简化为“一个 final affinity 矩阵 + Hungarian”。
- 不采用：完整 OC-SORT 状态机；ReID 重建距离（PBD 已提供外观
  表示）。

## 4. 对既有 L1-C 已审计方法的定向复核（L1-D 视角）

针对 L1-D 需要回答的四个问题，复核 L1-C audit 中已读代码的结论：

| 问题 | 方法 | 复核结论（基于已读官方代码） |
|---|---|---|
| confidence gating | LG-Track/COVTrack | LG-Track 用置信度乘性进入 cost；COVTrack 用 cycle-consistency assoc_conf 控制残差融合比例 → 支持“可靠性门控”。 |
| residual correction | COVTrack | FeatureFusionModule 输出门控权重（Sigmoid）+ 残差融合；但 COVTrack 是特征融合，不是 affinity 残差。本项目 affinity 残差 + 门控是等价的最小可验证形式。 |
| set-level ID assignment | MOTIP/FDTA/OVTR/CAMELTrack | MOTIP/FDTA：unknown self-attn + cross-attn→trajectory，K+1 词表；OVTR：persistent queries + Hungarian newborn；CAMELTrack：GAFFE 全集合交互 + InfoNCE。三方一致：**必须 set-level**。 |
| NEW 解耦 | CAMELTrack / 传统 tracker | CAMEL 无 NEW 类，匹配阈值产生 unmatched；UAF 的 K+1 NEW 类被证明偏置。→ L1-D 采用“匈牙利 + 阈值”，NEW 由 shared shell 决定。 |
| history-conditioned | HATReID-MOT / MOTIP trajectory buffer | HATReID 用历史特征变换当前特征；MOTIP 用截断历史 buffer + times/masks。→ L1-D 用最近历史 box/PBD 做 track 特征（短窗口，不做 long memory）。 |
| hard-negative training | HNCD-MOTR / OVTR fp_ratio / CAMEL Swap | 都证明 hard negative 训练必要。→ 训练数据按 IoU/PBD margin 低、多对一冲突采样，同时保留 easy 样本。 |

## 5. 审计结论（决定 L1-D 结构约束）

1. 主学习目标必须是 **assignment ranking / contrastive**，不能是
   K+1 CE 含 NEW 类（UAF 失败 + CAMEL InfoNCE 证据）。
2. 模型必须能看到 **候选×轨迹的完整集合**（GAFFE 式 set-level
   interaction），不能只做 pairwise MLP。
3. 强 base affinity（IoU/PBD/motion 融合）必须保留；
   学习器输出 **有界残差 + 可靠性门控**，避免 from-scratch 扰动
   （UAF/CAMEL 对比证明）。
4. NEW/生命周期与关联解耦：匈牙利后按共享阈值判 unmatched，出生规则
   对所有方法一致。
5. 可加入语义互斥与运动方向一致性作为 pair 特征，但只在真实标注
   可用时启用。

