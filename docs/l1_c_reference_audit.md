# Stage L1-C Reference Audit（Unified Contextual Association）

审计时间：2026-08-09。范围：与 Unified Association Decoder 直接相关的
2024–2026 官方实现（ID prediction / persistent track / multi-cue association /
open-vocabulary MOT）。原则：只记录实际阅读过的官方代码或明确标注
`NO VERIFIED OFFICIAL CODE`；不根据论文结构图或聊天记忆写实现。

## 0. 审计总表

| Method | Year | Venue | Official repo | Verified | Commit | License |
|---|---|---|---|---|---|---|
| MOTIP | 2025 | CVPR | github.com/MCG-NJU/MOTIP | yes | ffc0e905ac196a603027eca8d18fb0dff48c8bcc | Apache-2.0 |
| MOTIP-2 | 2025 | CVPR | github.com/GISer-WB/MOTIP-2 | yes | 012856c1dc13b324064e79339ae71054518d1b5e | MIT |
| FDTA | 2026 | CVPR | github.com/Spongebobbbbbbbb/FDTA | yes | b3b3b778acf93fa4269663b5ea1fd1d5ff8c6730 | MIT |
| HATReID-MOT | 2025/ECCV26 | ECCV 2026 | github.com/HELLORPG/HATReID-MOT | yes | 3eb440c288bdc5e8548a49c43107f6543c74b264 | Apache-2.0 |
| OVTR | 2025 | ICLR | github.com/jinyanglii/OVTR | yes | 500e72c19bf5f7f8717546911a5639fdc26bfee5 | 未提供 LICENSE |
| COVTrack | 2025 | ICCV | HuggingFace clarkqian/COVTrack（本地 masa/COVTrack-main，无 git 元数据） | yes | n/a | Apache-2.0 |
| HNCD-MOTR | 2026 | PR | github.com/zhyzetton/HNCD-MOTR | yes | 1c31207c72f83e6f6b4c867028fe17a980e485a4 | 未提供 LICENSE |
| CO-MOT | 2023 | arXiv | github.com/BingfengYan/CO-MOT | yes | 1e0618a7bb242a611b24e48b0c5ceab682b8f459 | MIT |
| GTR | 2022 | ECCV | github.com/xingyizhou/GTR | yes | 7138b95b5c7951e763af2a3ced15cb29ac8fc9de | MIT |
| UniTrack | 2026 | arXiv | github.com/ostadabbas/UniTrack | yes | afdd9869d31ff115d2fe03b14dd36e0b4f366557 | 未提供 LICENSE |
| OC-SORT | 2023 | CVPR | github.com/noahcao/OC-SORT | yes | 8462e7e729a93ccd3bd995c0a79a890336cb3a0b | MIT |
| MeMOTR | 2023 | ICCV | github.com/MCG-NJU/MeMOTR | yes | 本地 clone | MIT |
| COVTrack++ | 2026 | arXiv | 论文称 code+dataset 将公开 | no | n/a | n/a |
| GOVTrack | 2025/26 | 检索无官方仓库 | n/a | no | n/a | n/a |
| SAM2-OV | 2026 | AAAI | 论文无官方仓库链接 | no | n/a | n/a |

所有本地路径位于 `references/`（OVTR/COVTrack 为读取其他项目目录中的官方副本，
不复制进本项目核心包）。

## 1. MOTIP — Multiple Object Tracking as ID Prediction（CVPR 2025）

- Local: `references/identity_decoding/MOTIP`
- Files inspected: `models/runtime_tracker.py`、`models/motip/id_decoder.py`、
  `models/motip/trajectory_modeling.py`、`models/motip/id_criterion.py`
- Scientific problem: 把关联从 pairwise matching 改为“当前 detection 在既有
  trajectory ID 词表上做 in-context ID prediction”。
- Architecture:
  - TrajectoryModeling：历史 trajectory 编码器（`trajectory_modeling.py`），
    输入 `trajectory_features/boxes/id_labels/times/masks`，形状
    `(B,G,T,N,C)`；以 masked 方式编码每帧每 object 的历史。
  - IDDecoder（`id_decoder.py`）：`feature_dim + id_dim` 拼接后，
    unknown detections 之间 self-attn；unknown 与 trajectory 之间 cross-attn；
    cross-attn 带 `trajectory_times >= unknown_times` 的因果掩码 +
    relative position embedding；每层输出 id logits
    （`embed_to_word` 线性层，词表 = num_id_vocabulary + 1，+1 为 new-ID）。
  - ID embedding：`id_label_to_embed` 用 one-hot → `word_to_embed` 线性层；
    new-ID 用 empty embedding。
- Track representation: 截断历史缓冲（`miss_tolerance-2` 帧）：
  features/boxes/id_labels/times/masks，逐帧追加、mask 表示缺失。
- Candidate representation: detection output embedding（DETR query embedding）。
- Association formulation: ID prediction（词表分类），训练用 focal CE；
  推理用 argmax + 阈值；newborn 用分数阈值过滤；new ID 数量受剩余词表上限约束。
- Semantic cue: category 在检测分类头中；association 本身不显式使用类别。
- Motion cue: trajectory boxes + times 编码（motion-aware trajectory-aware）。
- What we adopt:
  - sequence-local ID 词表思想（不建全数据集固定 ID 分类器）；
  - trajectory buffer 截断 + missing mask；
  - unknown self-attn（candidate 间竞争）；
  - cross-attn 因果时间掩码（只用历史 trajectory 关联当前 candidate）。
- What we do not adopt: 固定大 ID 词表（我们的输入是任意 clip 动态 track set，
  直接对 track set 做 assignment logits，词表大小 = 当前 track 数 + 1）；
  端到端 DETR 联合训练（LocateAnything 冻结）。
- Reason: Unified MOT 的 track ID 是 sequence-local；固定词表无法跨序列复用，
  而“动态 track set 上的 assignment head”是 ID prediction 思想的最小合法等价。

## 2. MOTIP-2（官方后续）

- Local: `references/identity_decoding/MOTIP-2`（commit 012856c1，MIT）。
- 借鉴：ID 一致性约束、history encoding 更新方式（与 MOTIP 相同的
  trajectory buffer 思路）。
- 不采用：整体代码（本项目为冻结 LocateAnything + 独立 UA decoder）。

## 3. FDTA — From Detection to Association（CVPR 2026）

- Local: `references/association_2025_2026/FDTA`
- Files inspected: `models/fdta/id_decoder.py`、
  `models/fdta/Temporal_Adapter.py`、`models/fdta/trajectory_modeling.py`
- Architecture:
  - IDDecoder：unknown features 拼 empty ID embedding，trajectory features
    拼 learned ID embedding；self-attn(unknown) + cross-attn(unknown→trajectory)
    逐层更新；每层 `embed_to_word_layers` 输出 id logits（K+1 词表）。
  - `generate_empty_id_embed` 明确表示 new-ID 槽位 = num_id_vocabulary。
  - 训练时 `shuffle()` 周期性打乱 ID 词表防止顺序偏置。
- Training objective: ID prediction focal CE（`id_criterion.py` 同 MOTIP），
  只监督有效 unknown 位置。
- Inference: Hungarian + 阈值（FDTA 官方推理逻辑）。
- What we adopt:
  - trajectory features + ID embedding 拼接的输入构造；
  - unknown 之间 self-attn + unknown→trajectory cross-attn 的关联方向；
  - 每层输出 assignment logits（我们改为 K+1，K=当前 active tracks）。
- What we do not adopt: 6 层大 temporal adapter；K+1 固定词表；
  end-to-end 检测-关联联合训练。
- Reason: 保留“set-level competition + history cross-attention”的官方证据结构，
  去掉 dataset 无关的检测耦合。

## 4. HATReID-MOT — History-Aware Transformation（ECCV 2026）

- Local: `references/association_2025_2026/HATReID-MOT`
- Files inspected: `HAT-MASA/transform/lda/lda.py`、
  `HAT-MASA/transform/lda/standard_scaler.py`、README
- Architecture: 对每个 trajectory 的历史 ReID features 拟合加权 LDA
  （类 = trajectory ID，`S_b/S_w` 广义特征分解），用投影矩阵把当前特征
  变换到更可分的子空间再 cosine 匹配。是 plug-and-play 特征变换，
  不是端到端 decoder。
- What we adopt: “历史信息改变当前 appearance feature 空间”这一思想
  （我们的 track encoder 用 attention 更新 track query，而非 LDA）。
- What we do not adopt: LDA 本身（每序列在线矩阵分解不便于跨数据集学习，
  且与 set-level 竞争机制不兼容）。
- Reason: 官方代码证明 history-conditioned transformation 有效；
  但我们的目标是可训练的统一 association 规则，LDA 是启发式统计变换。

## 5. OVTR — End-to-End Open-Vocabulary MOT（ICLR 2025）

- Local: `/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/third_party/research_refs_phase4n/OVTR/ovtr`
- Files inspected: `models/updater.py`（QueryInteractionModule /
  Category_Information_Propagator）、`models/ovtr.py`
  （RuntimeTrackerBase / OVFrameMatcher）、`models/tco.py`
- Architecture:
  - Persistent track queries：active tracks 每帧更新 query_pos/query_tgt
    （self-attn：`q=k=query_pos+output_embedding` + FFN），再与 newborn
    slots 拼接进入 decoder。
  - Category_Information_Propagator：active tracks 之间 self-attn 聚合
    category/image embedding，更新 query_tgt；ref_pts 用 pred_boxes 的
    inverse_sigmoid。
  - 关联：track query 自身携带 obj_idxes（sequence-local ID），
    未匹配 slots 用 Hungarian 匹配 newborn GT；推理时 obj_idxes 保持，
    disappear_time >= miss_tolerance 才移除。
  - 训练：random_drop_tracks + fp_ratio 增加 hard negatives；
    Category-agnostic matcher + 每帧 loss。
- What we adopt:
  - “track query 是持久对象、每帧与候选集合整体交互”的 set-level 结构；
  - active track 之间 self-attn（全局竞争）；
  - obj_idxes（sequence-local ID）生命周期管理；
  - miss_tolerance / disappear_time 的短 gap 保留规则。
- What we do not adopt: 端到端 DETR（LocateAnything 冻结）；CLIP 类别分支
  （我们用 LocateAnything 的 semantic/generation evidence）。

## 6. COVTrack — Adaptive Multi-Cue Fusion（ICCV 2025）

- Local: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/COVTrack-main`
  （官方实现，HuggingFace clarkqian/COVTrack 发布权重；目录无 git 元数据）
- Files inspected: `ovtrack/models/roi_heads/ovtrack_roi_head.py`
  （FeatureFusionModule）、`ovtrack/models/trackers/ovsort_tracker.py`
- Architecture:
  - FeatureFusionModule：appearance/bbox/cls 三个 512 维特征分别线性映射，
    门控网络（`Linear(3d→d→2)+Sigmoid`）输出 bbox/cls 门控权重，
    残差层融合；自适应融合比例 `max_fusion_ratio*(1-assoc_conf)^2`，
    assoc_conf 由 cycle consistency 提供。
  - OVSortTracker：Kalman motion + appearance cosine + semantic/category
    特征统一在 association cost 中融合（`motion_weight` 等超参）。
- What we adopt: 多 cue（geometry/semantic/appearance）进同一个关联头的
  设计思想；用可学习门控而不是固定加权（我们的 relation-aware attention
  是它的可学习版本）。
- What we do not adopt: 手写 cost fusion + Kalman 主导；单帧独立 fusion
  （我们保留 track history）。
- Reason: COVTrack 证明 multi-cue fusion 能提升 OV-MOT；但 Unified MOT
  主方法必须是 trainable set-level decoder。

## 7. HNCD-MOTR（2026，Pattern Recognition）

- Local: `references/association_2025_2026/HNCD-MOTR`
- Files inspected: README、models 目录结构（本地 clone）
- 核心：hard-negative confusion-aware denoising，改进 end-to-end tracking
  的局部关联；是 MOTR 系的 local association 增强。
- Adopt: hard-negative 训练策略思想（我们在 clip 采样中保留 top1/top2 接近的
  hard association）。
- Not adopt: 整体 MOTR 结构。

## 8. 经典关联基线（已审计，用于 C0–C1/C4 对照）

- OC-SORT（MIT，commit 8462e7e7）：7 维恒速 Kalman + OCM 第二轮 IoU；
  C1 motion baseline 依据。
- CO-MOT（MIT，commit 1e0618a7）：end-to-end 与 non-end-to-end 之间
  “coopetition label assignment + shadow query”，理解 set 级分配的不平衡问题。
- GTR（MIT，commit 7138b95b）：global tracklet 图关联。
- UniTrack（无 LICENSE，commit afdd9869）：detection/identity/temporal
  一致性的图表示学习；只读不复制。
- MeMOTR（MIT）：long-memory 高可信写入 + EMA；本阶段只保留
  “高可信才更新 track feature”思想，不实现 memory bank。

## 9. 无官方代码（明确记录）

- COVTrack++（arXiv 2026）：论文明确 “code and dataset will be publicly
  available”，检索未发现官方 repo → `NO VERIFIED OFFICIAL CODE`。
- GOVTrack：多次检索未发现官方 repo → `NO VERIFIED OFFICIAL CODE`。
- SAM2-OV（AAAI 2026）：论文页/检索未发现官方 repo → `NO VERIFIED
  OFFICIAL CODE`。

以上三项只作为论文级概念记录，不作为实现依据。

## 10. 结论（Stage L1-C 设计依据）

1. Association 必须是 set-level：candidate 之间 self-attn、candidate→track
   cross-attn、track 之间 self-attn（MOTIP/FDTA/OVTR 三方一致）。
2. ID 是 sequence-local：用动态 track set 上的 assignment logits
   （K+1：existing tracks + NEW），不用全数据集固定 ID 分类器。
3. Track 用短历史 buffer + missing mask + relative time（MOTIP/FDTA）。
4. Multi-cue（PBD appearance / geometry / motion / semantic）必须进入同一
   关联头（COVTrack 门控融合思想），但主方法不用手写加权。
5. NEW / lifecycle 对所有方法共享同一规则（association-controlled 公平性）。
6. Hard negative 采样（HNCD-MOTR/OVTR fp_ratio）进入 clip 构建。
