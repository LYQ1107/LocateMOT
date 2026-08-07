# Stage L1-A Reference Audit（2026-08-07）

本文件记录 Stage L1-A 实现前实际阅读的 2025–2026 与经典方法官方实现。原则：不根据论文结构图或记忆写复杂算法；每个设计点必须有真实代码证据。克隆/阅读范围只覆盖与 trajectory association、motion、memory、lost/reactivation 直接相关的高价值仓库。

## 1. FDTA — From Detection to Association（CVPR 2026，官方）

- Official repo: https://github.com/Spongebobbbbbbbb/FDTA
- Local path: `references/association_2025_2026/FDTA`
- Commit: `b3b3b778acf93fa4269663b5ea1fd1d5ff8c6730`（2026-03-21）
- License: MIT
- Files inspected:
  - `models/fdta/Temporal_Adapter.py`
  - `models/fdta/trajectory_modeling.py`
  - `models/fdta/id_decoder.py`
  - `models/fdta/fdta.py`
- Core problem: 把连续帧 detection 聚合成带身份判别力的 trajectory embedding，再做 ID 预测。
- Trajectory representation: `History_motion_embedding` = 6 层因果限制 temporal TransformerEncoderLayer + PositionEmbeddingSine + missing-frame mask；`trajectory_modeling.py` 提供 FFN adapter，把 trajectory feature 与 ID embedding 融合。
- Motion model: 时间位置编码 + 因果 transformer 直接建模历史运动；无显式 Kalman。
- Memory design: 轨迹按帧堆叠，T 维 history；missing frame 用 mask 表示。
- Lost/reactivation: 轨迹保留全部历史（由序列长度决定）；IDDecoder 词表含 new-ID 槽；未匹配 detection 分配 new ID。
- Training: 同一帧 unknown detections 间 self-attention（第一层后）；ID 预测用 K+1 词表 focal CE；训练时周期性 shuffle ID 词表防顺序偏置。
- Inference: Hungarian + 阈值。
- What we adopt: 轻量 temporal transformer 思路；relative time 编码；missing-frame mask；检测间竞争。
- What we do not adopt: 6 层大 temporal adapter（本项目 2 层轻量版）；端到端 DETR 联合训练；K+1 ID 词表（我们保留 B6 NO_MATCH 头）。
- Why: Stage L1-A 冻结 LocateAnything 与 B6，只需轻量 trajectory encoder 提供 reference token；不做全套 ID 词表是为了保持 T3–T6 与 T2 的对照可解释。
- Our implementation ownership: clean reimplementation，参考其公开接口与设计思想，未复制代码。

## 2. MOTIP — Motion-Aware Trajectory-aware ID Prediction（CVPR 2025，官方）

- Official repo: https://github.com/MCG-NJU/MOTIP
- Local path: `references/identity_decoding/MOTIP`
- Commit: `ffc0e905`（L0-D 已详细审计）
- License: MIT
- Files inspected: `models/runtime_tracker.py`（trajectory fields、`_update_trajectory_infos`）、`models/traj_model.py`、`models/matcher.py`
- Trajectory representation: `trajectory_features/boxes/id_labels/times/masks` 形状 `(T, N, ...)`；每帧追加当前匹配结果，按 `miss_tolerance`（默认 30）截断保留最近 `miss_tolerance-2` 帧。
- Motion model: trajectory boxes 与时间一起编码；MOTIP 名称即 motion-aware trajectory-aware。
- Memory design: 全历史缓冲 + 截断；track 移除后从缓冲删除。
- Lost/reactivation: miss_tolerance 内允许缺失（mask）；超过后 track 被移出。
- Training: 用 GT 轨迹监督 ID 预测；newborn 用阈值过滤低分新 ID。
- Inference: ID 词表预测 + 阈值。
- What we adopt: 历史缓冲截断思想（K 固定 8 的短窗是 miss_tolerance 的小型化）；时间与 box 一起进 encoder；mask 表示缺失帧。
- What we do not adopt: 30 帧全历史（我们 K=8，保持轻量并让 T3 与 T2 的差异归因于 trajectory 而非容量）；端到端 DETR。
- Why: 规格要求先选合理 K；MOTIP 证据是 30 帧窗口，但我们阶段目标不是 SOTA 而是证明 trajectory context 价值，K=8 足够形成轨迹并控制训练复杂度。

## 3. MOTIP-2（官方后续）

- Local path: `references/identity_decoding/MOTIP-2`，commit `012856c1`（L0-D 已审计）。
- 借鉴：ID 一致性约束、history encoding 更新方式。未复制代码。

## 4. MeMOTR — Long-Term Memory-Augmented Transformer（ICCV 2023，官方）

- Official repo: https://github.com/MCG-NJU/MeMOTR
- Local path: `references/memory_tracking/MeMOTR`
- License: MIT
- Files inspected: `models/query_updater.py`
- Memory design:
  - `long_memory` 每 track 一个向量，`short_memory = short_memory_fusion(query_feat, output_embed)`；
  - memory attention：`q = short_memory + query_pos`，`k = long_memory + query_pos`；
  - 更新：`long_memory = (1-lambda)*long_memory + lambda*output_embed`（EMA），且只有 `is_pos`（高可信匹配）才写；训练阈值 `update_threshold`。
- Lost/reactivation: 未匹配 track 保留 long_memory；用于后续帧再关联。
- What we adopt: 高可信匹配才写 memory（对应 T5 memory write confidence）；EMA 风格更新；anchor token 永久保留（对应 MeMOTR long_memory 的长期性）。
- What we do not adopt: 不把整个 query 更新网络接入 B6（B6 冻结）；不做 multi-head memory attention 大模块。
- Why: T5 只需最小 memory 证明长期信息价值；复杂 memory bank 留到后续阶段。

## 5. OC-SORT — Observation-Centric SORT（CVPR 2023，官方）

- Official repo: https://github.com/noahcao/OC-SORT
- Local path: `references/association_2025_2026/OC-SORT`
- Commit: `8462e7e729a93ccd3bd995c0a79a890336cb3a0b`
- License: MIT（YOLOX 部分 Apache-2.0）
- Files inspected:
  - `trackers/ocsort_tracker/ocsort.py`（KalmanBoxTracker、OCSort.update、OCM 第二轮）
  - `trackers/ocsort_tracker/association.py`（iou/giou/ciou/diou/ct_dist、linear_assignment）
  - `trackers/ocsort_tracker/kalmanfilter.py`
- Motion model: 7 维恒速 Kalman（`x=[x,y,s,r,vx,vy,vs]`），`F/H` 固定；`R[2:,2:]*=10`、`P[4:,4:]*=1000`、`Q[4:,4:]*=0.01`；`convert_bbox_to_z` 用 `[cx,cy,area,aspect]`。
- OCM: 第一轮 IoU(Kalman predict, det) 后，未匹配 track 用 last observation 做第二轮 IoU 关联；速度方向先验（`speed_direction`、inertia）。
- Lifecycle: `time_since_update`、`hit_streak >= min_hits` 才输出；`time_since_update > max_age` 删除；`last_observation` 保留。
- What we adopt: T1 完全按官方逻辑的 clean wrapper（7 维恒速 Kalman + OCM 第二轮）；shared birth shell（unmatched det → tentative → min_hits 确认）与 `max_age` 生命周期。
- What we do not adopt: ByteTrack 低分第二轮（T1 保持纯 OC-SORT 风格，`use_byte=False`）；类别维度（DanceTrack 只有 person）。
- Why: 规格要求 T1 必须来自真实官方逻辑，不能凭记忆写 Kalman。

## 6. MOTR（ECCV 2022，官方）

- Local path: `references/association_transformers/MOTR`
- License: MIT
- Files inspected: `models/query_updater.py`、`models/motr.py`
- 借鉴：track query 传播 + 未匹配 query 保留（lost 语义）；未复制代码。
- 不采纳：端到端检测-跟踪联合训练。

## 7. SORT（2016）

- OC-SORT 内嵌 SORT 的 KalmanBoxTracker 与 lifecycle；T1 与 shared birth shell 的 lifecycle 直接参照该官方逻辑。

## 8. MATR — Motion-Aware Transformer（2025，arXiv:2509.21715）

- Search result: NO VERIFIED OFFICIAL CODE FOUND。
- `vl2g/MATR` 仓库为另一篇论文（Moment Alignment Transformer），不是 arXiv:2509.21715 的官方实现；已明确排除，不写入审计为“官方 MATR”。
- 处理：如使用 movement prediction / query collision 思想，标注 `paper-guided clean implementation`，依据论文公式，不伪装官方。

## 9. D-CTRL detector 来源

- Detector: YOLOX-X，ByteTrack DanceTrack 官方权重（OC-SORT 官方 MODEL_ZOO 明确说明其 DanceTrack YOLOX 权重继承自 ByteTrack）。
- Local weights: `/data3/testdata/vranlee/code_previous/HybridSORT/pretrained/bytetrack_dance_model.pth.tar`
- sha256: `b8d1afba08f801f3fe2cb122faf3fc9af6c7856405a0da94cf91cdd5eb9b3321`
- Engine: OC-SORT 官方 repo（vendored YOLOX），commit `8462e7e7...`，MIT/Apache-2.0。
- 该权重为本地已有、来源为 ByteTrack/HybridSORT 官方 pretrained 目录，License 允许复现使用（YOLOX Apache-2.0）。

## 10. 结论（Stage L1-A 设计依据）

1. TrajectoryEncoder：FDTA temporal adapter 思想（因果时间 transformer + relative time + missing mask）的 2 层轻量版；K=8（MOTIP miss_tolerance=30 的小型化，规格建议）。
2. MotionPredictor：轻量 2 层 MLP 输出 `(dx,dy,dw,dh)`，SmoothL1；paper-guided clean implementation（MATR 无官方代码，不伪装）。
3. Memory：MeMOTR 高可信写入 + EMA/anchor 永久保留；T5 先保守写。
4. Lost/reactivation：MOTIP/MeMOTR/MOTR 共同思想——未匹配 track 保留、gap 后可用相似度再关联；motion 随 gap 降权。
5. T0/T1 生命周期：OC-SORT/SORT 官方逻辑；birth shell 对 T0–T6 完全共享。
