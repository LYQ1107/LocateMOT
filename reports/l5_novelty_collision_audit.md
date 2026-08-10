# Stage L5 — Novelty Collision Audit

日期：2026-08-11。

## 1. 候选创新声明

本项目拟提出的核心声明（按优先级）：

1. **Specification-Conditioned Unified Identity**：同一个 video 在不同
   specification（ALL / category / instance）下，真实物体的 persistent
   identity 不应漂移；不同 spec 可以改变 association evidence 与
   competition，但 identity semantics 必须由 GT trajectory 统一锚定。
2. **GT-Anchored Temporal Identity State**：track 的 identity 不是
   single-frame embedding（L1-B 已证伪），而是因果压缩的 observation
   history state；训练只以 GT identity 为监督锚点，允许输入含历史错误。
3. **Cross-Spec Relation-Structure Consistency**：不要求
   h_ALL == h_SPEC，而要求两个视图在共同 identity 上的 relation matrix
   R(i,j) 一致；两个视图分别对 GT 监督，从而允许 restricted view
   比 ALL view 更正确（不被错误 imitation 拉回）。
4. **One-checkpoint heterogeneous MOT**：DanceTrack / MOT17 / MOT20 /
   BDD100K multi-class 由同一模型覆盖，无 dataset-specific head/router/
   threshold（沿用 L3-U0 的正结果）。

## 2. 撞车检查对象

对 2023–2026 已审计的官方方法逐一检查：

| 方法 | 是否有相同声明 | 结论 |
|---|---|---|
| MOTIP / MOTIP-2 | 无：同一 spec 内 ID prediction，无跨 spec 身份稳定性 | 不撞车 |
| TrackFormer / MOTR / MeMOTR / SambaMOTR / CO-MOT / HNCD-MOTR | 无：track query 生命周期与单 spec association，无 spec restriction 一致性 | 不撞车 |
| CAMELTrack / LG-Track / LLTrack / HATReID-MOT / FDTA | 无：多 cue 关联 / ReID 判别，不研究 spec 子集下的身份语义 | 不撞车 |
| Path Consistency | 部分：多路径一致性监督；但路径=时间子采样，非 spec 诱导子集；无 GT-anchored identity 语义 | 部分重叠（一致性 loss 家族），已区分 |
| SOTFormer | 无：单目标 GT-primed 初始化；无 set competition / 跨 spec | 不撞车 |
| NOOUGAT | 无：online/offline 统一图关联，无 spec 一致性；且无官方代码 | 不撞车（设计近亲已记录） |
| GLEE / OVTR / OVTrack / QTrack / AnyTrack / SAM2MOT / SAM3 / Grounded-SAM2 / TRACT / TempRMOT / TellTrack / EPIPTrack / GOVTrack | 无：prompt 输入接口或 open-vocab，身份一致性只在单一 prompt 内部维持 | 不撞车 |
| NOVA / V²-SAM / DOVTrack | 无：3D autoregressive / cross-view 表示对齐 / 数据效率；无 candidate-subset identity | 不撞车 |
| DecoderTracker / Dual-Path Temporal Decoder / AssociaTR / Gated Temporal Fusion | 无：temporal decoder 或轨迹级输出，无 spec restriction 身份语义 | 不撞车 |
| UniTrack / VICP / UPCL | 无：跨域 ReID / trajectory 平滑，无 spec 子集一致性 | 不撞车（L1-B 已覆盖） |

## 3. 需要明确规避的已有概念名

- 「Path Consistency」：避免使用该名字，改称「cross-spec relation
  consistency」并注明差异（GT-anchored + spec-induced subsets）。
- 「Universal ReID」：明确不是 ReID，L1-B 已证伪。
- 「Temporal Fusion Transformer（TFT）」：已有强同名工作（forecasting），
  不使用该名称；使用「GT-Anchored Temporal Identity State」。
- 「ID Prediction」：MOTIP 已占用；Route B 若实施需用
  「sequence-local dynamic ID prediction」并明确与 MOTIP 的区别
  （GT-anchored + cross-spec shared ID target + 无 dataset-global
  词汇表）。

## 4. 结论

未发现与「specification-restriction invariant identity semantics +
GT-anchored temporal state」直接等价的已公开工作。核心风险点在于
「consistency loss」家族（Path Consistency 等），已通过 GT-anchored
target + relation-structure 形式与之区分。

ICLR-level 可行性评估：

- 问题信号：L4 已证实（restricted evidence 改变 identity dynamics，
  Dance instance P1 AssA 0.8406 vs P0 0.5592）；
- 机制新颖性：spec-conditioned evidence + shared identity semantics
  的分离，在已审计文献中没有直接对应；
- 需要实验证明：一个 checkpoint 在 4+ 异构域上同时改善
  cross-spec drift 与标准 tracking metrics。
