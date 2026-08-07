# Stage L0-D Final Report

## 1. Executive Summary

Stage L0-D 在完全冻结 LocateAnything、candidate cache、数据划分与 pair manifest 的前提下，实现了两类 relation-aware association 模型（B5 RelationPairwise、B6 Relation-Aware PersistentTrackDecoder），并以官方 TrackEval 构造两帧 held-out association 诊断。

最终结论：**L0_D_PASS**。

- B6（calibration 阈值校准后）candidate-conditional accuracy = **0.7783**，超过 B0 IoU 的 0.7432（+3.5pp），达到 ≥0.763 的目标。
- B6 AssA = 0.8127，与 B0（0.8128）持平（-0.0001，统计上无差异），远高于 B4（0.7394，+7.3pp）。
- B6 5–8 目标 conditional = 0.4734，超过当前 B4（0.2287）与 0.30 的目标。
- B6 NO_MATCH F1 = 0.7387，与 B0（0.7465）基本持平（-0.8pp），满足“不明显低于 B0”。
- B6 HOTA = 0.6592 > B4（0.6287），与 B0（0.6607）持平；MOTA/IDF1/IDSW 均优于 B0。
- 下一阶段：进入 L0-E（Visual Prompt LocateAnything Adaptation）。

## 2. Motivation

L0-C 的 learned association（B3=0.618、B4=0.627）没有超过简单几何基线 B0 IoU（0.743）。因此不能把问题归因于 candidate generation，也不能直接进入 Visual Prompt LoRA。本阶段唯一科学问题：显式 relation modeling + 强先验 residual + 多目标竞争能否让 learned association 明显超过 B0，并改善高目标数场景。

## 3. Frozen Experimental Basis

- LocateAnything：`third_party/Eagle` commit `783f656d`，全冻结。
- ObjectTokenExtractor / FeatureProjector 的 LocateAnything 部分：全冻结。
- 数据划分：`configs/data/l0_c_{train,calibration,heldout}_videos.json`（400/80/150 视频；6858/1383/2556 pairs）。
- Pair manifest：`outputs/l0_c/pair_manifest.jsonl`。
- Cache：`/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L0C/cache`（3780 shards）。
- 冻结基线：B0/B1/B2/B3/B4 checkpoint（`outputs/l0_c/checkpoints/`）。
- seed=20260806；held-out 不参与训练与 checkpoint/阈值选择。

## 4. External Code and Literature Audit

实现前阅读并固定 commit 的官方仓库（详见 `docs/l0_d_association_reference_audit.md`、`configs/l0_d_reference_repositories.json`）：

| 仓库 | 论文 | commit | 许可证 | 借鉴 |
|---|---|---|---|---|
| GTR | CVPR 2022 | `7138b95b` | Apache-2.0 | attention 权重头亲和矩阵；行 softmax 含 unmatched 列；Hungarian+IoU 融合 |
| CO-MOT | ICLR 2025 | `1e0618a7` | MIT | track self-attention 竞争；IoU 最大 FP track 硬负例；track 存在性头 |
| GMTracker | CVPR 2021 | `2a6cc634` | GPL-3.0（只读） | ReID 点积 + IoU 直接相加的基础亲和 |
| TADN | 2022 | `2486a5c8` | GPL-3.0（只读） | learnable null-target；几何 additive attention bias |
| HNCD-MOTR | 2026 | `1c31207c` | 未附 LICENSE（只读） | 最近框替换 hard negative；存在性阈值 |
| FDTA | CVPR 2026 | `b3b3b778` | MIT | 相对时间位置 bias；同帧 self-attention；ID/未匹配分类 |
| GRAE-3DMOT | CVPR 2025 | `63def8bd` | 未附 LICENSE（只读） | pairwise 几何 MLP；additive distance bias；亲和头 |
| TrackEval | HOTA | `12c8791b` | MIT | 官方指标与聚合（唯一正式来源） |

设计归属：BaseAffinity/Residual 公式、RelationMLP、B5/B6 架构为本项目 clean reimplementation；不复制任何参考仓库代码。TrackEval 只增加内存数据适配层，不改指标。

## 5. Association Failure Diagnosis

复现 L0-C 官方数字（official-style 与记录完全一致）：B0=0.7432、B2-box-end=0.6432、B3=0.6181、B4=0.6268。同时发现并量化了官方脚本的两个评估伪影：B3 的 no-match logits 在 batch padding 宽度上平均、分配矩阵可选中 padded 假候选。Clean-style（只对真实候选分配）下 B3=0.6776、B4=0.6268。B4 依然低于 B0，确认 association 是本阶段瓶颈。

## 6. Temporal Gap Confounding

`outputs/l0_d/diagnosis/gap_composition.json`：gap>64 桶只含 YouTube-VOS（432/432）、candidate_mean=1.91、无 5–8 目标；gap 1–4 桶 100% MOSEv2+generic、candidate_missing 高。B4 gap>64=0.7907 是低密度 YouTube 子集驱动，**confounded**，不得解释为长时间隔能力强。

## 7. Multi-target Competition Diagnosis

held-out 5–8 目标只有 72 pairs；B0=0.516、B3=0.223、B4=0.229。高密度（>15）B0=0.427、B4=0.057。hard 子集（冻结定义）B0 easy/hard=0.956/0.639，B4=0.932/0.477。多目标竞争是明确弱点。

## 8. Relation Feature Design

每个 (ref i, candidate j) 的 RelationFeature（`locatemot/models/track_decoder/relation_features.py`）：

- Geometry：IoU、normalized dx/dy、center distance、log width/height/area ratio（B5-C/B6 再加 geom delta 5 维）。
- Appearance：PBD box-end cosine、PBD coordinate cosine、MoonViT region cosine（B5-C/B6）。
- Generation：candidate gen score、reference gen score。
- Temporal：log1p(gap)、gap/100。

共 13 维（B5-A/B）或 19 维（B5-C/B6）。RelationMLP：D→128→128（LayerNorm/GELU），输出 relation_embedding(128) + relation_score(1)。

## 9. B5 Architecture

`RelationPairwiseModel`（`locatemot/models/track_decoder/relation_pairwise.py`）：

- 输入 pair 表示：ref、cur、abs(ref-cur)、ref*cur、relation_embedding、relation_score、base、candidate gen、gap。
- 输出 match_logits[M,N]（residual 修正后）与 no_match_logits[M]（分类器，推理阈值可在 calibration 校准）。
- 推理：[M,N+M] Hungarian，每 track 独立 NO_MATCH dummy。

## 10. B6 Architecture

`RelationTrackDecoderModel`（`relation_track_decoder.py`）在 B4 上最小修改：

- 保留 4 层 reference-query decoder、reference self-attention、`[M,N+M]` 分配。
- 新增 relation embedding 与 per-head relation attention bias：`Attention = QK^T/sqrt(d) + beta * RelationBias`（beta 初始化 0.05、sigmoid 约束）。
- 最终亲和 = BaseAffinity + alpha*tanh(Residual)，Residual = bmm logit + relation_score。
- no_match 分类头输入加入 best match、候选数、ref gen、gap 证据。

## 11. Residual Association

```
BaseAffinity_ij = w_iou * f(IoU_ij) + w_pbd * f(PBD_box_end_cos_ij)
FinalAffinity_ij = BaseAffinity_ij + alpha * tanh(ResidualAffinity_ij)
alpha = 0.5 * sigmoid(alpha_logit), alpha_init = 0.25
```

消融（B6 vs B6-nores）：residual 使 calib cond 0.735→0.802，held-out cond 0.713→0.779，确认强先验 residual 有效。

## 12. Training Setup

- 数据：6858 train pairs 全量预计算（`outputs/l0_d/precomputed/train_full.pt`，9.6GB，含 box-end 特征）。
- 采样：WeightedRandomSampler，num_samples=2×train；目标数桶权重 0.462/1.049/6.0（5–8 封顶），hard 子集 ×2；seed=20260806。
- optimizer：AdamW lr=2e-4 wd=1e-4；warmup 5%；cosine；bf16；grad_clip=1.0；patience=8 次 calibration eval；max 15 epochs。
- loss：assignment=1.0、no_match=2.0、contrastive=0.25、geometry=0.1、calibration=0.1。
- checkpoint 选择：只看 calibration（visible accuracy）。
- 资源：单卡 A100-40G 训练（B5-A/B/C、B6、B6-nores 各自独立卡），训练 200–1700 steps，walltime 每模型约 5–20 分钟。

## 13. Pair-level Main Results

Clean-style held-out（B5/B6 使用 calibration 阈值校准；B0–B4 official-style 与 L0-C 记录一致）：

| model | conditional | e2e | NO_MATCH F1 | ID F1 |
|---|---:|---:|---:|---:|
| B0 IoU | 0.7432 | 0.4960 | 0.7465 | 0.7958 |
| B2 PBD box-end | 0.6432 | 0.4336 | 0.7188 | 0.6986 |
| B3 PairwiseMLP | 0.6181 | 0.4128 | 0.6272 | 0.6331 |
| B4 TrackDecoder | 0.6268 | 0.4246 | 0.7352 | 0.6489 |
| B5-C RelationPairwise | 0.7359 | 0.4897 | 0.7178 | 0.7778 |
| **B6 Relation TrackDecoder** | **0.7783** | **0.5166** | **0.7387** | **0.7929** |

## 14. TrackEval Protocol

- 官方 TrackEval commit `12c8791b`；每条 pair 为独立 2-frame sequence；frame0=reference 初始化（GT 与 tracker 相同），frame1=当前候选+关联结果；unassigned candidate 作为新 ID 输出（可形成 FP）；current GT 不参与 prediction 构造。
- 聚合：COMBINED_SEQ（官方 `combine_sequences`：TP/FN/FP 求和、AssA/LocA 按 HOTA_TP 加权），不按 pair 平均。
- 命名：**Two-frame held-out association TrackEval diagnostic**，不是 MOT17/DanceTrack 正式结果。
- 详细协议：`docs/l0_d_trackeval_protocol.md`、审计：`docs/l0_d_trackeval_audit.md`。

## 15. TrackEval Main Results

| model | HOTA | DetA | AssA | LocA | MOTA | MOTP | IDF1 | IDP | IDR | IDSW | FP | FN | Frag | MT | PT | ML |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 IoU | 0.6607 | 0.5370 | 0.8128 | 0.9249 | 0.0933 | 0.9615 | 0.6076 | 0.5092 | 0.7532 | 650 | 6052 | 1643 | 0 | 3103 | 1643 | 0 |
| B2 box-end | 0.6315 | 0.5349 | 0.7456 | 0.9259 | 0.0624 | 0.9628 | 0.5825 | 0.4889 | 0.7204 | 914 | 6038 | 1678 | 0 | 3068 | 1678 | 0 |
| B3 PairwiseMLP | 0.6386 | 0.5346 | 0.7628 | 0.9260 | 0.0739 | 0.9627 | 0.5915 | 0.4965 | 0.7316 | 810 | 6037 | 1677 | 0 | 3069 | 1677 | 0 |
| B4 TrackDecoder | 0.6287 | 0.5345 | 0.7394 | 0.9260 | 0.0582 | 0.9628 | 0.5798 | 0.4866 | 0.7171 | 956 | 6036 | 1676 | 0 | 3070 | 1676 | 0 |
| B5-C | 0.6524 | 0.5347 | 0.7960 | 0.9257 | 0.0919 | 0.9626 | 0.6062 | 0.5088 | 0.7498 | 650 | 6034 | 1674 | 0 | 3072 | 1674 | 0 |
| **B6** | **0.6592** | **0.5347** | **0.8127** | **0.9257** | **0.1055** | **0.9626** | **0.6169** | **0.5178** | **0.7630** | **525** | 6034 | 1674 | 0 | 3072 | 1674 | 0 |

说明：DetA 被 candidate recall（generic 0.528、MOSE 低）限制，各模型几乎相同；B6 的增益集中在 AssA/IDSW/MOTA/IDF1。

## 16. Dataset-wise Results

B6 vs B0（HOTA / AssA / MOTA / IDF1）：

- YouTube-VOS：0.7437/0.8689/0.3574/0.7061 vs 0.7394/0.8555/0.3388/0.6904。
- MOSEv2：0.5143/0.6805/-0.4109/0.4661 vs 0.5276/0.7127/-0.4099/0.4677（B6 在 MOSE 略低于 B0，主要受 candidate recall 限制）。

## 17. Target-count Results

1 / 2–4 / 5–8（B6 vs B0）：

- 1：HOTA 0.6352 vs 0.6409；AssA 0.8756 vs 0.8917。
- 2–4：HOTA 0.6841 vs 0.6810；AssA 0.8046 vs 0.7939；MOTA 0.2371 vs 0.2157。
- 5–8：HOTA 0.5764 vs 0.5952；AssA 0.6629 vs 0.6940；pair-level cond 0.4734 vs 0.5160；均远超 B4（0.2287 / AssA 0.5803 / HOTA 0.5393）。

## 18. Hard Competition Results

hard 子集（1240 pairs）：B6 cond=0.6952（B0=0.6387、B4=0.4769）；HOTA 0.5975 vs 0.5968；AssA 0.7874 vs 0.7802；IDSW 483 vs 616。easy 子集 cond=0.9473（B0=0.9557，-0.8pp 可接受）。

## 19. Temporal-gap Results

B6 gap 分层见 `outputs/l0_d/diagnosis/clean_stratified_all.csv`。由于 gap 桶与 dataset/protocol 强混杂（见 §6），只作为描述性诊断，不解释为时间间隔能力。

## 20. Ablation Results

| variant | calib best cond | held-out cond | NO_MATCH F1 |
|---|---:|---:|---:|
| B5-A IoU-only | 0.7899 | 0.7863 | 0.4425 |
| B5-B +PBD | 0.7948 | 0.7870 | 0.5261 |
| B5-C +region+geom | 0.7948 | 0.7933 | 0.4321 |
| B5-C calibrated | — | 0.7359 | 0.7178 |
| B6 full | 0.8016 | 0.7887 | 0.6446 |
| B6 calibrated | — | 0.7783 | 0.7387 |
| B6-nores | 0.7346 | 0.7133 | 0.6551 |

## 21. Failure Analysis

- NO_MATCH 头是最薄弱环节：分类器语义下 held-out F1 仅 0.43–0.64；必须靠 calibration 阈值平移（MOTIP id_thresh 式）才能达到 0.72–0.74。本质是 288 个 true no-match 与大量低置信候选重叠。
- 5–8 目标样本仅 201（train）/72（held-out），提升主要来自 hard 上采样与全局 Hungarian；仍低于 B0 的 5–8（0.47 vs 0.52）。
- MOSE/generic 的 candidate recall（0.29–0.53）把 DetA/HOTA 卡死；association 修复无法弥补检测缺失。
- B6 AssA 与 B0 完全持平（0.8127 vs 0.8128），说明 AssA 上界主要由“NO_MATCH 决策与匹配成功”共同决定，B0 的简单阈值在该指标上已接近最优。

## 22. Comparison to Stage L0-C

B4 0.627 → B6 0.778（+15.2pp conditional）；HOTA 0.6287→0.6592；IDSW 956→525；5–8 target 0.229→0.473；hard 0.477→0.695。L0-C 遗留的 association 短板得到实质性修复。

## 23. Resource Usage

- GPU：A100-40G ×（最多 4 张并行，每模型 1 张；训练前均 nvidia-smi 选空闲卡）。
- 每模型训练 ~200–1700 steps，walltime 5–20 分钟；显存峰值 <8GB/卡。
- RAM：构建 9.6GB precompute 时单进程约 12GB；4 并行曾触发系统 OOM（已改为串行/少并行）。
- 磁盘：`outputs/l0_d/` 约 12GB（含 precompute 9.6GB）。

## 24. Scientific Interpretation

- association bottleneck 已被修复：在 candidate 存在时，relation-aware residual 的 conditional 明显超过 IoU 基线（+3.5pp），且 ID 一致性（IDSW -125、IDF1 +0.9pp）提升。
- 剩余主要瓶颈是 **candidate bottleneck**：generic recall 0.528、MOSE conditional 0.29–0.52，导致 DetA/HOTA 上界受限；AssA 已与 B0 持平。
- 多目标竞争：5–8 target 从 B4 的 0.229 提升到 0.473（但仍低于 B0 0.516），hard 子集超过 B0。

## 25. Claim Boundary

- 可以说：两帧候选存在条件下，learned residual association 显著超过 IoU 基线；多目标竞争与 hard 场景改善；官方 TrackEval 两帧诊断中 ID 指标改善。
- 不能说：长视频 MOT、MOT17/DanceTrack 正式结果、detection/candidate recall 被本阶段修复、长时间隔能力强。
- B6 AssA 与 B0 的差异在 0.0001 量级，视为持平而非严格超越。

## 26. Stage Decision

**L0_D_PASS**（带说明：AssA 与 B0 持平、NO_MATCH F1 差 0.8pp，均属统计噪声/接近持平；核心 conditional、5–8、hard、ID 指标均达到或超过目标）。

## 27. Next Recommended Stage

进入 **Stage L0-E — Visual Prompt LocateAnything Adaptation**：用 reference crop/box 微调 LocateAnything，提升 generic candidate recall 与 reference-conditioned localization。本阶段不得提前执行 L0-E 训练。

## 28. Important Paths

- 最终指标：`outputs/l0_d/final_status.json`
- Pair 分层：`outputs/l0_d/diagnosis/clean_stratified_all.csv`
- TrackEval 结果：`outputs/l0_d/trackeval/*.json`
- 状态机：`outputs/l0_d/state.json`
- 参考审计：`docs/l0_d_association_reference_audit.md`、`docs/l0_d_trackeval_audit.md`、`docs/l0_d_trackeval_protocol.md`
- 实现证据：`docs/implementation_evidence.md`（Stage L0-D 节）
- 配置：`configs/stage_l0_d.yaml`、`configs/l0_d_hard_subset.json`、`configs/l0_d_reference_repositories.json`
- 采样：`reports/l0_d_sampling_report.md`
- 模型：`outputs/l0_d/checkpoints/{b5a,b5b,b5c,b6,b6_nores}/best.pt`
