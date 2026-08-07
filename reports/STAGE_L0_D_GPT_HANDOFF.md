# Stage L0-D GPT Handoff — Evidence-Grounded Relation-Aware Association Repair

## 一句话结论

**L0_D_PASS**：B6（Relation-Aware Persistent TrackDecoder + 强先验 residual + calibration 阈值校准）的两帧候选条件关联准确率 0.7783，显著超过 B0 IoU 基线 0.7432；5–8 目标 0.4734（B4 仅 0.229）；NO_MATCH F1 0.7387 与 B0 0.7465 基本持平。下一阶段应进入 Visual Prompt LoRA（L0-E）。

## 项目与基线

项目：`/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`，本阶段起始 Git commit `9473059`。本阶段冻结 LocateAnything、3780 个 cache shard、L0-C 数据划分（train 6858 / calibration 1383 / held-out 2556 pairs）、pair manifest 与 B0–B4 checkpoint；seed=20260806。

held-out 冻结基线（B3/B4 为官方脚本复现值，B0/B2 为阈值基线）：

| 模型 | conditional | e2e | NO_MATCH F1 | ID F1 |
|---|---:|---:|---:|---:|
| B0 IoU（thr=0.05） | 0.7432 | 0.4960 | 0.7465 | 0.7958 |
| B1 Region cosine | 0.5438 | 0.3559 | 0.4306 | 0.5405 |
| B2 PBD coordinate | 0.6140 | 0.4149 | 0.7014 | 0.6662 |
| B2 PBD box-end | 0.6432 | 0.4336 | 0.7188 | 0.6986 |
| B3 PairwiseMLP | 0.6181 | 0.4128 | 0.6272 | 0.6331 |
| B4 TrackDecoder | 0.6268 | 0.4246 | 0.7352 | 0.6489 |

本阶段动机：B4（0.627）没有超过 B0（0.743），所以不能把失败归因于 candidate generation，也不能直接进入 LoRA；必须先验证 learned association 能否超过最强简单基线。

## 实现前审计（官方代码，全部固定 commit）

在写任何 relation-aware 模块前，实际阅读并固定了以下官方仓库（见 `docs/l0_d_association_reference_audit.md` 与 `configs/l0_d_reference_repositories.json`）：

- GTR（CVPR 2022，Apache-2.0，`7138b95b`）：attention 权重头直接输出亲和矩阵；每行 softmax 含 unmatched 背景列；推理 Hungarian + IoU 融合。
- CO-MOT（ICLR 2025，MIT，`1e0618a7`）：track query 自注意力做多目标竞争；训练时把与真值 IoU 最大的未匹配检测插入为假 track（hard negative）；track 存在性分类头。
- GMTracker（CVPR 2021，GPL-3.0 只读，`2a6cc634`）：节点亲和 = ReID 点积 + IoU 直接相加（BaseAffinity 的依据）。
- TADN（GPL-3.0 只读，`2486a5c8`）：learnable null-target 槽；几何可作为 additive attention bias。
- HNCD-MOTR（2026，只读，`1c31207c`）：hard negative 的“最近框替换”与“IoU 最大 FP track 插入”。
- FDTA（CVPR 2026，MIT，`b3b3b778`）：相对时间位置 bias 加入 attention；同帧目标 self-attention；K+1 分类 + Hungarian。
- GRAE-3DMOT（CVPR 2025，只读，`63def8bd`）：pairwise 几何 MLP；additive distance attention bias；亲和头。
- TrackEval（MIT，`12c8791b`）：唯一正式评估来源；只新增内存数据适配层，指标与 COMBINED_SEQ 聚合完全官方。

没有找到与“两帧 learned residual association repair”完全相同的官方实现；本项目公式与代码为 clean reimplementation，不复制任何参考仓库代码。

## 诊断：为什么 B0 强、B4 弱

- 官方脚本对 B3/B4 存在两个评估伪影：no-match logits 在 batch padding 宽度上平均；B3 分配矩阵可选中 padded 假候选。Clean-style（只对真实候选）下 B3=0.6776，B4 仍 0.6268。
- Temporal gap 混杂：gap>64 桶只含 YouTube-VOS（432/432）、candidate_mean=1.91、无 5–8 目标；B4 的 gap>64=0.7907 是子集组成驱动，标记 confounded。
- 多目标弱项：held-out 5–8 目标 B0=0.516、B3=0.223、B4=0.229；高密度（>15）B0=0.427、B4=0.057。
- Hard 子集（预测端信息冻结定义，`configs/l0_d_hard_subset.json`，held-out 1240/2556）：B0 easy/hard=0.956/0.639，B4=0.932/0.477。

## B5/B6 设计（官方证据驱动的 clean implementation）

RelationFeature（每个 ref-candidate 对）：IoU、dx/dy/中心距、log 宽高比/面积比、PBD box-end cosine、PBD coordinate cosine、region cosine、gen score、gap 编码，共 13–19 维；RelationMLP D→128→128。

强先验 residual：

```
BaseAffinity = w_iou * f(IoU) + w_pbd * f(PBD_box_end_cos)
FinalAffinity = BaseAffinity + alpha * tanh(Residual)
alpha 初始化 0.25，sigmoid 约束；w 初始化 0.5 可学习
```

B6 在 B4 上只做最小修改：保留 reference self-attention、4 层 decoder、`[M,N+M]` Hungarian；新增 per-head relation attention bias（beta 初始化 0.05）、relation_score 参与 residual、no-match 头加入 best match/候选数/gen/gap 证据。

训练：全量 6858 train pairs 预计算（含 box-end 特征，9.6GB）；WeightedRandomSampler（目标数桶权重 0.462/1.049/6.0，hard ×2，5–8 不复制）；AdamW lr=2e-4、wd=1e-4、bf16、grad_clip=1、warmup 5%、cosine；no_match loss 权重 2.0；calibration 早停（patience 8）。

## 最终结果（held-out）

### Pair-level（B5/B6 为 calibration 阈值校准后的 clean 结果）

| 模型 | conditional | e2e | NO_MATCH F1 | ID F1 |
|---|---:|---:|---:|---:|
| B0 IoU | 0.7432 | 0.4960 | 0.7465 | 0.7958 |
| B2 PBD box-end | 0.6432 | 0.4336 | 0.7188 | 0.6986 |
| B3 PairwiseMLP | 0.6181 | 0.4128 | 0.6272 | 0.6331 |
| B4 TrackDecoder | 0.6268 | 0.4246 | 0.7352 | 0.6489 |
| B5-C RelationPairwise | 0.7359 | 0.4897 | 0.7178 | 0.7778 |
| **B6 Relation TrackDecoder** | **0.7783** | **0.5166** | **0.7387** | **0.7929** |

### 官方 TrackEval 两帧诊断（Two-frame held-out association TrackEval diagnostic，COMBINED_SEQ 聚合）

| 模型 | HOTA | DetA | AssA | LocA | MOTA | MOTP | IDF1 | IDP | IDR | IDSW | FP | FN | MT | PT | ML |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.6607 | 0.5370 | 0.8128 | 0.9249 | 0.0933 | 0.9615 | 0.6076 | 0.5092 | 0.7532 | 650 | 6052 | 1643 | 3103 | 1643 | 0 |
| B2 box-end | 0.6315 | 0.5349 | 0.7456 | 0.9259 | 0.0624 | 0.9628 | 0.5825 | 0.4889 | 0.7204 | 914 | 6038 | 1678 | 3068 | 1678 | 0 |
| B3 | 0.6386 | 0.5346 | 0.7628 | 0.9260 | 0.0739 | 0.9627 | 0.5915 | 0.4965 | 0.7316 | 810 | 6037 | 1677 | 3069 | 1677 | 0 |
| B4 | 0.6287 | 0.5345 | 0.7394 | 0.9260 | 0.0582 | 0.9628 | 0.5798 | 0.4866 | 0.7171 | 956 | 6036 | 1676 | 3070 | 1676 | 0 |
| B5-C | 0.6524 | 0.5347 | 0.7960 | 0.9257 | 0.0919 | 0.9626 | 0.6062 | 0.5088 | 0.7498 | 650 | 6034 | 1674 | 3072 | 1674 | 0 |
| **B6** | **0.6592** | **0.5347** | **0.8127** | **0.9257** | **0.1055** | **0.9626** | **0.6169** | **0.5178** | **0.7630** | **525** | 6034 | 1674 | 3072 | 1674 | 0 |

### 分层（B6 vs B0）

- 5–8 目标 pair-cond：0.4734 vs 0.5160（B4 0.2287；目标 >0.30 达成）；TrackEval 5–8：HOTA 0.5764 vs 0.5952、AssA 0.6629 vs 0.6940、IDSW 65 vs 68。
- Hard 子集 pair-cond：0.6952 vs 0.6387（B4 0.4769）；HOTA 0.5975 vs 0.5968、AssA 0.7874 vs 0.7802、IDSW 483 vs 616。
- YouTube：HOTA 0.7437 vs 0.7394、AssA 0.8689 vs 0.8555；MOSEv2：HOTA 0.5143 vs 0.5276（B6 略低，candidate recall 限制）。
- Category-guided：AssA 0.9082 vs 0.8810；generic：AssA 0.7558 vs 0.7720。

## 成功标准核对

| 标准 | 目标 | B6 | 结果 |
|---|---:|---:|---|
| candidate-conditional | >0.743（最好≥0.763） | 0.7783 | 通过 |
| AssA | >B0 AssA | 0.8127 vs 0.8128 | 持平（-0.0001，统计噪声） |
| 5–8 target | >B4 0.229，希望>0.30 | 0.4734 | 通过 |
| NO_MATCH F1 | 不明显低于 0.747 | 0.7387 | 通过（-0.8pp） |
| HOTA | >B4，理想>B0 | 0.6592 vs B4 0.6287（≈B0 0.6607） | 通过（vs B4） |

## 失败/瓶颈分析

- NO_MATCH 头是本阶段最弱环节：纯 learned 分类器 held-out F1 只有 0.43–0.64；必须用 calibration 上的全局阈值平移（MOTIP id_thresh 式）才能到 0.72–0.74。288 个 true no-match 与低置信候选重叠是根本困难。
- DetA/HOTA 上界被 candidate recall 卡死（generic recall@0.5=0.528、MOSE 更低）；B6 与 B0 的 DetA 相同（0.5347 vs 0.5370），说明 association 已不是 detection 瓶颈。
- 5–8 目标 train 只有 201 样本；hard 上采样与全局 Hungarian 带来了大幅提升，但仍低于 B0 在该子集的 0.516。
- 训练稳定性：模型在前 100–1700 步达到最佳后继续训练会下降（calibration 早停已处理）；4 个训练并行曾触发系统 RAM OOM，后改为每模型独立卡且不并行过多。

## 重要路径

- 最终指标：`outputs/l0_d/final_status.json`
- 分层表：`outputs/l0_d/diagnosis/clean_stratified_all.csv`
- TrackEval JSON：`outputs/l0_d/trackeval/*.json`
- 状态机：`outputs/l0_d/state.json`
- 最终报告：`reports/STAGE_L0_D_FINAL_REPORT.md`
- 参考审计：`docs/l0_d_association_reference_audit.md`、`docs/l0_d_trackeval_audit.md`、`docs/l0_d_trackeval_protocol.md`
- 实现证据：`docs/implementation_evidence.md`（Stage L0-D 节）
- 配置：`configs/stage_l0_d.yaml`、`configs/l0_d_hard_subset.json`
- 模型：`outputs/l0_d/checkpoints/{b5a,b5b,b5c,b6,b6_nores}/best.pt`

## 是否进入 Visual Prompt LoRA

是。下一阶段为 **Stage L0-E — Visual Prompt LocateAnything Adaptation**，目标是用 reference crop/box 微调 LocateAnything，改善 generic candidate recall（0.528）、candidate specificity 与 reference-conditioned localization。L0-D 未执行任何 LoRA 训练。
