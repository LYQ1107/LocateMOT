# Stage L0-D Association Reference Audit

生成时间：2026-08-07

本文件只记录实际阅读过的官方实现。所有 clone 固定 commit，只下载与本阶段（learned multi-object association / relation-aware repair）直接相关的少量高价值仓库。

## 1. GTR — Global Tracking Transformers（CVPR 2022，官方）

- 官方 URL：https://github.com/xingyizhou/GTR
- 本地路径：`references/association_2025_2026/GTR`
- branch/commit：master @ `7138b95b5c7951e763af2a3ced15cb29ac8fc9de`（2022-08-02）
- 许可证：Apache-2.0
- 实际阅读文件：
  - `gtr/modeling/meta_arch/gtr_rcnn.py`（`run_global_tracker`）
  - `gtr/modeling/roi_heads/gtr_roi_heads.py`（`_forward_transformer`、`_activate_asso`、`detr_asso_loss`、`_box_pe`）
  - `gtr/modeling/roi_heads/association_head.py`（`ATTWeightHead`）
  - `gtr/modeling/roi_heads/transformer.py`（decoder layer）
  - `gtr/config.py`（ASSO_HEAD 配置）
- 实际实现要点：
  - query=当前帧 proposals，memory=窗口内所有帧 proposals；decoder 输出后由 `ATTWeightHead`（q/k MLP → bmm）直接生成 `M x N` 关联分数。
  - 归一化：每行拼接一个 0 背景列后 softmax（`_activate_asso`），等价于每个 query 在“全部检测 + 未匹配”上分类。
  - 位置先验：可学习 box PE（xywh 归一化 + 查表插值，`_box_pe`），可选时间 embedding，加在 memory/query 上。
  - 训练标签：按每帧 IoU 给每个 proposal 一个目标实例 ID，未匹配 proposal 指向 background 列；CE loss（`detr_asso_loss`）。
  - 推理：检测级分数按 track 内实例取 max 聚合为轨迹级（`traj_score = asso_nonk @ id_inds`），再与最后框 IoU 取 max（`with_iou`），最后 Hungarian + overlap threshold。
- 借鉴设计：attention 权重头直接输出亲和矩阵；按行 softmax 含 unmatched 列；推理 Hungarian + 几何阈值；box 位置编码先验。
- 未采纳：检测-检测全窗口 Transformer（我们只有两帧）；跨帧全连接 memory。

## 2. CO-MOT（ICLR 2025，官方）

- 官方 URL：https://github.com/BingfengYan/CO-MOT
- 本地路径：`references/association_2025_2026/CO-MOT`
- branch/commit：main @ `1e0618a7bb242a611b24e48b0c5ceab682b8f459`（2025-12-15）
- 许可证：MIT
- 实际阅读文件：
  - `models/qim.py`（QueryInteractionModulev2 等）
  - `models/memory_bank.py`
  - `models/motr.py`（`_post_process_single_image`、`calc_loss_for_track_scores`、`RuntimeTrackerBase`）
  - `models/matcher.py`（HungarianMatcher）
- 实际实现要点：
  - QIM：track query 自注意力（q=k=pos2posemb(ref_pts)+output_embedding，v=output_embedding），只更新高置信 track 的 query/ref_pts；训练时随机 drop track、按最大 IoU 插入 FP track（`_add_fp_tracks`）作为 hard negative。
  - MemoryBank：`save_proj` 保存每 3 帧 embedding；`temporal_attn`（query=当前 output_embedding，key/value=历史 bank）；`track_cls` 线性头输出存在性分数（等价 NO_MATCH 头）。
  - 训练分配：每帧 Hungarian 匹配 GT 后把 ID 写回 track query；未匹配的 track query 当作 FP/消失处理。
- 借鉴设计：track query 自注意力竞争；训练中显式插入与真值 IoU 最大的假 track（hard negative）；memory bank 用保存周期与分数阈值控制可靠性；track 存在性分类头。
- 未采纳：端到端检测-跟踪联合训练；长时 memory bank（Stage L0 不实现）。

## 3. GMTracker（CVPR 2021，官方）

- 官方 URL：https://github.com/haiyang426/GMTracker
- 本地路径：`references/association_2025_2026/GMTracker`
- branch/commit：main @ `2a6cc6343b6f99bfae4edde4d0f58e6c9ca14cdf`（2021-11-30）
- 许可证：GPL-3.0（只读参考，不复制代码）
- 实际阅读文件：`GMMOT/model.py`、`GMMOT/graph_encoder.py`、`utils/build_graphs.py`
- 实际实现要点：
  - 节点亲和：`Mp0 = U_src^T U_tgt + iou`（ReID 点积与 IoU 直接相加）。
  - 跨图消息传递：`emb1 + lambda * Mp0 @ emb2`，再经线性层与归一化。
  - 边亲和（二阶图匹配）与 QP 可微分配。
- 借鉴设计：外观相似度与几何 IoU 直接组合成基础亲和（支持 BaseAffinity 公式）。
- 未采纳：Kronecker 边匹配与 QP 分配（成本高、非本问题必要）；GPL 代码不复制。

## 4. TADN — Transformer-based Assignment Decision Network（官方）

- 官方 URL：https://github.com/psaltaath/tadn-mot
- 本地路径：`references/association_2025_2026/tadn-mot`
- branch/commit：main @ `2486a5c8d94706f50d2af4f035bc97178609b9ca`（2024-02-21）
- 许可证：GPL-3.0（只读参考，不复制代码）
- 实际阅读文件：`tadn/components/transformer.py`、`tadn/mot/managers.py`、`tadn/mot/metrics.py`
- 实际实现要点：
  - 双流 Transformer：targets（含 learnable null-target embedding）与 detections 各自编码，输出后 `sdp similarity = K^T Q / sqrt(d)`，行 softmax（detection → target+null）。
  - 空间嵌入：4 维 xyxy → MLP（Tanh）与外观嵌入拼接；或把 IoU 类指标（`pairwise_ulbr1_metric`）作为 additive attention bias（`memory_mask`）。
  - 分配：先按行 argmax 判断 detection 是否属于 null-target（未匹配 detection 用于新生），再对剩余矩阵做 Hungarian。
- 借鉴设计：显式 null-target 槽；几何可作为 additive attention bias；softmax 行分类 + Hungarian。
- 未采纳：detection→target 方向（我们保留 reference→candidate）；双流编码。

## 5. HNCD-MOTR — Hard Negative Confusion-aware Denoising（2026，官方）

- 官方 URL：https://github.com/zhyzetton/HNCD-MOTR
- 本地路径：`references/association_2025_2026/HNCD-MOTR`
- branch/commit：main @ `1c31207c72f83e6f6b4c867028fe17a980e485a4`（2026-05-31）
- 许可证：README 未附 LICENSE（只读参考，不复制代码）
- 实际阅读文件：`models/cdn.py`、`models/hncd.py`、`models/criterion.py`、`models/query_updater.py`、`models/runtime_tracker.py`
- 实际实现要点：
  - Contrastive Denoising：每个 GT 生成正 query（加噪声）与负 query；负 query 的框被替换为“最近 GT 框”（center distance 最近或 k-NN，`cdn_negative_source`），制造混淆负样本。
  - QueryUpdater 训练时按 `fp_insert_ratio` 把与 active track IoU 最大的未匹配检测插入为 FP track。
  - 推理 RuntimeTracker：track score 低于阈值 → 删除 track（等价 NO_MATCH）。
- 借鉴设计：hard-negative 构造的具体操作（最近框替换、IoU 最大 FP track 插入）；存在性阈值。
- 未采纳：去噪 query 注入 Transformer 的训练技巧（Stage L0 两帧数据不需要 CDN 序列布局）。

## 6. FDTA — From Detection to Association（CVPR 2026，官方）

- 官方 URL：https://github.com/Spongebobbbbbbbb/FDTA
- 本地路径：`references/association_2025_2026/FDTA`
- branch/commit：main @ `b3b3b778acf93fa4269663b5ea1fd1d5ff8c6730`（2026-03-21）
- 许可证：MIT
- 实际阅读文件：`models/fdta/id_decoder.py`、`models/fdta/trajectory_modeling.py`、`models/fdta/id_criterion.py`、`models/fdta/fdta.py`
- 实际实现要点：
  - IDDecoder：trajectory feature 与 one-hot ID embedding 拼接（feature+id 联合 token）；unknown detections 为 query、trajectory 为 key/value 的 cross-attention；`cross_attn_mask = traj_time >= curr_time` 因果掩码；可学习 relative position embedding（按时间差查表）加到 attention bias。
  - 同帧 unknown 之间自注意力（第一层之后），实现检测间竞争。
  - ID 预测：`embed_to_word` 在 K+1 词表（K 个已知 ID + 1 个 new-ID）上分类，focal CE 损失；推理用 Hungarian + 阈值。
  - `shuffle()`：周期性打乱 ID 词表，防止学习到顺序偏置。
- 借鉴设计：时间相对位置偏置作为 attention bias（我们的 RelationBias 时间编码依据）；unknown 间 self-attention 竞争；ID/NO_MATCH 分类头。
- 未采纳：词表式 ID 预测（Stage L0 的 track 是两帧内固定的，无需全局 ID 词表）。

## 7. GRAE-3DMOT（CVPR 2025，官方）

- 官方 URL：https://github.com/altkddhfcjs/GRAE-3DMOT
- 本地路径：`references/association_2025_2026/GRAE-3DMOT`
- branch/commit：main @ `63def8bde5e199a4e77fdf4fab76a4b3511fe132`（2025-12-22）
- 许可证：README 未附 LICENSE（只读参考，不复制代码）
- 实际阅读文件：`models/main.py`、`models/utils/transformer.py`、`models/utils/cross_attention.py`
- 实际实现要点：
  - 几何关系编码：pairwise spatial features（3D 中 19 维）经 MLP → relation feature；DFFL 用 distance 调制 gamma/beta（初始 1/0）。
  - 时间关系编码：pairwise temporal features + `attn_dist = -temporal_dist` 直接加到 attention score（additive distance bias）。
  - 亲和头：decoder 每层由 relation-aware feature 输出 aff_score；下一层 attention bias = `tau * (-aff_score)`（亲和门控注意力）。
- 借鉴设计：pairwise 几何 MLP 编码 relation；additive distance bias；relation-aware affinity 头。
- 未采纳：3D 运动学特征与门控迭代。

## 8. 既有参考复查（MOTIP / MOTIP-2 / TrackFormer / MOTR / MeMOTR）

- MOTIP `ffc0e905`：ID 词表 + trajectory cross-attention + Hungarian 扩展 newborn 列；推理 `id_thresh`。
- MOTIP-2 `012856c1`：`linear_sum_assignment(1 - extended_id_confs)`。
- TrackFormer `e468bf15`：track query 跨帧；训练 `track_query_false_positive_prob` 加入未匹配框作为 FP track query（hard negative）。
- MOTR `8690da33`：track-disappear（matched_gt_idxes=-1）与 score 阈值管理。
- MeMOTR `eb7a177b`：long_memory 每 track 保存历史 embedding，query updater 中 memory attention + EMA 更新（仅高置信 track）。
- 这些在 Stage L0-C 已记录于 `docs/implementation_evidence.md`；L0-D 只借鉴其中与本阶段两帧关联接口相关的部分。

## 结论

没有找到与“两帧 learned residual association repair”完全相同的官方实现；本阶段的设计由以下官方证据组合而成：

- BaseAffinity = 外观 + 几何直接相加（GMTracker `Mp0 = U^T U + iou`；GTR `traj_score = max(asso, IoU)`）。
- 显式 pairwise 几何 MLP（GRAE spatial_proj；TADN spatial embedding）。
- additive attention bias（TADN memory_mask；GRAE `attn_dist`；FDTA relative temporal PE）。
- query 自注意力做多目标竞争（CO-MOT QIM；FDTA self-attn）。
- 显式 unmatched/null 槽 + Hungarian（GTR background 列；TADN null-target；MOTIP newborn 列；本项目 [M,N+M]）。
- hard negative：训练时插入与真值 IoU 最大的假 track/框（CO-MOT `_add_fp_tracks`；TrackFormer FP track；HNCD 最近框替换）。

其余为 clean reimplementation（本项目自己编写，不复制任何参考仓库代码）。
