# 参考仓库清单（Reference Repository Inventory）

生成时间：2026-08-06

检索原则：只下载与本阶段（Stage L0）直接相关的少量高价值官方项目；每个仓库独立目录；固定 commit；不把第三方仓库复制进 `locatemot` 核心包；不直接运行来源不明脚本；不下载来源不明权重。

## 1. Eagle / LocateAnything（官方基准确认）

- 项目名称：Eagle（LocateAnything 位于 `Embodied/`）
- 对应论文：LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding（arXiv:2605.27365）
- 官方 URL：https://github.com/NVlabs/Eagle
- 本地路径：`/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/third_party/Eagle`
- 别名路径：`references/locateanything -> ../../third_party/Eagle`
- clone 日期：2026-08-06
- branch：main
- commit SHA：`783f656d127ee498137b5ff52603ce36c292d317`
- 许可证：仓库代码 Apache-2.0（`LICENSE`）；模型权重 NVIDIA non-commercial license（`Embodied/LICENSE_MODEL`）
- 参考的具体文件：
  - `Embodied/README.md`
  - `Embodied/locateanything_worker.py`
  - `Embodied/eaglevl/utils/locany/modeling_locateanything.py`（自定义 `generate`）
  - `Embodied/eaglevl/utils/locany/generate_utils.py`（PBD 解码）
  - `Embodied/eaglevl/model/locany/modeling_locateanything.py`（训练 forward）
  - `Embodied/eaglevl/model/locany/configuration_locateanything.py`
  - `Embodied/eaglevl/model/locany/modeling_qwen2.py`、`eaglevl/utils/locany/modeling_qwen2.py`
  - `Embodied/eaglevl/train/locany_finetune_magi_stream.py`
  - `Embodied/eaglevl/train/tools.py`（`apply_visual_prompt_to_sample`）
  - `Embodied/eaglevl/train/dataset.py`
  - `Embodied/document/TRAINING.md`、`DATA_PREPARATION.md`、`RESULTS.md`
  - `Embodied/shell/locate-anything-lora-visual-prompt.sh`
  - `Embodied/LICENSE_MODEL`
- 参考的具体模块：LocateAnythingForConditionalGeneration、PBD generate（MTP/NTP/Hybrid）、decode_bbox_avg/decode_ref/handle_pattern、block=6 注意力、visual prompt LoRA 数据转换、worker API。
- 最终是否复用代码：不直接修改官方仓库；以 wrapper / forward hook / subclass 方式接入。
- 若复用，复用了哪些部分：官方输出 token 约定（`<ref>`、`<box>`、`<0>..<1000>`、`<null>`、`<box>none</box>`）；worker 的 prompt 模板；visual_prompt=true 的训练数据格式；LoRA 超参默认（LLM rank 64、MoonViT 冻结、MLP 可训练）。
- 若未复用，借鉴了什么设计：PBD 六 token block 与 hidden-state 的对应关系（见 `docs/pbd_token_mapping.md`）；Fast/Hybrid/Slow 三种解码路径；visual prompt 通过 `<image-2>` 追加的方式。

## 2. MOTIP（CVPR 2025 官方）

- 项目名称：MOTIP
- 对应论文：Multiple Object Tracking as ID Prediction（arXiv:2403.16848）
- 官方 URL：https://github.com/MCG-NJU/MOTIP
- 本地路径：`references/identity_decoding/MOTIP`
- clone 日期：2026-08-06；branch：main
- commit SHA：`ffc0e905ac196a603027eca8d18fb0dff48c8bcc`
- 许可证：Apache-2.0
- 参考文件：`models/motip/motip.py`、`models/motip/id_decoder.py`、`models/motip/trajectory_modeling.py`、`models/motip/id_criterion.py`、`models/runtime_tracker.py`、`train.py`、`docs/GET_STARTED.md`、`docs/TUTORIAL.md`、`data/seq_dataset.py`
- 关键观察：
  - 历史轨迹编码：DETR output embedding 经 `TrajectoryModeling`（FFN adapter + norm + FFN）作为 trajectory feature；轨迹时间、mask、box 一并保存。
  - ID embedding：one-hot ID（K 个词 + 1 个 newborn 槽）经 `word_to_embed` 线性层得到 id_dim embedding，与 feature 拼接成 2C 维。
  - 当前检测：同样拼接“空 ID embedding”（newborn 槽 one-hot）。
  - 交互：cross-attention，query=当前检测 embeds，key/value=历史轨迹 embeds；层间有 self-attention（除第一层）；`cross_attn_mask = traj_time >= curr_time` 阻止未来；相对时间位置编码加到注意力偏置。
  - ID 预测：每层把 `unknown_embeds[..., -id_dim:]` 送入 `embed_to_word`（K+1 分类），可用 aux loss。
  - NEW/NO_MATCH：K+1 类中的最后一类是 newborn（新 ID）；没有显式 NO_MATCH 类。推理时用阈值：ID conf < id_thresh 视为未分配（等价 NO_MATCH）。
  - 训练标签：同一视频 clip 内按轨迹 ID 构造 labels；`AUG_NUM_GROUPS` 多个 ID 分配组增强数据利用。
  - 推理一对一：Hungarian 在扩展矩阵（新增 newborn 列）上求解；每个 detection 至多一个 ID，每个 ID 至多一个 detection。
  - 长短轨迹：`SAMPLE_LENGTHS == REL_PE_LENGTH >= MISS_TOLERANCE`；轨迹按时间窗口保存，超出 tolerance 视为新目标。
- 是否复用代码：否（仅设计参考）。
- 借鉴设计：PersistentTrackDecoder 的 track-as-query、candidate-as-key/value、ID/NO_MATCH 分类、匈牙利一对一、相对时间 embedding。

## 3. MOTIP-2（independent implementation / reproduction repository）

- 项目名称：MOTIP-2
- 对应论文：同上（MOTIP）
- 官方 URL：https://github.com/GISer-WB/MOTIP-2
- 本地路径：`references/identity_decoding/MOTIP-2`
- clone 日期：2026-08-06；branch：main
- commit SHA：`012856c1dc13b324064e79339ae71054518d1b5e`
- 许可证：Apache-2.0
- 来源判定：经重新确认，无法证明该仓库由 MOTIP 原作者或官方组织（MCG-NJU）维护；stargazers 少且与原始 MOTIP 主仓库独立。因此不得称其为“官方后续代码库”，按 independent implementation / reproduction repository 处理，引用层级低于原始 MOTIP 官方仓库（MCG-NJU/MOTIP）。
- 参考文件：`models/id_decoder.py`、`models/trajectory_modeling.py`、`models/motip.py`、`models/seq_decoder.py`、`configs/*.yaml`
- 关键观察：与主仓库一致的 ID 预测思想；IDDecoder 显式实现 multihead self/cross attention；`related_temporal_embeds` 为可学习相对时间偏置；支持 `MULTI_TIMES_ID_DECODER` ensemble；推理用 `linear_sum_assignment(1 - extended_id_confs)`。
- 是否复用代码：否。
- 借鉴设计：确认 MOTIP 核心设计可复现，并为我们的 decoder 提供“第一层不做 self-attn”等实现细节。

## 4. TrackFormer

- 项目名称：TrackFormer
- 对应论文：TrackFormer: Multi-Object Tracking with Transformers（CVPR 2022）
- 官方 URL：https://github.com/timmeinhardt/trackformer
- 本地路径：`references/association_transformers/TrackFormer`
- clone 日期：2026-08-06；branch：main
- commit SHA：`e468bf156b029869f6de1be358bc11cd1f517f3c`
- 许可证：Apache-2.0
- 参考文件：`src/trackformer/models/tracker.py`、`matcher.py`、`transformer.py`、`src/track.py`、`cfgs/track.yaml`
- 关键观察：track queries 由上一帧 decoder 输出 embedding 构成，与 object queries 一起输入 decoder；输出前半为 tracks、后半为新检测；matcher 对 track query 强制匹配自身 GT ID（cost=-1），新检测走 Hungarian；tracker 管理 inactive tracks 与 ReID。
- 是否复用：否。
- 借鉴：track query 跨帧复用、输出顺序分离、一对一强制约束。

## 5. MOTR

- 项目名称：MOTR
- 对应论文：MOTR: End-to-End Multiple-Object Tracking with TRansformer（ECCV 2022）
- 官方 URL：https://github.com/megvii-research/MOTR
- 本地路径：`references/association_transformers/MOTR`
- clone 日期：2026-08-06；branch：main
- commit SHA：`8690da3392159635ca37c31975126acf40220724`
- 许可证：MIT（仓库内 LICENSE 为 MIT；GitHub API 显示 Other 属误标）
- 参考文件：`models/motr.py`、`models/qim.py`、`models/memory_bank.py`、`models/deformable_transformer_plus.py`、`datasets/detmot.py`
- 关键观察：track query 更新用 QIM（self-attention + MLP + 位置更新）；tracklet-aware label assignment；消失 track 槽位（matched_gt_idx=-1）训练时不计算 box 回归；memory bank 可选，保存高分 embedding，每 3 帧写入，temporal attention 聚合。
- 是否复用：否。
- 借鉴：track query 生命周期与消失槽位（NO_MATCH 的 Transformer 等价物）；memory bank 更新策略（本阶段只记录，不实现）。

## 6. MeMOTR

- 项目名称：MeMOTR
- 对应论文：MeMOTR: Long-Term Memory-Augmented Transformer for Multi-Object Tracking（ICCV 2023）
- 官方 URL：https://github.com/MCG-NJU/MeMOTR
- 本地路径：`references/memory_tracking/MeMOTR`
- clone 日期：2026-08-06；branch：main
- commit SHA：`eb7a177b9cbcb89742ec69b2545ab3af2ea31a80`
- 许可证：MIT
- 参考文件：`models/query_updater.py`、`models/memotr.py`、`models/deformable_decoder.py`、`structures/track_instances.py`
- 关键观察：short memory 用当前输出 embedding 与历史融合；long memory 用指数滑动平均（`long_memory_lambda`）；memory attention 的 query=short memory+pos，key/value=long memory+pos；新 track 的 long memory 初始化为 query embedding。
- 是否复用：否。
- 借鉴：两帧阶段不实现完整 memory bank，但 PersistentTrackState 字段设计吸收其“reference object token 作为锚点”的思想。

## 7. SAM 2

- 项目名称：SAM 2
- 对应论文：SAM 2: Segment Anything in Images and Videos
- 官方 URL：https://github.com/facebookresearch/sam2
- 本地路径：`references/memory_tracking/sam2`
- clone 日期：2026-08-06；branch：main
- commit SHA：`2b90b9f5ceec907a1c18123530e92e794ad901a4`
- 许可证：Apache-2.0
- 参考文件：`sam2/modeling/memory_encoder.py`、`memory_attention.py`、`sam2_base.py`、`sam2_video_predictor.py`、`sam2/modeling/sam/prompt_encoder.py`
- 关键观察：prompt encoder 支持 point/box/mask；视频推理维护 per-object memory（maskmem features + pos enc）；memory attention 使用 temporal pos enc；object pointer 表示对象存在性；conditioning frames 只取最近若干帧。
- 是否复用：否。
- 借鉴：两帧接口中的 prompt 编码（box/crop → token）、object token 与 memory 的分离设计（Stage L0 只做两帧接口，不做完整 bank）。

## 8. ViPT

- 项目名称：ViPT
- 对应论文：Visual Prompt Multi-Modal Tracking（CVPR 2023）
- 官方 URL：https://github.com/jiawen-zhu/ViPT
- 本地路径：`references/visual_prompt_tracking/ViPT`
- clone 日期：2026-08-06；branch：main
- commit SHA：`f49f30186ebff5587600a61ff224bb341c0a3243`
- 许可证：MIT
- 参考文件：`lib/models/vipt/vit_prompt.py`、`vit_ce_prompt.py`、`ostrack_prompt.py`、`lib/train/base_functions.py`
- 关键观察：可学习 prompt tokens 在视觉主干 patch embedding 后注入（shallow/deep），参考模板和搜索区域共享 prompt；冻结主干，只训练 prompt（0.84M 参数）。
- 是否复用：否（LocateAnything 的 visual prompt 走官方 `<image-2>` + LoRA 路线，不移植 ViPT 主干）。
- 借鉴：visual prompt 在视觉主干之后注入的设计原则、冻结主干只训练 prompt/LoRA 的参数效率思路。

## 未 clone 但已记录

- UniVG-R1（GRPO 通用视觉 grounding）：官方仓库 AMAP-ML/UniVG-R1，HEAD `44868ea30073c104d026186418291757454ae9d7`，许可证未声明。仅记录，不 clone（Stage L0 禁止 RL）。
- Vision_GRPO（GRPO 视觉 grounding 教程）：FusionBrainLab/Vision_GRPO，Apache-2.0，HEAD `65ec90d090f93dae1f303ea7eeed8c9d0c06f64b`。仅记录。
- MedLoc-R1（医学视觉 grounding GRPO）：MembrAI/MedLoc-R1，代码尚未正式发布，HEAD `9ae2b29854db16d09d6c5c6d59b13b3c70c1b48b`。仅记录。

## 未找到官方实现

- MOTRv3：论文存在，但未发现官方公开仓库；相近官方实现以 MOTR 为准。
- R1-SAM：未检索到官方仓库。
- MedGround-R1：GitHub 仓库 404，仅论文可查。
