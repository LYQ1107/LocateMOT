# 实现前证据（Implementation Evidence）

本文件记录每个核心模块编码前实际阅读的官方参考、观察到的实现、以及最终设计决策。任何一项新增核心模块都必须先补充本文件再编码。

## Module: ObjectTokenExtractor

- Scientific purpose: 为 LocateAnything 输出的每个预测框建立严格一一对应的对象特征（Object Token），供两帧关联使用。
- Official references inspected: Eagle/LocateAnything（NVlabs/Eagle `Embodied/`）、SAM 2、MeMOTR。
- Repository commits:
  - Eagle `783f656d127ee498137b5ff52603ce36c292d317`
  - sam2 `2b90b9f5ceec907a1c18123530e92e794ad901a4`
  - MeMOTR `eb7a177b9cbcb89742ec69b2545ab3af2ea31a80`
- Files inspected:
  - `Embodied/eaglevl/utils/locany/modeling_locateanything.py`（`generate` 循环、MTP/AR、KV cache 截断）
  - `Embodied/eaglevl/utils/locany/generate_utils.py`（`sample_tokens`、`decode_bbox_avg`、`handle_pattern`）
  - `Embodied/eaglevl/model/locany/modeling_locateanything.py`（`forward`、`extract_feature`）
  - `Embodied/eaglevl/model/moon_vit/modeling_vit.py`（MoonViT 输出）
  - `Embodied/locateanything_worker.py`（输出解析）
  - SAM 2 `memory_encoder.py`、`memory_attention.py`、`sam2_video_predictor.py`
  - MeMOTR `query_updater.py`
- Observed implementation:
  - PBD 一次并行解码 6 个 token（`box_start + x1 + x2 + y1 + y2 + box_end`），模型内部使用固定长度 block（训练 `block_size=6`），`generate` 在 MTP 模式下从 logits 的最后 6 个位置解码。
  - `sample_tokens` 对每 batch 的 `[6, vocab]` logits 做 top-k 加权平均解码出 6 个 token；`handle_pattern` 区分 `coord_box` / `point_box` / `empty_box` / `ref_object` / `im_end`。
  - 训练 forward 中 `hidden_states = outputs.last_hidden_state`，语言模型输出 `CausalLMOutputWithPast(hidden_states=...)`，说明每个生成 token 都有对应 hidden state。
  - MoonViT 输出视觉特征经 `mlp1`（LayerNorm -> Linear -> GELU -> Linear）注入 LLM embedding；`extract_feature` 返回视觉 token 序列。
  - SAM 2 对每个对象保存 mask 相关 memory feature 与 object pointer；MeMOTR 用 query embedding 初始化 long memory。
- Parts adopted:
  - PBD box block 的 6 个 token 位置作为 hidden-state 提取窗口；候选 ObjectToken = block 内 hidden states 的聚合（先以 `box_end` 位置与全 block mean 做对照）。
  - 预测框作为 MoonViT region pooling 的区域输入。
  - 输出 box 顺序即 block 顺序（解析 `answer` 时按 `<box>` 出现顺序与解码 block 顺序一一对应）。
- Parts intentionally not adopted:
  - 不把 SAM 2 的 mask memory encoder 或 MeMOTR 的长期 memory 更新逻辑放进 Stage L0（本阶段只做两帧接口）。
  - 不假设最后一个 token 就是对象特征，需先做映射验证（`reports/pbd_token_mapping.md`）。
- Reason for final design: 官方代码明确 PBD block 与 token 位置一一对应；hidden states 由 `output_hidden_states=True` 可直接获得，因此第一版 ObjectToken 采用 “PBD block hidden state + region feature + geometry + confidence” 的融合设计，但保留 region-only 与 PBD-only 两个消融。

## Module: VisualPromptAdapter

- Scientific purpose: 支持 reference box / reference crop 两类 visual prompt，注入 LocateAnything 并保持官方数据与 token 约定。
- Official references inspected: Eagle/LocateAnything、ViPT、SAM 2。
- Repository commits:
  - Eagle `783f656d127ee498137b5ff52603ce36c292d317`
  - ViPT `f49f30186ebff5587600a61ff224bb341c0a3243`
  - sam2 `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- Files inspected:
  - `Embodied/locateanything_worker.py`（`_crop_visual_prompt`、`_replace_visual_prompt_text`、`_build_messages`）
  - `Embodied/eaglevl/train/tools.py`（`apply_visual_prompt_to_sample`、`_crop_normalized_box`）
  - `Embodied/eaglevl/train/locany_finetune_magi_stream.py`（`visual_prompt` 参数）
  - `Embodied/document/DATA_PREPARATION.md`（`<image-1>`、`<image-2>` 占位符与 JSONL 格式）
  - ViPT `lib/models/vipt/vit_prompt.py`（prompt token 注入）
  - SAM 2 `prompt_encoder.py`（box prompt 编码）
- Observed implementation:
  - 官方 worker 已实现 `visual_prompt_box` / `visual_prompt`：把 reference crop 作为 `<image-2>` 追加到消息中，`_replace_visual_prompt_text` 替换 `<visual_prompt>` / `{visual_prompt}` 或 `replace_text`。
  - 官方 README 明确声明：当前公开 `nvidia/LocateAnything-3B` 权重不支持 visual prompt 推理；必须用官方 visual prompt 微调脚本训练。
  - 官方训练转换 `apply_visual_prompt_to_sample`：positive 单类别检测样本中，把类别文本替换为 `<image-N>` 占位符，crop 从源图按 GT box 裁剪并追加到 `image_list`；negative `<box>None</box>` 保持文本 prompt。
  - 官方 LoRA 脚本默认 LLM LoRA rank 64、MoonViT 冻结、MLP projector 可训练；数据 recipe 中 `visual_prompt: true` 开启转换。
  - ViPT 的可学习 prompt token 是在视觉主干 patch embedding 后注入，且参考/搜索共享参数；SAM 2 的 box prompt 是坐标正弦位置编码 + 可学习 corner embedding。
- Parts adopted:
  - 采用官方 `<image-2>` 占位符与 `visual_prompt_box`（normalized_1000 / normalized_1 / pixel）接口。
  - Stage L0 的 VisualPromptAdapter 第一版 = 官方 crop 编码路径 + prompt 模板；训练走官方 `visual_prompt=true` 数据格式。
  - LoRA 配置沿用官方脚本默认（LLM LoRA 64、MoonViT 冻结、MLP 可训练、bf16、gradient checkpointing、DeepSpeed ZeRO-2）。
- Parts intentionally not adopted:
  - 不实现 ViPT 式“往 MoonViT 内部注入可学习 prompt token”的第一版（官方路线未要求、且权重不支持）。
  - 不把 reference box 编码为 SAM 2 式 point embedding；第一阶段只使用官方 crop 路径。
- Reason for final design: 用户要求“必须采用官方模型实际支持的数据格式和 special token，不得只写自然语言后假定模型理解视觉 prompt”；官方代码提供了明确的 crop + `<image-2>` + LoRA 路线，因此按官方格式实现并记录当前权重限制。

## Module: PairwiseMLP

- Scientific purpose: 简单两帧 pairwise 匹配基线（B3），验证 fused token 的可区分性。
- Official references inspected: MOTIP（轨迹 embedding 的 FFN 建模）、MOTR（QIM 的 MLP 更新）、TrackFormer（ReID 相似度）。
- Repository commits: MOTIP `ffc0e905...`、MOTR `8690da3...`、TrackFormer `e468bf1...`。
- Files inspected:
  - MOTIP `models/motip/trajectory_modeling.py`
  - MOTR `models/qim.py`
  - TrackFormer `src/trackformer/models/tracker.py`（`reid` 函数）
- Observed implementation:
  - MOTIP 的 trajectory modeling 是 FFN（adapter + norm + FFN + norm），没有 Transformer。
  - MOTR QIM 用 self-attn + MLP 更新 track query。
  - TrackFormer ReID 用 `reid_sim_threshold` 与 cosine similarity 匹配 inactive tracks。
- Parts adopted: pairwise MLP 输入 = reference token 与 current token 的拼接（+ 可选的 box 几何与 gap 特征），输出 logit 经 sigmoid/softmax。
- Parts intentionally not adopted: 不在 PairwiseMLP 中加入 self-attention（那是 Track Decoder 的职责）。
- Reason for final design: 作为最简基线，官方实现（MOTIP FFN、QIM MLP）证明 MLP 是合理最简建模；PairwiseMLP 用于回答“简单特征匹配是否已足够”。

## Module: PersistentTrackDecoder

- Scientific purpose: 多目标两帧一对一身份关联，支持 NO_MATCH，不依赖语言模型生成数字 ID。
- Official references inspected: MOTIP、MOTIP-2、TrackFormer、MOTR、MeMOTR。
- Repository commits:
  - MOTIP `ffc0e905ac196a603027eca8d18fb0dff48c8bcc`
  - MOTIP-2 `012856c1dc13b324064e79339ae71054518d1b5e`
  - TrackFormer `e468bf156b029869f6de1be358bc11cd1f517f3c`
  - MOTR `8690da3392159635ca37c31975126acf40220724`
  - MeMOTR `eb7a177b9cbcb89742ec69b2545ab3af2ea31a80`
- Files inspected:
  - MOTIP `models/motip/id_decoder.py`、`models/runtime_tracker.py`
  - MOTIP-2 `models/id_decoder.py`、`models/seq_decoder.py`
  - TrackFormer `src/trackformer/models/tracker.py`、`matcher.py`
  - MOTR `models/qim.py`、`models/motr.py`
  - MeMOTR `models/query_updater.py`
- Observed implementation:
  - MOTIP：current detections 作为 query，历史轨迹（feature + ID embedding）作为 key/value；层间 self-attention 处理当前目标竞争；输出 K+1 维 ID logits；训练用 CE/focal，推理用 Hungarian + 阈值。
  - MOTIP-2：同样的 IDDecoder，第一层不做 self-attn，可学习相对时间位置偏置。
  - TrackFormer/MOTR：track query 跨帧复用，decoder self/cross attention 自动完成关联；一对一由 matcher/Hungarian 强制。
  - MeMOTR：memory attention 读取 long memory。
- Parts adopted:
  - 输入 = M 个 reference track tokens（queries）与 N 个 current object tokens（keys/values）。
  - 4 层 decoder，d_model=256，8 heads，FFN 1024；第一层可选不做 self-attn（按 MOTIP 经验）。
  - 输出 `assignment_logits [M, N+1]`，最后一列为 NO_MATCH。
  - 推理用 Hungarian（或等价一对一分配），保证一个 candidate 至多分配给一个 track。
- Parts intentionally not adopted:
  - 不引入 ID 词表分类（MOTIP 的 K+1 方式需要 ID 循环/词表管理，Stage L0 只有固定 reference 集合，直接输出 N+1 分配更简单且满足“不生成数字 ID”）。
  - 不实现 newborn 类（本阶段无 automatic birth）。
  - 不实现完整 memory bank / QIM 生命周期。
- Reason for final design: MOTIP 证明了 “current 作为 query、trajectory 作为 memory、self-attn 处理竞争、分类头输出关联” 的端到端关联范式；Stage L0 的固定两帧设定使其可简化为 M×N 分配 + NO_MATCH，同时保留 MOTIP 的上下文竞争与一对一约束。

## Module: Hungarian assignment

- Scientific purpose: 保证推理时一对一分配（一个 candidate 最多一个 track，一个 track 最多一个 candidate 或 NO_MATCH）。
- Official references inspected: MOTIP、MOTIP-2、TrackFormer。
- Repository commits: MOTIP `ffc0e905...`、MOTIP-2 `012856c1...`、TrackFormer `e468bf1...`。
- Files inspected:
  - MOTIP `models/runtime_tracker.py`（`_hungarian_assignment`）
  - MOTIP-2 `models/motip.py`（`linear_sum_assignment(1 - extended_id_confs)`）
  - TrackFormer `src/trackformer/models/matcher.py`
- Observed implementation:
  - MOTIP-2 直接 `scipy.optimize.linear_sum_assignment(1 - id_confs)`，把 newborn 列复制为每行重复列以支持“未匹配”选项。
  - MOTIP main 有 `assignment_protocol`（hungarian / id-max / object-max），默认 hungarian；论文消融说明 Hungarian 与简化贪心差异 <0.3 HOTA。
  - TrackFormer matcher 强制 track query 与自身 ID 匹配，其余走标准 Hungarian。
- Parts adopted: Stage L0 推理使用 `linear_sum_assignment` 对 `[M, N+1]` 代价矩阵求解；NO_MATCH 列可作为未分配槽位。
- Parts intentionally not adopted: 不实现多种分配协议（只做 Hungarian + 阈值），不做 ID 词表扩展。
- Reason for final design: 官方实现一致使用 Hungarian 保证一对一；阈值（id_thresh）控制 NO_MATCH 决策，符合本阶段要求。

## Module: NO_MATCH

- Scientific purpose: 支持 reference 目标在当前帧不可见/不存在合法 GT 时输出 no-match。
- Official references inspected: LocateAnything、MOTIP、MOTR、SAM 2。
- Repository commits:
  - Eagle `783f656d...`
  - MOTIP `ffc0e905...`
  - MOTR `8690da3...`
  - sam2 `2b90b9f...`
- Files inspected:
  - `Embodied/eaglevl/utils/locany/generate_utils.py`（`is_valid_box_frame` 的 empty_box 分支）
  - `Embodied/locateanything_worker.py`（`<box>none</box>`）
  - MOTIP `models/runtime_tracker.py`（`id_thresh`、未分配处理）
  - MOTR `models/motr.py`（track-disappear：`matched_gt_idxes = -1`，box loss 忽略）
  - SAM 2 `sam2_video_predictor.py`（`NO_OBJ_SCORE` placeholder）
- Observed implementation:
  - LocateAnything 用 `<box>none</box>` 表示无对象；训练数据 negative 样本保留文本 prompt（visual_prompt=true 不转换 negative）。
  - MOTIP 用阈值：ID 置信低于 `id_thresh` 即未分配（相当于 NO_MATCH）。
  - MOTR 用 track-disappear 槽位（matched GT=-1），分类仍监督、回归忽略。
  - SAM 2 用 `NO_OBJ_SCORE` 大负值表示对象缺失。
- Parts adopted:
  - Track Decoder 中 NO_MATCH 为可学习 token/列，输出 logit 与候选一起参与分类。
  - 损失权重 `L_no_match=0.5`（按项目规格）。
  - Visual prompt 负样本按官方格式保留 `<box>none</box>`。
- Parts intentionally not adopted: 不引入 lost age / termination（Stage L1）。
- Reason for final design: 官方实现提供了三种 no-match 表示（文本 none、阈值未分配、disappear 槽位）；Stage L0 用显式 NO_MATCH 列最直接，且能与 Hungarian 天然结合。

## Module: visual prompt LoRA 数据管线

- Scientific purpose: 按官方 visual prompt 微调格式构建两帧训练数据（reference crop + current image / reference full image + box）。
- Official references inspected: Eagle/LocateAnything。
- Repository commits: Eagle `783f656d127ee498137b5ff52603ce36c292d317`。
- Files inspected:
  - `Embodied/shell/locate-anything-lora-visual-prompt.sh`
  - `Embodied/eaglevl/train/locany_finetune_magi_stream.py`（`visual_prompt` 开关、`apply_visual_prompt_to_sample` 调用）
  - `Embodied/eaglevl/train/tools.py`（crop 与占位符替换）
  - `Embodied/document/DATA_PREPARATION.md`
- Observed implementation:
  - 数据为 JSONL（ShareGPT 格式），`conversations` 中 human/gpt 两轮；`image` 或 `image_list` 放图像路径。
  - recipe JSON 中 `visual_prompt: true` 的数据集自动把 positive 单类别 detection 转换为 crop prompt；source image 保持 image-1，crop 作为 image-2 追加；类别文本被 `<image-N>` 替换。
  - 负样本 `<box>None</box>` 不转换，保持类别文本。
  - LoRA 脚本 `USE_LLM_LORA=64`、`FREEZE_LLM=True`、`FREEZE_BACKBONE=True`、`FREEZE_MLP=False`；`attn_implementation=magi` 默认（A100 需改为 sdpa + 4K 序列）。
- Parts adopted:
  - 完全沿用官方 JSONL/recipe 结构与 token 约定。
  - 两帧扩展：`image_list=[current_frame, reference_crop]`，human 文本用 `<image-1>`/`<image-2>` 占位符与官方任务模板。
  - 训练超参按官方默认 + A100 适配（`--attn_implementation sdpa --max_seq_length 4096 --causal_attn False --block_size 6`）。
- Parts intentionally not adopted: 不修改官方 `tools.py`；通过生成符合官方格式的 JSONL 实现两帧转换。
- Reason for final design: 用户明确要求“必须采用官方模型实际支持的数据格式和 special token”，官方代码已提供完整转换逻辑，第一版只需生成与其完全兼容的数据。
