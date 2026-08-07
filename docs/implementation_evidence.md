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

### 方向修订（2026-08-06，Stage L0-B）

MOTIP 官方实现的方向是 current detections 作为 query、historical trajectories 作为 key/value。当前 Stage L0 设计是 reference tracks 作为 query、current objects 作为 key/value。这不是对 MOTIP 实现的直接照搬，而是为“固定 reference 集合 → current 候选的一对一检索”有意反转：

- 本设计让每个 reference track 独立查询 current 候选，并让 NO_MATCH 成为每个 track 的独立决策，天然支持多个 track 同时不可见。
- 第一版将其作为 Stage L0 的正式设计。
- 接口必须保留方向可切换性：后续至少做一个轻量对照（reference-query vs current-query），本阶段不训练两种完整模型，只确保接口支持比较。

### NO_MATCH 分配矩阵修订（2026-08-06，Stage L0-B）

原始设计 `assignment_logits [M, N+1]` 若直接交给 Hungarian，单个 NO_MATCH 列只能被使用一次，错误地禁止了多个 reference 同时不可见。修订为：

- 模型原始输出仍为 `match_logits [M, N]` 与 `no_match_logits [M, 1]`；
- 推理分配矩阵扩展为 `[M, N+M]`：
  - 前 N 列是真实 current candidates；
  - 后 M 列是每个 reference track 自己的 NO_MATCH dummy；
  - track i 只允许使用 dummy 列 N+i；其他 track 的 dummy 列对 track i 设为不可用大代价；
- 或者使用数学等价、允许每个 track 独立 NO_MATCH 的一对一求解。

必须新增测试：3 个 reference tracks 全部 NO_MATCH；2 个 NO_MATCH + 1 个真实匹配；多个 track 不能共享同一真实 candidate；每个 track 只能使用自己的 NO_MATCH dummy。

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
- Parts adopted: Stage L0 推理使用 `linear_sum_assignment` 对扩展后的 `[M, N+M]` 代价矩阵求解；track i 的 NO_MATCH dummy 列是 N+i，其他 dummy 列置为大代价；这允许任意多个 track 同时 NO_MATCH，同时保持一对一。
- Parts intentionally not adopted: 不实现多种分配协议（只做 Hungarian + 阈值），不做 ID 词表扩展。
- Reason for final design: 官方实现一致使用 Hungarian 保证一对一；阈值（id_thresh）控制 NO_MATCH 决策。MOTIP 的 newborn 列其实是每个 detection 自己的“未匹配”槽；Stage L0 的 track 侧 NO_MATCH 需要每个 track 一个独立槽，因此扩展为 [M, N+M]。

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

## Stage L0-B 实测证据（2026-08-06）

### Module: PBD Generation Trace（instrumented）

- Scientific purpose: 不猜测 box 与 hidden state 的关系，在官方 generate 循环上记录每个 block 事件。
- Official references inspected: NVlabs/Eagle `Embodied/eaglevl/utils/locany/modeling_locateanything.py::generate`、`generate_utils.py`。
- Repository commits: Eagle `783f656d127ee498137b5ff52603ce36c292d317`。
- Files inspected: 同上 + `locateanything_worker.py`。
- Observed implementation: MTP 每步 6 token；`sample_tokens` 从最后 6 个 logits 解码；`handle_pattern` 区分 coord_box/point/empty/ref/end；hybrid 在 error_box 时切 AR、box_end 时切回 MTP；官方 `@torch.no_grad()`。
- Parts adopted: 完整复现循环（同一 `sample_tokens`/`handle_pattern` 函数），用 forward hook 只捕获最后两层 hidden states，避免 `output_hidden_states=True` 的全量内存。
- Parts intentionally not adopted: 不修改官方文件；不复制整段官方代码进入核心包（驱动逻辑为本项目实现，解码函数直接调用官方模块）。
- Reason for final design: 需要事件级证据；同时验证 traced 输出与官方 `model.generate` 逐字一致（实测 MATCH=True）。

### Module: PBD ObjectToken 映射与坐标顺序

- Observed implementation（实测）:
  - 6-token 顺序 = `[box_start, x1, y1, x2, y2, box_end]`（合成图 GT (100,200,300,400) → 输出 157,416,469,838）。
  - accepted coord_box 数量 = parsed boxes = ObjectToken（149=149=149，100%）。
  - rejected MTP / point / None / ref / end 不生成 ObjectToken。
  - Hybrid fallback 中被拒 MTP 的部分 box token 与 AR token 合并为最终 box，hidden 位置按 token 实际输入位置取。
- Parts adopted: 事件 schema（GenerationBlockEvent）、输出顺序 `output_order`、hidden-state 位置记录。
- Parts intentionally not adopted: 不把 point/empty/ref 作为对象特征；不保存所有层 hidden states。
- Reason for final design: 映射门禁要求 accepted blocks = final boxes = ObjectTokens，且只允许 accepted coordinate box block 进入特征。

### Module: MoonViT Region Extractor

- Observed implementation: image processor 产生 pre-merge grid `(H/14, W/14)`；`MoonVitPretrainedModel` 用 2×2 patch merger 输出 `(H/28, W/28, 4608)` 特征；无 cls token；多图按 token 维度拼接。
- Parts adopted: `MoonViTRegionExtractor` 用 `image_grid_hws` 计算 feature grid，归一化 box → `ceil` 映射到特征坐标，框内 mean pooling；同一图像内 token 不跨图（单图索引）。
- Parts intentionally not adopted: 不做 ROIAlign 第一版（mean pooling 足够接口验证）；不跨图像读取 token。
- Reason for final design: 官方代码明确了 grid 映射；实测 feature token 数 = grid 乘积。

### Module: Fused ObjectToken 接口

- Parts adopted: `ObjectTokenProjection`（pbd 2048 → 256、region 4608 → 256、geometry(5) → 32、generation score → 32、fuse → 256），随机初始化，未训练。
- Parts intentionally not adopted: 不把随机投影的相似度当作科学结果（sanity 中 fused AUC 仅作接口检查）。
- Reason for final design: 规格要求统一 256 维 fused feature，且明确随机投影只用于接口测试。

### Module: NO_MATCH 分配（Hungarian 扩展）

- Parts adopted: `[M, N+M]` 代价矩阵；track i 的 NO_MATCH dummy 列为 N+i；其他 dummy 列大代价；`linear_sum_assignment` 求解。
- 测试：3 track 全 NO_MATCH、2 NO_MATCH+1 匹配、candidate 不共享、每 track 只用自己 dummy，全部通过。
- Reason for final design: 修正了原始 `[M, N+1]` 只能使用一次 NO_MATCH 的设计错误。

---

# Stage L0-D Association Evidence（2026-08-07）

## 诊断证据：为什么 B0 IoU 很强

### Module: Baseline diagnosis (B0/B1/B2/B3/B4 held-out)

- Scientific problem: 在 learned association 设计前，先量化简单几何/外观基线在哪些子组占优。
- Official references inspected: 无（本项目已冻结基线）。
- Files inspected: `outputs/l0_c/baseline_results.csv`、`tools/evaluate_l0_c.py`、`tools/l0d_analyze.py`。
- Observed implementation: 官方脚本对 B3/B4 存在两个评估伪影：no_match logits 在 batch padding 宽度上平均；B3 分配矩阵使用 padding 后宽度（可选中 padded 假候选）。我们在 `tools/l0d_analyze.py` 中同时复现 official-style 与 clean-style。
- Results:
  - Official-style（与 L0-C 记录完全一致）：B0 cond=0.7432 / e2e=0.4960 / NO_MATCH F1=0.7465；B2-box-end cond=0.6432；B3 cond=0.6181；B4 cond=0.6268。
  - Clean-style（只对真实候选分配）：B3 cond=0.6776；B4 cond=0.6268；B0 不变 0.7432。
  - 5–8 target：B0=0.516，B3=0.223，B4=0.229。
  - hard 子集（冻结定义，held-out 1240/2556）：B0 easy=0.9557 / hard=0.6387；B4 easy=0.9315 / hard=0.4769。
- Parts adopted: 用 clean-style 作为本阶段 B5/B6 对比口径；official-style 用于复现 L0-C。
- Parts intentionally not adopted: 不把 padding 伪影当作模型能力。
- Reason: 避免基线数字不可比。

## Temporal-gap confound

- `outputs/l0_d/diagnosis/gap_composition.json` 实测：gap>64 桶只含 YouTube-VOS（432/432）、candidate_mean=1.91、无 5–8 目标样本；gap 1–4 桶 100% MOSEv2+generic、candidate_missing 高。B4 gap>64=0.7907 主要由更容易的低密度 YouTube 子集驱动，标记为 confounded，不解释为长时间隔能力强。

## 参考实现证据（每个设计点）

### Module: RelationFeature / RelationMLP

- Scientific problem: 显式 pairwise 几何/外观关系先验，避免 Transformer 从零重学。
- Official references inspected: GRAE-3DMOT（`spatial_proj(19)->MLP`、DFFL）、TADN（spatial embedding MLP）、GMTracker（IoU+ReID 相加）。
- Repository commits: GRAE `63def8bd`、TADN `2486a5c8`、GMTracker `2a6cc634`。
- Files inspected: GRAE `models/main.py`、TADN `components/transformer.py`、GMTracker `GMMOT/model.py`。
- Observed implementation: 官方用 MLP 编码 pairwise 几何，输出 relation feature 与 scalar affinity。
- Parts adopted: RelationMLP：输入几何 delta（IoU、中心距、尺寸比等）+ 外观 cos（PBD box-end）+ generation score + 时间 gap → D→128→128，输出 relation_embedding(128) + relation_score(1)。
- Parts intentionally not adopted: 不做 3D 运动学、不做图卷积、不做门控迭代。
- Reason: 两帧关联问题只需要轻量 pairwise 编码；官方代码没有证明更复杂结构在此场景必要。

### Module: BaseAffinity + Residual

- Scientific problem: 保留 IoU 强先验，learned 只做有限修正。
- Official references inspected: GMTracker（`Mp0 = U^T U + iou`）、GTR（`max(asso, IoU)`）、TADN（IoU additive bias）。
- Observed implementation: 官方把外观相似度与 IoU 直接组合。
- Parts adopted: `BaseAffinity = w_iou * f(IoU) + w_pbd * f(PBD_box_end_cos)`，`FinalAffinity = BaseAffinity + alpha * tanh(Residual)`，alpha 初始化 0.25、经 sigmoid 约束在 [0,1)，w 初始 0.5 可学习。此公式为本项目 clean design（数学定义公开，无逐字复制）。

### Module: Relation-aware attention bias

- Scientific problem: 把 relation 信息注入 decoder cross-attention。
- Official references inspected: TADN（memory_mask = weighted IoU additive bias）、GRAE（`attn_dist = -temporal_dist` additive bias）、FDTA（relative temporal PE additive bias）。
- Observed implementation: 官方均为 additive bias 形式。
- Parts adopted: `Attention_ij = Q_i K_j/sqrt(d) + beta * RelationBias_ij`；beta 初始化 0.05，经 sigmoid 约束。

### Module: Multi-object competition

- Scientific problem: 5–8 target 明显弱，需要显式竞争。
- Official references inspected: CO-MOT QIM（track self-attention）、FDTA IDDecoder（同帧 self-attention）、MOTIP（self-attn after layer 0）。
- Parts adopted: B6 保留 reference self-attention，并增加 relation-aware global matching；B5 通过 row-softmax 与 Hungarian 全局竞争。

### Module: Hard-negative sampling

- Scientific problem: 训练要覆盖候选混淆与多目标竞争。
- Official references inspected: CO-MOT `_add_fp_tracks`、TrackFormer FP track、HNCD 最近框替换。
- Observed implementation: 官方在训练中把与真值 IoU 最大的未匹配检测插入为假 track/框。
- Parts adopted: train split 按 target count 加权采样（single≈25%、2–4≈45%、5–8≈30%）；batch 内 hard 子集（IoU margin 小 / 密度高 / 共享候选）上采样 2 倍；不复制少量 5–8 样本。

### Module: TrackEval two-frame dataset adapter

- Scientific problem: 用官方指标评估两帧关联。
- Official references inspected: TrackEval `12c8791b`。
- Observed implementation: `combine_sequences` 聚合；MOT Challenge 数据 xywh 格式。
- Parts adopted: 新增 `locatemot/evaluation/two_frame_trackeval.py`，从内存数据构造 raw_data，其余 preproc/similarity/指标/聚合全部复用官方 TrackEval。
