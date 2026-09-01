# LocateAnything 架构审计

来源：NVlabs/Eagle commit `783f656d127ee498137b5ff52603ce36c292d317` + HF 模型卡片。

## 整体结构

- 视觉编码器：MoonViT-SO-400M（`MoonVitPretrainedModel`），patch 14，hidden 1152，27 层，merge kernel 2×2（即像素重组 4×），输出 4×1152=4608 维特征。
- 投影：`mlp1` = LayerNorm(4608) -> Linear(LLM hidden 2048) -> GELU -> Linear(2048)；HF 卡片称 MLP projector，训练脚本 `--mlp_connector_layers 2`。
- 语言模型：Qwen2.5-3B-Instruct（自定义 `Qwen2ForCausalLM`，支持 magi/sdpa 与 MTP block mask）。
- 输出公式：block-based PBD 结构。

## PBD 与 block 结构

- 模型训练/推理使用固定长度 block：`block_size=6`（官方脚本 `--block_size 6`）。
- 一个 block = 6 个 token 位置：`[box_start, x1, x2, y1, y2, box_end]`（坐标 token 为 `<0>..<1000>`，即 coord_start..coord_end 连续 id）。
- 官方 README/HF 卡片描述输出块类型：Semantic block（`<ref>label</ref>`）、Box block（`<box><x1><y1><x2><y2></box>`）、Negative block（`<box>none</box>`）、End block（`<|im_end|>`）；未用位置以 `<null>` 填充。
- 训练时 MTP 注意力：`causal_attn=False`，block 内双向/并行，block 间因果；`create_block_diff_mask_by_pe_4d(block_size=6, ...)`。
- 推理模式：
  - fast（MTP only）：每步并行预测 6 token。
  - slow（NTP only）：纯自回归。
  - hybrid（默认）：MTP 为主，`error_box`/格式异常时回退 AR，遇到 `box_end` 再切回 MTP。
- `generate` 中 `n_future_tokens=6`；`_prepare_inputs_in_mtp` 把最后 token 复制 + 5 个 `<text_mask>` 拼成 6 位置输入，position id 对齐；从 `outputs.logits[:, -6:, :]` 解码。

## 解码细节（generate_utils.py）

- `is_valid_box_frame`：检查 `box_start` 概率、`none`+`box_end` 组合（empty_box）、结束位置 `box_end/null/im_end` 概率。
- `decode_bbox_avg`：对位置 1–4 做 top-k（默认 5），要求 top-k 中至少一个坐标 token；hybrid 模式下若 `first_valid_probs<0.9 且 valid_count>1 且 max-min>60` 判为异常并把该坐标置 0（触发 fallback 逻辑）。
- `decode_ref`：`<ref>` 起始的语义 block。
- `handle_pattern`：把 6 token 分类为 im_end / empty_box / coord_box / point_box / error_box / ref_object；point 是前 4 个 token（`<box><x><y></box>`）。

## Hidden states 与 token 的对应

- `forward` 返回 `CausalLMOutputWithPast(hidden_states=outputs.hidden_states)`；自定义 `generate` 支持 `output_hidden_states` 透传。
- 因此每个生成 token（含 PBD block 内 6 个位置）都有对应 hidden state；候选 ObjectToken 的映射方案见 `docs/pbd_token_mapping.md`。

## 输入输出与数据格式

- 输入：RGB 图像（原始分辨率，最多约 2.5K）+ 文本 prompt；processor 动态缩放，图像 token 数可变。
- 输出：上述 block 结构文本。
- 训练数据：JSONL ShareGPT 格式；坐标 `[0,1000]` 归一化整数；多图像用 `<image-1>`、`<image-2>` 占位符。
- visual prompt：`<image-2>` 追加 reference crop；负样本 `<box>none</box>`。

## 对 Stage L0 的直接影响

1. ObjectToken 提取必须基于 PBD 6-token block 的 hidden states，不能假设“最后一个 token”。
2. Fast/Hybrid/Slow 三种模式对应不同解码路径，token 映射验证需要分别测试。
3. A100 训练必须用 sdpa + 4K 序列。
4. visual prompt 需要 LoRA 微调，当前权重不直接支持。
