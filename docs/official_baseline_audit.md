# 官方基线审计（Official Baseline Audit）

## 基本信息

- 官方仓库：https://github.com/NVlabs/Eagle
- 本地路径：`third_party/Eagle`（`references/locateanything` 为软链）
- commit：`783f656d127ee498137b5ff52603ce36c292d317`（2026-06-24）
- branch：main
- 官方模型：https://huggingface.co/nvidia/LocateAnything-3B
- 官方模型页面 README 已读取（HF API + raw README）。
- 论文：https://arxiv.org/abs/2605.27365 （LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding, ECCV 2026）
- 代码许可证：Apache-2.0（`third_party/Eagle/LICENSE`）
- checkpoint 许可证：NVIDIA non-commercial license（`Embodied/LICENSE_MODEL`），允许学术/非商业研究。

## 必须记录项

### 官方 repo commit

`783f656d127ee498137b5ff52603ce36c292d317`，main，clone 于 2026-08-06。

### 代码许可证

Apache-2.0（仓库顶层 `LICENSE`）。注意：`Embodied/eaglevl/model/locany/modeling_locateanything.py` 等文件头写 “Licensed under The MIT License”，但仓库顶层 LICENSE 是 Apache-2.0；以仓库正式 LICENSE 为准，并保留文件头原样。

### checkpoint 许可证

NVIDIA License（非商业），见 `third_party/Eagle/Embodied/LICENSE_MODEL`：允许非商业研究/评估用途、复制、修改、再分发（须附完整许可证与版权声明）；不允许商业使用（NVIDIA 及其关联方除外）。

### MoonViT 模型路径

- 官方 HF 页面声明视觉编码器为 MoonViT-SO-400M（MIT License，moonshotai/MoonViT-SO-400M）。
- 代码路径：`Embodied/eaglevl/model/moon_vit/modeling_vit.py`；配置 `MoonViTConfig`：patch_size=14，hidden_size=1152，27 层，16 heads，intermediate 4304，merge kernel (2,2)。
- 权重随 `nvidia/LocateAnything-3B` 一起发布，不单独下载。

### Qwen 模型配置

- 语言模型：Qwen2.5-3B-Instruct（Qwen Research License）。
- 代码路径：`Embodied/eaglevl/model/locany/modeling_qwen2.py`（训练）/ `Embodied/eaglevl/utils/locany/modeling_qwen2.py`（推理）。
- `LocateAnythingConfig` 关键字段：
  - `image_token_index=151667`
  - `box_start_token_id=151668`，`box_end_token_id=151669`
  - `ref_start_token_id=151672`，`ref_end_token_id=151673`
  - `coord_start_token_id=151677`，`coord_end_token_id=152677`
  - `none_token_id=4064`
  - `null_token_id=152678`，`im_end_token_id=151645`，`switch_token_id=152679`，`default_mask_token_id=151676`

### projector 结构

`mlp1 = Sequential(LayerNorm(vit_hidden_size*4), Linear(-> llm_hidden), GELU, Linear(-> llm_hidden))`。MoonViT 特征经 pixel-shuffle 4 倍通道后直接映射；`mlp_connector_layers` 参数支持 2 层（训练脚本默认 2）。

### PBD 代码入口

- 推理入口：`Embodied/eaglevl/utils/locany/modeling_locateanything.py::LocateAnythingForConditionalGeneration.generate`
- PBD 解码：`Embodied/eaglevl/utils/locany/generate_utils.py`：`sample_tokens` / `decode_bbox_avg` / `decode_ref` / `handle_pattern` / `is_valid_box_frame`
- 训练入口：`Embodied/eaglevl/train/locany_finetune_magi_stream.py`，MTP block 训练（`block_size=6`，`causal_attn=False`）
- 官方 worker：`Embodied/locateanything_worker.py`

### PBD 输出解析入口

- `locateanything_worker.py::LocateAnythingWorker.parse_boxes` / `parse_points`。
- 输出格式：`<ref>label</ref><box><x1><y1><x2><y2></box>`；point `<box><x><y></box>`；none `<box>none</box>`；坐标是 `[0,1000]` 整数（除以 1000 得相对坐标）。
- 注意：官方解码内部 token 顺序为 `[box_start, x1, x2, y1, y2, box_end]`，文本输出按 `x1,y1,x2,y2` 呈现（`decode_bbox_avg` 的 top-k 作用于位置 1–4；`decode_ref`/`handle_pattern` 负责文本结构）。解析以官方 worker 的 regex 为准。

### visual prompt LoRA 入口

- 脚本：`Embodied/shell/locate-anything-lora-visual-prompt.sh`
- 数据转换：`Embodied/eaglevl/train/tools.py::apply_visual_prompt_to_sample`（`visual_prompt=true` 时调用）
- 默认参数：`USE_LLM_LORA=64`，`USE_BACKBONE_LORA=0`，`FREEZE_LLM=True`，`FREEZE_BACKBONE=True`，`FREEZE_MLP=False`，bf16，DeepSpeed ZeRO-1（脚本默认，可用 ZeRO-2），grad checkpoint。

### 当前官方权重是否直接支持 visual prompt

不支持。`Embodied/README.md` 明确说明：当前公开 `nvidia/LocateAnything-3B` 权重不支持 visual prompt 推理；visual-prompt-capable 权重将在未来版本发布。官方提供了 LoRA 微调代码与数据格式，必须在自有数据上微调。

### A100 训练限制

- Magi Attention 仅支持 Hopper/Blackwell（H100/H800/H20/Blackwell）。
- A100（Ampere）只能用 `sdpa`，官方文档指出 SDPA 仅支持约 4K 序列的微调。
- 因此 Stage L0 A100 训练配置：`--attn_implementation sdpa`、`--max_seq_length 4096`、`--max_num_tokens_per_sample 4096`、`--block_size 6`、`--causal_attn False`、bf16、gradient checkpointing、DeepSpeed ZeRO-2、per-device batch 1 + gradient accumulation。
- 官方 HF 页面说明 A100 支持推理；batch runtime 的 `la_flash` 路径在 A100 4K probe 上峰值显存 11.71 GB（vs SDPA 35.12 GB）。

### 推荐 attention backend

- 训练（A100）：`sdpa`（官方文档明示）。
- 推理（A100）：`la_flash`（HF 模型仓库附带的 batch runtime，FlashAttention varlen sparse range，无需自编译 CUDA 扩展）；标准路径可用 `sdpa` 或 `flash_attention_2`（若环境有 flash-attn）。

### 官方 issue 相关说明

- #86（open）：询问 visual prompt（bounding box 作为输入）未来支持；无官方答复正文，但 README 已说明当前权重不支持。
- #81（open）：LoRA 微调支持；官方已在 2026-06 发布 `locate-anything-lora-visual-prompt.sh`。
- #68（open）：单张 L40 推理 OOM（40.42 GiB 分配失败）；官方 batch runtime README 给出 A100 4K 图 la_flash 11.71 GB 结果，提示高分辨率需 `la_flash` 或降低分辨率。
- #85（open）：Continual SFT 数据格式 AssertionError/ValueError；说明自定义 JSONL 必须严格遵守官方 token 与消息格式。
- #53（closed）：关于 backbone 选择与 RL 未来方向的问题；记录到 `docs/future_rl_reference.md`。

## 结论

官方 repo 与模型均可达；代码许可 Apache-2.0、模型非商业许可允许本项目研究用途。A100 训练需使用 SDPA 与 ≤4K 序列。当前权重不支持 visual prompt 推理，Stage L0 的 visual prompt 必须走官方 LoRA 微调路线。
