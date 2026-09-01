# Stage L1-C LocateAnything LoRA 官方实现审计

审计时间：2026-08-09。唯一基准：NVlabs/Eagle 官方仓库
（`third_party/Eagle`，commit 783f656d127ee498137b5ff52603ce36c292d317，
2026-06-24，origin https://github.com/NVlabs/Eagle.git）。
LocateAnything-3B 权重为本地 `models/LocateAnything-3B`（nvidia 官方）。

## 1. 官方训练入口

- Shell: `Embodied/shell/locate-anything-lora-visual-prompt.sh`
- Train: `Embodied/eaglevl/train/locany_finetune_magi_stream.py`
- Data: `Embodied/document/DATA_PREPARATION.md`（JSONL + recipe meta）
- Save/load: HF Trainer 标准 checkpoint（`output_dir`，
  `save_strategy=steps`，`save_steps`，`save_total_limit`）

## 2. 官方默认 LoRA 配置（脚本实测值）

| 参数 | 官方默认 | 说明 |
|---|---|---|
| `use_llm_lora` | 64 | LLM decoder LoRA rank |
| `use_backbone_lora` | 0 | vision backbone LoRA rank（默认关闭） |
| `freeze_llm` | True | 基础 LLM 冻结 |
| `freeze_backbone` | True | vision backbone 冻结 |
| `freeze_mlp` | False | connector/MLP 可训练 |
| `mlp_connector_layers` | 2 | MLP connector 层数 |
| `lr` | 2e-5 | cosine + warmup 500 |
| `attn_implementation` | magi | 仅 Hopper/Blackwell；A100 必须换 |
| `causal_attn` | False | PBD block 注意力 |
| `bf16` | True | |
| `grad_checkpoint` | True | |
| `deepspeed` | zero_stage1_config.json | |

## 3. LoRA 插入位置（代码证据）

`eaglevl/model/locany/modeling_locateanything.py`:

- `wrap_backbone_lora(r, alpha=2r)`：PEFT LoraConfig，
  target_modules = vision `self_attn.q/k/v/out_proj` + `mlp.fc1/fc2`，
  `get_peft_model(self.vision_model)`。
- `wrap_llm_lora(r, alpha=2r)`：target_modules =
  `self_attn.q/k/v/o_proj` + `mlp.gate/down/up_proj`，
  `get_peft_model(self.language_model)`，
  `enable_input_require_grads()`。
- 训练脚本在加载 model 后：
  `if model_args.use_backbone_lora: model.wrap_backbone_lora(...)`；
  `if model_args.use_llm_lora: model.wrap_llm_lora(...)`。

## 4. A100 兼容性

官方 TRAINING.md 明确：Magi Attention 仅支持 Hopper/Blackwell；
非 Hopper（A100/L40）用 `--attn_implementation sdpa`，且 SDPA 只支持
约 4K 序列微调。因此本机 A100 必须：

- `--attn_implementation sdpa`
- `--max_seq_length` / `--max_num_tokens_per_sample` / `--max_num_tokens`
  压到 4K 内（按实际 sample 长度，先测量）。
- 保留 `--causal_attn False` 的 PBD 训练格式。

记录：非官方默认改动，原因 = 硬件架构不支持 magi kernel。

## 5. Visual Prompt 数据格式（官方机制）

`eaglevl/train/tools.py`:

- `apply_visual_prompt_to_sample`：取 human 检测 prompt 中“单一 positive
  类别”的文本，替换为 `<image-N>` placeholder；从 source image（image-1）
  按 GT box crop 出参考图作为 image-2+；negative（`<box>None</box>`）
  保持文本。
- Recipe 中 dataset 配置 `"visual_prompt": true` 才启用。
- README 声明：当前发布的 `nvidia/LocateAnything-3B` 权重不直接支持
  visual prompt 推理；visual-prompt-capable 权重将来自未来发布。

含义：官方 visual prompt 是“同一图内参考 crop + 文本转 placeholder”的
训练增强，不是“reference frame + current frame”双帧匹配接口。我们的
Route B 若要做 tracking adaptation，应基于官方 JSONL + crop 格式构造
“前一帧 crop 作为 visual prompt、当前帧作为 image-1”的训练样本；
但必须先做技术验证，确认推理时该格式可工作（若当前权重不支持，
则记录为 `AC_LORA_FIXED_BOX_FEATURE_EXTRACTION_UNSUPPORTED` 或
`VISUAL_PROMPT_WEIGHTS_UNAVAILABLE`，不伪造公平性）。

## 6. PBD hidden state 与 association loss 回传

官方训练 loop 使用 `model(input_ids, labels, pixel_values, ...)`
计算标准 language modeling loss（`locany_finetune_magi_stream.py`）。
模型 forward 返回 `hidden_states`（`modeling_locateanything.py` forward /
`output_hidden_states`），即 PBD block hidden 在训练态可获得。
但官方训练入口没有提供 association loss 钩子；若要做 joint
grounding+association，需要在官方 trainer 基础上加自定义 loss 回传，
或采用官方许可的 sequential 方案（先 LoRA grounding 训练，冻结后缓存
features，再训练 UA decoder）。实现阶段先跑 smoke 验证梯度路径，
不假设 joint 可行。

## 7. 保存/加载

HF Trainer 保存 adapter + base model config；加载用
`AutoModel.from_pretrained(output_dir)`（或 merge adapter）。
LoRA smoke 必须验证：save → load → 相同 prompt 输出正常。

## 8. 本阶段将采用的官方资产

- 官方训练脚本与 arguments（A100 参数修改后运行）。
- 官方 PEFT LoRA 封装（不自己写 LoRA）。
- 官方 JSONL/recipe 数据格式（从 train 标注生成 grounding / visual-prompt
  JSONL）。
- 官方 PBD/hidden-state 提取（复用现有 ObjectTokenExtractor 基础设施，
  在训练态验证梯度可达性后再决定 joint 或 sequential）。
