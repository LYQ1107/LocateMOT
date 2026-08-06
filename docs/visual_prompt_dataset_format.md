# Visual Prompt 数据格式（官方口径）

依据 `third_party/Eagle/Embodied/document/DATA_PREPARATION.md`、`eaglevl/train/tools.py`、`locateanything_worker.py`。

## 官方 JSONL 格式

每行一个样本：

```json
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}], "image": "relative/path.jpg"}
```

- 坐标：`[0,1000]` 归一化整数。
- 多图：`image_list` + `<image-1>`、`<image-2>` 占位符；无占位符时自动在首个 user 消息前加 `<image-1>`。
- 输出标签：
  - 框：`<ref>label</ref><box><x1><y1><x2><y2></box>`
  - point：`<box><x><y></box>`
  - none：`<box>none</box>`

## 官方 visual prompt 训练转换

recipe 中数据集配置 `"visual_prompt": true` 时：

1. 只转换 positive 单类别检测样本（`_extract_detection_category` 可解析的类别）。
2. 取 GPT 回复中该类别第一个 GT box，从源图裁剪 crop（`_crop_normalized_box`，坐标 ×1000 归一化）。
3. 源图保持 `<image-1>`；crop 追加为 `<image-2>`、`<image-3>`...；类别文本原位替换为占位符。
4. negative 样本（`<box>None</box>`）保持文本类别 prompt，不转换。
5. `sample["image"] = image_list + appended_crops`。

注意：官方转换目前是“同一张源图 + 该图内 GT crop”，不是两帧/两图。Stage L0 需要扩展为 current frame（image-1）+ reference crop（image-2），数据结构仍与官方完全兼容。

## 官方 LoRA 脚本入口

```bash
export HF_TOKEN=...
export META_PATH=./locany_recipe/visual_prompt_recipe.json
bash shell/locate-anything-lora-visual-prompt.sh 1 work_dirs/locany_lora_visual_prompt
```

默认：LLM LoRA rank 64、MoonViT 冻结、MLP projector 可训练、bf16、grad checkpoint、DeepSpeed ZeRO-1（可用 ZeRO-2）。

A100 适配：`attn_implementation=sdpa`、`max_seq_length<=4096`、`max_num_tokens*<=4096`、`block_size 6`、`causal_attn False`。

## Stage L0 两帧样本格式（计划）

```json
{
  "conversations": [
    {"from": "human", "value": "Locate the object in <image-1> that corresponds to the highlighted object in <image-2>."},
    {"from": "gpt", "value": "<ref>target</ref><box><120><200><450><500></box>"}
  ],
  "image_list": ["current_frame.jpg", "reference_crop.jpg"]
}
```

负样本：

```json
{"conversations": [
  {"from": "human", "value": "Locate the object in <image-1> that corresponds to the highlighted object in <image-2>."},
  {"from": "gpt", "value": "<box>none</box>"}
], "image_list": ["current_frame.jpg", "reference_crop.jpg"]}
```

- reference crop 外扩比例固定为 10% 上下文（规则待定稿后写入 `configs/data/visual_prompt_train.json`）。
- JSON 只保存索引与路径，不复制图像。
- 训练 recipe 中 `visual_prompt: true` 只对“由两帧数据构造的视觉 prompt 正样本”有意义；官方转换函数不应直接用于两帧数据（它假设同一源图），我们会生成已含 `<image-2>` 的 JSONL。
