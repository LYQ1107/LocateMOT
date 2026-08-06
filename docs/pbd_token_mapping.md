# PBD Box ↔ Object Token 映射方案（待验证）

状态：设计稿，尚未运行模型验证。Stage L0-B 将以 `reports/pbd_token_mapping.md` 记录验证结果。

## 依据（官方代码）

- `generate` 的 MTP 路径：`next_token_logits = outputs.logits[:, -n_future_tokens:, :]`，`n_future_tokens=6`。
- block token 顺序：`[box_start, x1, x2, y1, y2, box_end]`（`decode_bbox_avg`）。
- `handle_pattern` 输出 `coord_box` 时 tokens 为 6 个；`point_box` 为 4 个；`empty_box` 为 `[box_start, none, box_end]`；`ref_object` 为语义 block。
- `forward` 支持 `output_hidden_states=True`。

## 映射规则（候选）

对每个预测框 j：

1. 解析 answer 中的 `<box>...`，按出现顺序编号（j = 0,1,2,...）。
2. 同时记录每个 block 在生成 token 序列中的绝对位置 `block_start`（box_start token 位置）与 `block_end`（box_end token 位置）。
3. PBD Hidden Token（方案 A）：取 `hidden_states[-1][block_start:block_end+1]` 聚合：
   - A1：`mean`（6 token 平均）
   - A2：`box_end` 位置
   - A3：`box_start` 位置
   - 验证三者与 box 的对应稳定性后选定（先做小实验）。
4. MoonViT Region Token（方案 B）：用预测 box 在 MoonViT 特征图上做 ROI 对齐/池化（或取框内 patch token 的均值）；需要拿到 `mlp1` 之前或之后的视觉特征。
5. Fused Object Token（方案 C）：PBD token（proj 到 256D）+ region token（proj）+ box geometry（归一化 xyxy 的 MLP）+ semantic embedding + confidence，统一投影到 256D。

## 必须验证

- 输出 3 个框时恰好 3 个 ObjectToken，顺序一致。
- batch 中不同图片不串位。
- Fast / Hybrid / Slow 三种模式都可对应。
- 同一输入重复推理结果确定（temperature=0 或固定 seed）。
- 解析 box 与 hidden state 映射不包含未来信息（MTP block 内双向是官方设计，跨 block 因果；对象 token 提取只用于特征，不引入未来 box）。

## 与训练侧 block 的关系

- 训练用 `block_size=6`、`causal_attn=False`：block 内 token 互相可见，block 间因果。ObjectToken 聚合 block 内 hidden states 与训练注意力一致。
- 若未来需要更“干净”的对象 token，可只在训练时额外加一个对象 token 头；Stage L0 先用现有 hidden states 验证是否足够。

## 输出结构

`ObjectToken`：

```json
{
  "box_xyxy": [x1, y1, x2, y2],
  "normalized_box": [x1/1000, y1/1000, x2/1000, y2/1000],
  "pbd_feature": "...",
  "region_feature": "...",
  "semantic_feature": "...",
  "confidence": 0.0,
  "fused_feature": "...",
  "block_start": 0,
  "block_end": 0,
  "query_text": "...",
  "image_size": [w, h],
  "decode_mode": "hybrid"
}
```

详细 schema 见 `docs/object_token_schema.json`（待定稿）。
