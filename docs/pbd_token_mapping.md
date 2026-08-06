# PBD Box ↔ Object Token 映射（已验证）

状态：Stage L0-B 已通过实测验证（2026-08-06）。详见 `reports/object_token_validation.md`。

## 依据（官方代码 + 实测）

- 官方 `generate`（`Embodied/eaglevl/utils/locany/modeling_locateanything.py`）的 MTP 路径：
  `next_token_logits = outputs.logits[:, -n_future_tokens:, :]`，`n_future_tokens=6`。
- block 为 6 个 token：`[box_start, c1, c2, c3, c4, box_end]`。
- 四个坐标位置的语义 = 最终文本顺序 = `<box><x1><y1><x2><y2></box>`。
  - 官方 `decode_bbox_avg` docstring 与 `handle_pattern` 注释写 “x1,x2,y1,y2”，与训练标签/worker parser 冲突；tokenizer 顺序解码不重排，实测以训练标签与 parser 为准。
  - 合成图实测：GT 矩形 (100,200,300,400)（像素）→ 模型输出 `<box><157><416><469><838></box>` → ObjectToken box_xyxy=[100.5,199.7,300.2,402.2]。
- `handle_pattern`：coord_box=6 tokens、point_box=4 tokens、empty_box=`[box_start, none, box_end]`、ref_object=语义 block。

## 映射规则（已实现并验证）

1. 每个 accepted coord_box 事件按输出顺序分配 `output_order`（即最终文本中 `<box>` 的出现顺序）。
2. block_start/block_end = 该 block 在输出 token 序列中的绝对位置。
3. PBD hidden states：
   - MTP：取 block 6 个输出位置对应的输入 hidden states（last 层 + penultimate 层）；
   - AR（Hybrid fallback）：逐 token 取“预测该 token 的输入位置”的 hidden state，合并为完整 box。
4. 特征变体：box_end（位置 5/最后一个 token）、coordinate mean（位置 1–4 均值）、full-block mean（6 个位置均值）。
5. MoonViT region：post 2×2 merge 特征网格 `(H/28, W/28)`，归一化 box 映射后做框内 mean pooling。
6. fused：PBD 2048 + region 4608 + geometry(5) + generation score → 随机初始化 Linear → 256。

## 验证结果

- 36 样本 / 9 图：accepted=149 = parsed=149 = ObjectToken=149；order/box 错配 0；完整性 100%。
- rejected MTP block（15 次 error_box）与 point/None/ref/end 均不产生 ObjectToken。
- Hybrid fallback（15 次）最终只保留被接受路径的 box。
- 同一 seed 重复运行：answer、box、hidden 位置、特征完全一致（最大差 0.0）。
- batch=1 标准路径；官方 batch runtime 2 图验证输出不串位。

## 与训练侧 block 的关系

- 训练 `block_size=6`、`causal_attn=False`：block 内互相可见、block 间因果；ObjectToken 聚合 block 内 hidden states 与训练注意力一致。
- 未来如需更“干净”的对象 token，可在训练时额外加对象 token 头；Stage L0 用现有 hidden states 已验证足够支撑接口与初步身份判别。

## ObjectToken 输出结构

见 `docs/object_token_schema.json`（已定稿）。关键字段：

- box_xyxy（像素）、normalized_box（0–1）
- pbd_box_end_feature / pbd_coordinate_mean_feature / pbd_full_block_mean_feature（2048 维，含 penultimate 版）
- region_feature（4608 维）
- geometry_feature（5 维）、generation_score、fused_feature（256 维）
- confidence_feature：null（官方没有可解释的 box 置信度）
- block_start/block_end、decode_mode、source_frame、model_commit、checkpoint_hash
