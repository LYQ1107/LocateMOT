# Stage L0-B：ObjectToken 映射验证报告

日期：2026-08-06

## 结论

LocateAnything 最终接受的预测框可以严格一一对应地提取 ObjectToken。36 个样本（9 张图 × 4–5 个查询）中：

- accepted coordinate box blocks = 149
- 最终解析框 parsed boxes = 149
- ObjectToken = 149
- 顺序错配 = 0
- box 错配 = 0
- 映射完整性 = 100%

## 方法与事件流

未修改 `third_party/Eagle`。通过 `locatemot/models/object_tokens/generation_trace.py` 复现官方 generate 循环（已验证与官方 `model.generate` 输出逐字一致），并以 forward hook 捕获最后两层 hidden states。每个生成步骤记录 GenerationBlockEvent：

- MTP/PBD accepted（coord_box）
- MTP rejected（error_box → Hybrid fallback）
- AR/NTP（含 fallback 中逐 token 生成）
- point / empty-None / ref object / end block

## 映射规则

- 每个 accepted coord_box 事件按输出顺序分配 `output_order`。
- ObjectToken 只从 accepted coord_box 生成；rejected MTP、point、None、ref、end 一律不生成。
- MTP block 的 hidden-state 位置 = block 的 6 个输出位置（输入位置相同）。
- Hybrid fallback 中，被拒绝 MTP 的部分 box token 与后续 AR token 合并为一个最终 box；该 box 的 hidden features 按每个 token 的实际输入位置聚合。
- batch 隔离：标准路径 batch=1；官方 batch runtime 2 图 batch 输出验证无串位。

## 结果表

| 项 | 数量 |
|---|---:|
| 验证图片 | 9 |
| 查询样本 | 36 |
| accepted box blocks | 149 |
| parsed final boxes | 149 |
| ObjectToken | 149 |
| 被拒绝 MTP blocks（error_box） | 15 |
| Hybrid fallback 事件 | 15 |
| point 事件 | 8 |
| None 事件 | 16 |
| batch 隔离失败 | 0 |
| 顺序/box 错配 | 0 |
| 映射完整性 | 100% |

## 坐标顺序确认（合成图）

在 640×480 白图上画 GT 红矩形 (100, 200, 300, 400)（像素），查询 "Locate the red rectangle."：

```text
<ref>red rectangle</ref><box><157><416><469><838></box>
```

解析后 ObjectToken box_xyxy = [100.5, 199.7, 300.2, 402.2]。

因此：

- 模型内部 6-token block 顺序 = `[box_start, x1, y1, x2, y2, box_end]`
- 最终文本与 worker parser 均为 xyxy 顺序
- 官方 `decode_bbox_avg`/`handle_pattern` 注释中写 “x1,x2,y1,y2” 是误导性注释；tokenizer 顺序解码不重排，训练标签与解析器定义了实际顺序。

## 特征完整性

- PBD box-end / coordinate-mean / full-block mean：2048 维（Qwen2.5-3B hidden），含 last 与 penultimate 两层。
- MoonViT region：4608 维原始特征（post 2×2 merge）。
- fused：256 维（随机初始化 projection，未训练）。
- 无 NaN/Inf；同一 seed 重复运行特征差 = 0.0。

## 资源

- 36 次推理总耗时 29.0s（不含模型加载），单次中位 0.34s，最长 12.86s（detect_text 截断循环）。
- 峰值显存 8.5GB（COCO 640px 级、bf16、sdpa LLM + eager vision）。
- 输出目录大小 37MB（含事件、tokens、CSV）。

## 限制

- 标准 HF 路径 batch=1；多图 batch 仅在官方 batch runtime 验证，未做 per-image hidden-state 隔离（batch runtime 不暴露 hidden states）。
- detect_text 在少文本图上会循环到 max_new_tokens（记录为异常，不作为 ObjectToken 主路径）。
