# Stage L1-C LoRA PBD Extraction（根因与解决）

## 现象

- `tools/cache_l1c_lora.py` + `ObjectTokenExtractor` 单帧长时间不返回；
- 直接 `model.generate` 正常。

## 根因 1：PEFT unwrap 死循环（已修复）

`generation_trace.py` 中的
`while hasattr(lm, "base_model"): lm = lm.base_model` 在
`merge_and_unload()` 后的 Qwen2ForCausalLM 上不终止：普通
Qwen2ForCausalLM 也暴露 `base_model` 属性（指向内部 Qwen2Model），
形成自引用链。修复：只在 `PeftModelForCausalLM` / `LoraModel` 类型上 unwrap。

## 根因 2：LoRA 训练数据 PBD 格式错误（已修复）

官方 `_BOX_RE` 要求 `<box><x1><y1><x2><y2></box>`（特殊 token
`<0>`–`<1000>` 格式）；我们第一版 JSONL 写成了字面量
`(x1,y1,x2,y2)`。这导致 LoRA 模型输出普通文本 box 而非 PBD token，
instrumented 解析到 0 个 accepted box。修复 `tools/build_l1c_lora_data.py`
后重新生成 27.8k 条 JSONL 并重训 LoRA 300 步（train_loss 1.94）。

## 解决后的提取验证

- 单帧（dancetrack0083 frame 1）提取 22 个 PBD boxes，约 9s/帧；
- 批量提取（7 GPU 分片）DanceTrack calibration 8,024 帧全部完成，
  平均约 3s/帧；
- 缓存校验：8,024 metas，pbd/region 维度正确（2048/4608），feature 全部
  finite。

## Frozen 等价性

- 同一帧（dancetrack0004 frame 1）当前 frozen extractor vs L1-A 旧缓存：
  boxes IoU=0.998（仅浮点差异）；PBD box-end / coord hidden cosine=1.0；
  两次重复运行输出完全一致（max diff=0.0）。
- 结论：当前 extractor 路径可复现旧 Frozen extractor。

## 结论

- `LORA_PBD_EXTRACTION_SUPPORTED`（满足等价性/稳定性/finite/resume）。
- 之前“阻塞”判定作废；根因是数据格式与 unwrap bug，不是生成路径本身。
