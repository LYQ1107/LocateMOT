# Stage L0-A：官方 LocateAnything-3B 最小复现报告

日期：2026-08-06

## 结论

官方 LocateAnything-3B 已在 A100 上成功加载并完成最小推理复现。使用官方默认采样参数时，单目标 grounding、多类别检测、负样本 none 均正常；批量/密集场景与文本检测存在已知的退化现象（详见观察）。

## 环境与模型

- 独立环境：`/home/lwr/anaconda3/envs/locatemot`（Python 3.12 venv）
- PyTorch：2.5.1+cu124；transformers：4.57.1；peft：0.12.0；accelerate：1.5.2
- GPU：A100-SXM4-40GB（本测试使用 GPU 8，复现时为空闲）
- 模型本地路径：`/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/models/LocateAnything-3B`
- 模型仓库 commit（Eagle）：`783f656d127ee498137b5ff52603ce36c292d317`
- checkpoint SHA256：
  - model-00001-of-00002.safetensors: `923cfc10fed19808067da6df85a9a4220ddc1f9eb91ceee94c0fecd05d0f2d58`
  - model-00002-of-00002.safetensors: `3459ba101f40594f3f62d3312014f1f8378b4ba3da3b1d562480045938fc7d47`
- 加载参数：bf16、`attn_implementation="sdpa"`（A100 无 MagiAttention，自动回退 sdpa；MoonViT flash_attn 未安装时回退 sdpa）
- 生成模式：hybrid（官方默认），max_new_tokens=2048，temperature=0.7，top_p=0.9，repetition_penalty=1.1

## 测试输入

- 6 张 COCO val2017 图像（640×426 至 375×500，来自 images.cocodataset.org）
- 固定 6 类查询：ground_single、ground_multi、detect（person/car/bicycle）、detect_text、point（traffic light）、negative（purple elephant）
- 共 36 个推理样本，原始结果保存在 `outputs/l0_a_reproduction/raw_outputs.jsonl`

## 结果摘要

| 查询 | 正常终止 | 典型结果 |
| --- | --- | --- |
| ground_single | 6/6 | 每图 1 个框 |
| ground_multi | 6/6 | 有红衣人时 2 框，无则 none |
| detect | 6/6 | 有目标时多框，无则各类别 none |
| detect_text | 4/6 正常 | 1–12 个文本框；2 张图退化循环（truncated） |
| point | 6/6 | 多数 none；1 张无红绿灯图输出 3 个 point（幻觉） |
| negative | 6/6 | 全部 `<box>None</box>` |

详细数值见 `outputs/l0_a_reproduction/l0_a_runtime.csv` 与 JSONL。

## 运行时间与显存

- 简单查询（ground_single/detect/negative）：0.2–1.8 s
- 密集/文本查询（正常）：0.3–2.2 s
- detect_text 退化循环：27.8–39.7 s（触达 max_new_tokens）
- 峰值显存：8.1–8.7 GB（单图、640×426 级别、bf16、sdpa）
- 图像尺寸：与推理耗时正相关（更大图像更多视觉 token）

## 关键观察

1. 必须使用官方默认采样参数（temperature=0.7、top_p=0.9、repetition_penalty=1.1）。greedy（temperature=0）下模型在 MTP 路径反复输出同一 box，无法终止，直到 max_new_tokens 截断。
2. 官方文档写的 `none` 实际输出为 `<box>None</box>`（大写）；解析时需大小写不敏感。
3. detect_text 在文本极少/无文本的图上可能陷入“逐像素文本框”循环并截断；这不是 OOM 而是生成路径问题，需要记录为失败案例。
4. point 查询在无对应目标时可能输出幻觉点（1/6 图），说明 point 模式没有严格 none 约束。
5. `magi_attention not available, falling back to sdpa` 为预期行为（A100 无 MagiAttention）。
6. HF 远程 `generate_utils.py` 与本地 Eagle commit 完全一致；`modeling_locateanything.py` 仅差 LoRA clone 一行，可视为同一实现。

## 复现判定

- 官方模型加载：通过
- 单目标文本 grounding：通过
- 多目标检测：通过（含负样本 none）
- 密集目标检测：未做专门密集图（当前 6 图无密集场景）；detect_text 退化已记录
- point 输出：解析通过，但存在幻觉现象
- batch inference：尚未验证（batch runtime 需要 HF 仓库 `batch_utils/`，已随模型下载，后续测试）

下一步：Stage L0-B PBD hidden-state 映射与 ObjectToken 提取。
