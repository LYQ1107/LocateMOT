# Stage L0-C：ObjectToken cache 报告

## 规模

- 视频：train 400 + calibration 80 + held-out 150 = 630
- 每视频帧数：4（0/3/10/80 位置的最近实际帧）
- 协议：category_guided（YouTube）+ generic（YouTube+MOSE）
- 总 job：3780，完成 3780（100%）
- cache 大小：429 MB（float16 safetensors）

## 格式

- `dataset/video/frame/protocol.safetensors` + `.meta.json` + `.complete`
- 原子写入：`.tmp` → rename → `.complete`；缺失 `.complete` 自动重跑补齐。
- 特征：pbd_coord_mean_last/penultimate、pbd_box_end_last、region、geometry、gen_score、boxes、crop_region。

## 时间与资源

- 3 个 GPU shard（当时空闲 GPU 1/2/3），每 shard ~1200 job；峰值显存约 13–18GB（4096 token 预算）。
- 无 token 帧（模型输出 none）写入空 shard 标记，避免重复运行。
- 修复记录：MTP error_block 的 AR 合并、空样本 `.complete`、无类别帧空 cache。

## Candidate recall（由 cache 统计）

| 协议 | recall@0.3 | recall@0.5 | recall@0.7 | GT 对象 |
|---|---:|---:|---:|---:|
| category_guided | 0.898 | 0.862 | 0.801 | 2102 |
| generic | 0.589 | 0.528 | 0.469 | 4202 |

结论：category-guided 候选质量明显高于 generic；generic 是主要候选瓶颈。
