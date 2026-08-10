# Stage L1-C Frozen vs LoRA PBD

评估单位：DanceTrack calibration（8 视频，8,016 帧有效帧，68,761 GT
detections；60,805 正样本对）。同一 query、同一 GT 匹配、同一 cosine 协议。

## 1. PBD 表示判别力

| 指标 | Frozen PBD | LoRA PBD (300步) |
|---|---:|---:|
| same-ID cosine 均值 | 0.9775 | 0.8900 |
| diff-ID (same-category) cosine 均值 | 0.9244 | 0.8533 |
| ROC-AUC (same vs diff) | 0.4629* | 0.3167* |
| PR-AUC | 0.5695 | 0.3302 |
| Same-Category R@1 | 0.9222 | 0.4349 |
| mAP | 0.9438 | 0.5800 |

*注：ROC-AUC 使用原始余弦分数；DanceTrack 同类候选余弦普遍很高，
分布重叠导致 AUC 偏低，R@1/mAP 更能反映排序质量。

## 2. 全视频 Association-Controlled（DanceTrack calibration）

| method | DetA | AssA | IDF1 | IDSW |
|---|---:|---:|---:|---:|
| Frozen raw PBD (C2) | 0.944 | 0.140 | 0.305 | 4,190 |
| LoRA raw PBD (L-PBD) | 0.801 | 0.042 | 0.097 | 25,414 |

## 3. 结论

- LoRA（300 步 grounding 适配）显著降低 PBD identity 判别力
  （R@1 −48.7pp，mAP −36.4pp）；
- LoRA 候选质量也下降（Recall@0.5 0.98→0.80，DetA 0.944→0.801）；
- 分类：`LORA_PBD_DEGRADED`（PBD 全面下降 + grounding 下降，
  同时满足 `LORA_TRACKING_GAIN_WITH_FORGETTING` 的遗忘侧）。
- 科学结论：当前短程 LoRA grounding 适配不值得用于 Unified MOT 主线；
  Frozen LocateAnything 表示更优。
