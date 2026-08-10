# Stage L1-B Road Multi-Class LODO Report

## 设置

- 缓存：BDD 8,001 帧（200 视频）+ TAO-Amodal 4,200 帧（100 视频，
  含 pilot 重叠），LocateAnything-3B 冻结，candidate→GT 逐帧最大 IoU。
- 训练：Identity Adapter（InfoNCE，full 输入），seed=20260806，30 epochs，
  per-dataset cap 200；身份数 BDD 7,556 / TAO 285 / 合并 7,841。
- 评估：Same-Category R@1/mAP，BDD gallery（207 queries）、TAO gallery
  （253 queries）。
- 协议：A_bdd（只训 BDD）、A_tao（只训 TAO）、A_road（合并训练）；
  跨数据集评估 = Leave-One-Dataset-Out。

## Same-Category R@1（核心表）

| gallery | raw best | A_bdd | A_tao | A_road |
|---|---:|---:|---:|---:|
| bdd100k | 0.831 (R1) | 0.681 (−15.0pp) | 0.671 (−15.9pp, unseen) | 0.744 (−8.7pp) |
| tao_amodal | 0.822 (R0) | 0.747 (−7.5pp, unseen) | 0.779 (−4.3pp) | 0.862 (+4.0pp) |

mAP 与查询数见 outputs/l1_b/same_category_retrieval_lodo_{raw,bdd,tao,road}.csv。

## 结论

1. **in-domain 不成立**：A_bdd 在 BDD 上比 raw 低 15.0pp；A_tao 在 TAO 上
   低 4.3pp。只有合并训练 A_road 在 TAO 上 +4.0pp。
2. **LODO 不泛化**：A_bdd→TAO −7.5pp；A_tao→BDD −15.9pp。两个方向在
   unseen dataset 上均退化，说明 adapter 仍学习 dataset-specific
   shortcut，而非 universal identity representation。
3. **road multi-class 方向未通过**：即使把 BDD/TAO 缓存扩到 12k 帧、
   身份数到 7.8k，Identity Adapter 仍不能稳定超过 raw PBD。

## Stage 更新

`L1_B_IDENTITY_SIGNAL_NOT_SUPPORTED` 维持不变；本 LODO 实验进一步排除
“数据规模不足”假设（v1 704 身份 → LODO 7.8k 身份，结论未反转）。
