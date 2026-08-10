# Stage L1-B Final Report

## 1. Executive Summary

Stage L1-B 验证“LocateAnything ObjectToken → Universal Identity Adapter →
Persistent Identity Token”方向。pilot（v1：1,860 帧 / 6 数据集 / 704 身份；
v2 加入 BDD100K：2,932 帧 / 1,064 身份）结果：
Identity Adapter 未能在 Same-Category Retrieval 上稳定超过最佳 raw
ObjectToken 基线。**Stage Decision: L1_B_IDENTITY_SIGNAL_NOT_SUPPORTED**。

## 2. Why L1-A Failed

L1-A 事实：frozen B6 在 DanceTrack full-video 上 HOTA 0.401→0.247，
trajectory/motion/memory/reactivation（T6）把 DetA 提到 0.888 但 AssA 仅
0.227、IDSW 5962；reactivation id-kept 仅 6.3%。结论：raw/frozen token 与
temporal 堆叠都不是 persistent identity 的正解，L1-B 改为先解决
identity representation 本身。

## 3. Unified MOT Goal

一个 checkpoint、一个 Identity Adapter、无 dataset-specific head，覆盖
single-category dense、multi-category long-tail、deformable/sparse。
不是 person ReID。

## 4. 2025–2026 Literature / GitHub Audit

- OG-ReID（CVPR 2026）：NO VERIFIED OFFICIAL IMPLEMENTATION（仓库无 refs）
- VICP（ICCV 2025）commit 5ae97924（无 LICENSE）：in-context prompt ReID，
  参考 prompt-conditioning 概念
- UPCL（NeurIPS 2025）commit c2c01c2b（无 LICENSE）：统一多类别 ReID，
  参考共享 embedding + prototype/ID loss
- UniTrack（2026）commit afdd9869（无 LICENSE）：图式 identity consistency，
  关联对照
- 复用 L1-A 审计：FDTA（Identity Contrastive）、MOTIP/MOTIP-2（ID
  prediction）
- 详情：docs/l1_b_reference_audit.md

## 5. Dataset Audit

真实统计见 docs/l1_b_dataset_identity_audit.md 与
outputs/l1_b/dataset_statistics.json：
DanceTrack（train 419 id）、MOT17（train 546 id）、MOT20（train 2215 id）、
YT-VOS（train 6459 id）、MOSE train（7631 id）、TAO-Amodal（train 500
videos / 54639 anns）、BDD100K（masa box_track_20 tracking 标签 + 本地图像：
train 200 视频 / 39,418 帧 / 15,558 身份 / 11 类）、TAO official
（MISSING_PUBLIC，TAO-Amodal 为可用替代）、MOTSynth（禁用）。

## 6. Unified Identity Supervision Schema

见 docs/l1_b_unified_schema.md：identity relation 三元组、temporal gap
分桶、sparse supervision mask、REAL_CANDIDATE_OBJECT_TOKEN 主来源 +
GT_ROI_FEATURE 仅作 diagnostic。

## 7. Data Selection

- Train（pilot）：DanceTrack/MOT17/MOT20/TAO-Amodal/YT-VOS/MOSE train
- Eval-only：各 test/valid（无公开 GT 或隐藏标注）
- Not used：MOTSynth（规格禁止）、BDD100K（本地无 tracking 标签）、
  official TAO（缺失，TAO-Amodal 替代）

## 8. Pilot Dataset Statistics

pilot v1：1,860 帧，704 可用身份（DanceTrack 62 / MOT17 166 / MOT20 340 /
TAO 23 / YT-VOS 44 / MOSE 69），22,437 观测；v2：+BDD 240 帧，缓存超集
2,932 帧，1,064 身份（BDD 275 / YT-VOS 79 / MOSE 119），26,367 观测。详情：
reports/l1_b_pilot_data_report.md。

## 9. ObjectToken Baselines

Same-Category R@1（同一 query/gallery 协议）：

| dataset | R0 PBD-box-end | R1 PBD-coord | R2 region | R3 fused |
|---|---:|---:|---:|---:|
| dancetrack | 0.919 | 0.855 | 0.016 | 0.613 |
| mot17 | 0.946 | 0.970 | 0.151 | 0.825 |
| mot20 | 0.869 | 0.774 | 0.384 | 0.495 |
| tao_amodal | 0.870 | 0.870 | 0.609 | 0.739 |
| ytvos | 0.750 | 0.591 | 0.250 | 0.568 |
| mose | 0.551 | 0.435 | 0.087 | 0.377 |

## 10. Universal Identity Adapter

locatemot/models/identity/identity_adapter.py：d=256，PBD box-end +
coordinate + region + geometry + gen 投影后融合（full）；消融变体
pbd-only。共享参数，无 dataset head。

## 11. Identity Objectives

主目标：InfoNCE（temperature=0.1，1 positive + 显式 negatives）。
无多余 loss 堆叠。

## 12. Hard Negative Strategy

训练负样本优先 same-video 其它 identity，再 cross-video；same-category
为主要负样本（person 数据集天然 same-category，TAO/YT-VOS 按类别查询）。

## 13. Sampling Strategy

dataset-balanced：每 epoch 每数据集最多 60 个身份；identity 内
anchor+positive 时间排序首两观测；单 seed 20260806。

## 14. Pilot Training Setup

GPU 8（40GB A100）；20–30 epochs；batch 32 identities；LR 1e-3 AdamW；
参数 ~2.6M；walltime ~15s/run（feature-level cache）；VRAM <2GB；
RSS 低（safetensors 按需读取）。

## 15. Identity Retrieval Main Results

完整 ROC-AUC/PR-AUC 见 outputs/l1_b/raw_token_retrieval.csv 与
same_category_retrieval.csv。R4（full）：dancetrack 0.903 / mot17 0.934 /
mot20 0.862 / tao 0.826 / ytvos 0.705 / mose 0.493（Same-Cat R@1）。

## 15b. v2 Retrieval（含 BDD）

| dataset | best raw R@1 | R4 full R@1 | R4 pbd R@1 |
|---|---:|---:|---:|
| dancetrack | 0.919 | 0.984 | 0.935 |
| mot17 | 0.970 | 0.934 | 0.934 |
| mot20 | 0.869 | 0.822 | 0.801 |
| bdd100k | 0.740 | 0.770 | 0.640 |
| tao_amodal | 0.870 | 0.957 | 0.783 |
| ytvos | 0.747 | 0.734 | 0.722 |
| mose | 0.588 | 0.571 | 0.580 |

macro best-raw 0.815 → R4 full 0.825（+1.0pp）。

## 16. Same-Category Identity Results（核心表）

| dataset | best raw R@1 | R4 full R@1 | R4 pbd R@1 |
|---|---:|---:|---:|
| dancetrack | 0.919 | 0.903 | 0.968 |
| mot17 | 0.970 | 0.934 | 0.910 |
| mot20 | 0.869 | 0.862 | 0.845 |
| tao_amodal | 0.870 | 0.826 | 0.826 |
| ytvos | 0.750 | 0.705 | 0.659 |
| mose | 0.551 | 0.493 | 0.551 |

## 17. Dataset-wise Results

见 §16 与 reports/l1_b_identity_retrieval.md。

## 18. Raw ObjectToken vs IdentityToken

v1：IdentityToken 只在 DanceTrack 超过 raw；v2（含 BDD，更多数据）：
full 在 DanceTrack +6.5pp / BDD +3.0pp / TAO +8.7pp 提升，其余 4 个数据集
下降。macro 平均（v2）raw 0.815 → R4 full 0.825。

## 19. Association-Controlled Protocol

要求：所有方法输出同一 detection set，仅 track ID 可变化（DetA/LocA/FP/FN
一致）。因 retrieval pilot 未通过，full-video association-controlled 未执行
（按规格 pilot fail 不强制进入）。

## 20. Association-Controlled Results

未执行（gate 未通过）。A3/A4（IdentityToken cosine / +B6 relation）留待
representation 信号成立后测试。

## 21. Leave-DanceTrack-Out

未执行（LODO 改为 road multi-class 方向验证，见 §21b；DanceTrack LODO
保留给后续）。pilot gate 未通过，按规格未进入原 LODO 路线。

## 21b. Road Multi-Class LODO（继续阶段，已完成）

BDD 8,001 帧 + TAO-Amodal 4,200 帧缓存；A_bdd / A_tao / A_road 训练
（7,556 / 285 / 7,841 身份）。Same-Category R@1：

| gallery | raw best | A_bdd | A_tao | A_road |
|---|---:|---:|---:|---:|
| bdd100k | 0.831 | 0.681 | 0.671 (unseen) | 0.744 |
| tao_amodal | 0.822 | 0.747 (unseen) | 0.779 | 0.862 |

结论：in-domain 与 LODO 均不通过；adapter 仍学 dataset shortcut。
详见 reports/l1_b_lodo_report.md。

## 22. Leave-Multiclass-Dataset-Out

已执行（road LODO）：A_tao→BDD −15.9pp、A_bdd→TAO −7.5pp，均相对 raw
退化，未通过。

## 23. Cross-Dataset Generalization

Pilot 内观察：v1 只在 DanceTrack 有效；v2 扩大到 road multi-class
（BDD/TAO）有效，但 dense person（MOT17/MOT20）与 deformable
（YT-VOS/MOSE）仍不迁移——跨 family 方向不一致证据。

Road LODO（12k 帧 / 7.8k 身份）：跨数据集两个方向都相对 raw 退化，
排除“数据规模不足”假设。

## 24. Full Unified Training

未执行。原因：pilot gate 失败（L1_B1_IDENTITY_SIGNAL_NOT_SUPPORTED）。

## 25. Identity Drift

未执行（pilot 未通过）。pilot cache 已具备计算 drift 的基础，留给后续。

## 26. Same-Class Dense Stress Test

DanceTrack/MOT17/MOT20 的 same-category 表见 §16；raw 已接近协议上限。

## 27. Multi-Class / Long-Tail Stress Test

TAO-Amodal（23 queries）与 YT-VOS（44 queries）pilot 统计不稳定；R4 未
提升，不满足 multi-class 推广证据。

## 28. Failure Analysis

见 reports/l1_b_failure_analysis.md 与 l1_b_lodo_report.md。要点：
raw PBD 已强；region 稀释；v1→v2→road LODO（704→1,064→7,841 身份）
逐级扩大数据均未反转结论；adapter 学 dataset shortcut。

## 29. Resource Usage

pilot cache：1,860 帧 × ~2s = 约 62 GPU-min（4 卡并行 ~16min）；
训练 <1 GPU-min；检索 <1min CPU；磁盘 <5GB（/data3）。

## 30. Scientific Interpretation

ObjectToken 的 PBD 特征已经携带相当强的 instance 判别信息（dense 场景
R@1 0.87–0.97）；当前 Identity Adapter 学到的变换是 dataset-specific 的
（v2 对 road multi-class 与 DanceTrack 表面有效，但 road LODO 证明是
dataset shortcut），没有达到“universal identity”要求。问题不在“缺少
投影”或“数据规模”，而在训练目标/分布与真实跨数据集身份判别不匹配。

## 31. Claim Boundary

可以说：pilot 协议下 Identity Adapter 不优于 raw PBD；PBD 是强 raw
身份特征；region/fused 未训练版不具备 top-1 身份判别力。
不能说：Identity Adapter 方向整体无效（pilot 规模小）；不能说任何
SOTA/benchmark 结论。

## 32. Stage Decision

`L1_B_IDENTITY_SIGNAL_NOT_SUPPORTED`（pilot gate 未通过）。

## 33. Next Recommended Stage

唯一主要建议：转向检测/候选质量与 association 端（raw PBD 已强，Identity
Adapter 在多种规模与协议下均无法稳定超越 raw）；若仍要研究 representation，
需先解决 hard-negative 协议与 dataset-specific shortcut 问题，而不是继续
扩大数据。

## 34. Important Paths

- docs/l1_b_reference_audit.md、docs/l1_b_dataset_identity_audit.md、
  docs/l1_b_unified_schema.md
- outputs/l1_b/dataset_statistics.json、raw_token_retrieval.csv、
  same_category_retrieval.csv、checkpoints/
- configs/l1_b/pilot_videos.json
- reports/l1_b_storage_plan.md、l1_b_pilot_data_report.md、
  l1_b_identity_retrieval.md、l1_b_same_category_analysis.md、
  l1_b_failure_analysis.md
- tools/l1_b_dataset_audit.py、build_l1b_pilot_split.py、
  cache_l1b_locateanything.py、eval_l1b_retrieval.py、
  train_l1b_identity_adapter.py
