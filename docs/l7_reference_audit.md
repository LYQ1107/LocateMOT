# Stage L7 Reference Audit（2025/2026 官方代码审计）

审计日期：2026-08-14。所有仓库已 clone 到 `references/l7/`，git remote、
commit SHA、license 见下表。本文件只记录实际阅读到的事实，不根据论文摘要转述。

## 仓库清单

| 项目 | 论文/会议 | 官方 URL | 本地路径 | commit SHA | License |
|---|---|---|---|---|---|
| OVTR | OVTR: End-to-End Open-Vocabulary MOT with Transformer, ICLR 2025 | github.com/jinyanglii/OVTR | references/l7/OVTR | 500e72c19bf5f7f8717546911a5639fdc26bfee5 | MIT |
| OVTrack | OVTrack: Open-Vocabulary MOT, CVPR 2023 | github.com/SysCV/ovtrack | references/l7/ovtrack | e188b32eccc049fd425e80b11a3bc45ce88edb31 | Apache-2.0 |
| OVT-B | OVT-B benchmark, NeurIPS 2024 D&B | github.com/Coo1Sea/OVT-B-Dataset | references/l7/OVT-B-Dataset | f033b314c659995936b1d3becd5baf1deb93e121 | Apache-2.0 |
| COVTrack | COVTrack: Continuous OV Tracking via Adaptive Multi-Cue Fusion, ICCV 2025 | github.com/zekunqian/COVTrack | references/l7/COVTrack | 9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b | Apache-2.0 |
| QTrack | QTrack: Query-Driven Reasoning for Multi-modal MOT, arXiv 2603.13759 (2026) | github.com/gaash-lab/QTrack | references/l7/QTrack | bc746fe246217a4de0ecac0318ba1cf9be94a604 | Apache-2.0 |
| TempRMOT | Bootstrapping Referring MOT, arXiv 2406.05039 (2024) | github.com/zyn213/TempRMOT | references/l7/TempRMOT | 6a65640d849fdee4a32bb055945ee34c3b0edeb1 | 无 LICENSE |
| ReaMOT | ReaMOT benchmark/framework, arXiv 2505.20381 (2025) | github.com/chen-si-jia/ReaMOT | references/l7/ReaMOT | 1695160007e57f30e7d758ea087bafe3d649e841 | MIT |
| STORM | STORM: End-to-End RMOT, arXiv 2604.10527 (2026) | github.com/amazon-science/storm-referring-multi-object-grounding | references/l7/storm-referring-multi-object-grounding | 0d87c3ba52a024ffb0ea9c533ec278ae5361f4fa | 无 LICENSE |

## OVTR（ICLR 2025，最近的 OVMOT 碰撞）

已读文件：`ovtr/models/ovtr.py`（1014 行）、`ovtr/models/matcher.py`、
`ovtr/teta/metrics/teta.py`、`ovtr/teta/datasets/tao.py`、README。

观察到的实现：

- MOTR 式 end-to-end Deformable-DETR：track query 逐帧传播，
  单帧 `_forward_single_image` 中 backbone 提取多层特征。
- specification 接口是**固定类别表**：预计算 CLIP text/image embedding
  （`iou_neg5_ens.pth`，1732 个 LVIS/TAO 类别），`self.text_embeddings`、
  `self.image_embeddings` 加载后转置；训练时按类别频次（0.7 次幂采样、
  排除 rare）选出 `select_id`，把对应 text embedding 经 `patch2query`
  线性投影后作为 `text_dict` 注入 transformer；推理时使用全部类别。
- 双分支 decoder：object-feature-alignment 分支 + 分类分支，`loss_align` /
  `loss_align_pre`（text feature 稳定约束）；CIP（category information
  propagation）把类别信息传播到 track query。
- 关联：track query output embedding 相似度（MOTR 式），无持久 memory bank、
  无显式 NEW/NO_MATCH 头、无 learned lifecycle。
- 训练：LVIS 静态图（检测预训练）+ TAO；损失为检测/分类/对齐组合。
- 评估：自带的 TETA 实现（`teta/metrics/teta.py`）：global alignment score
  （Jaccard）+ 逐帧 Hungarian；LocA/AssocA/ClsA，AssocA = match²/(gt+tk-match)。
- 官方表：TAO val TETA(novel) 31.4 / AssocA(novel) 34.5 / TETA(base) 36.6。

与我们的差异：OVTR 的 spec 是固定类别嵌入，closed-set ALL 与 open-vocab 共用
同一检测-跟踪 transformer，但**没有**跨 task 的因果身份动力学核心（无持久
记忆/生命周期/身份转移解码器），也没有 RMOT。其 TETA 代码可作为我们官方
评估协议的核对参考（license MIT，可复用协议思想，不复制实现）。

## OVTrack（CVPR 2023，OVMOT 经典基线）

已读：README、`tools/convert_datasets/create_tao_v1.py`、目录结构。

- 两阶段：MMDet 检测器 + quasi-dense embedding head 关联。
- 关联 embedding 用 CLIP 视觉特征蒸馏；`create_tao_v1.py` 把 TAO 类别映射到
  LVIS v1 category id（synset 对齐），生成 `validation_ours_v1.json`，
  即 OVMOT 官方评估使用的 annotation。
- 数据幻觉：扩散模型生成样本训练 embedding（static-image-only training）。
- 评估：TETA（LocA/AssocA/ClsA），Base/Novel 由 LVIS frequency 决定。

对我们的意义：TAO OVMOT 协议以 `validation_ours_v1.json` + TETA 为准；
我们可直接用官方转换脚本处理本地 TAO `validation.json`，不改原始数据。

## COVTrack（ICCV 2025，cue-reliability 直接碰撞）

已读：README、`ovtrack/models/roi_heads/ovtrack_roi_head.py`
（MCF forward，约 1540-1580 行）、configs、`tools/convert_datasets/create_tao_v1.py`。

观察到的实现（Multi-Cue Adaptive Fusion）：

- 主特征 = appearance association embedding；辅助特征 = bbox/location、
  classification/semantic，各经 `bbox_transform` / `cls_transform` 投影。
- gate network 输入 `[assoc, bbox, cls]` 拼接，输出 2 个 gate 值（intra-frame
  confidence）；gate 再乘检测 box conf / cls conf × 0.5。
- 残差融合：`enhanced = assoc + residual([assoc, gated_bbox, gated_cls]) * ratio`，
  `ratio = max_fusion_ratio * (1 - assoc_conf)^2`（inter-frame self-correction）。
- 训练在 C-TAO（TAO 连续帧补全标注，ctao_base.json）上。

碰撞结论：COVTrack 已经公开实现了“association-event-level 的 adaptive
appearance/motion/semantic fusion + 置信度门控”。因此我们**不得**把
“adaptive multi-cue fusion / cue confidence gating”写成第一创新。我们的
Dance repair 必须位于 UIDM identity-transition decoder 内部（持久 track state、
集合交互、continue/NEW/NO-MATCH 决策级可靠性），并明确与 COVTrack
“association-embedding 空间的特征门控 + 余弦匹配”区分；若不成立则调整 claim。

## QTrack（2026，query-driven RMOT 碰撞）

已读：README、`training_scripts/`、`verl/utils/reward_score/vision_reasoner.py`、
`tapo_training.sh`。

- 3B 多模态 VLM（LLaMA/Qwen 类），natural-language query 条件化多目标跟踪。
- RL 训练：verl（Ray+FSDP），TPA-PO（Temporal Perception-Aware Policy
  Optimization），reward = thinking/segmentation format + IoU/L1/point 结构化
  奖励（`vision_reasoner.py`），目标是 motion-aware reasoning。
- 新 RMOT26 benchmark，指标 MCP/MOTP/CLE/NDE；官方 0.30 MCP / 0.75 MOTP。

碰撞结论：QTrack 是 query-conditioned RMOT，但机制是大 VLM 推理 + RL 结构化
奖励，不是小型共享因果身份动力学 core，也不做 closed-set ALL 与 OV 共享。
我们把 TPA-PO 记入 `docs/future_rl_reference.md` 级别的 RL 参考（本阶段不执行
RL）。它与我们“同一 identity core + 轻量 spec encoder”路线机制不同。

## TempRMOT（2024，RMOT）

已读：README、`models/transrmot_pro.py`、`models/memory_bank.py`。

- MOTR 基础；frozen RoBERTa 文本编码器 + VisionLanguageFusionModule
  （cross-attention 于文本词特征，`forward_text`）。
- MemoryBank：`temporal_attn` 在历史 embedding 序列（mem_bank, max_his_length）
  上做 attention，逐帧 `save_proj` 更新；另有 SpatialTemporalReasoner
  （hist_len=4，帧间时空推理）。
- Refer-KITTI / Refer-KITTI-V2，HOTA 52.21 / 35.04。
- 无 LICENSE（不可复制代码，只借鉴接口思想）。

## STORM（2026，end-to-end RMOT）

已读：README、`storm-bench.json`（结构）。

- 本仓库只发布 STORM-Bench 数据（VidOR 80 类，train 6009 / test 714 clips，
  xyxy bbox、referring_expressions，30,700 表达），模型代码未发布。
- 论文模型是 end-to-end 多模态 LLM，joint grounding+tracking，无外部
  detector/tracker；66.7 HOTA / 78.3 IDF1。

## ReaMOT（2025）

- 本仓库当前只有 README + asset，无模型代码（README 声明接受后一周内开源）。
- 属于 reasoning-based RMOT 后续工作；本阶段不实现 reasoning 任务，仅记录。

## OVT-B（NeurIPS 2024 D&B）

- 基准数据：7 个视频数据集合并（AnimalTrack、GMOT-40、ImageNet-VID、LVVIS、
  OVIS、UVO、YouTube-VIS-2021），Base/Novel 类别 split，TETA 评估。
- 数据经百度网盘/Google Drive 分发，服务器当前无数据且磁盘紧张，
  本阶段主用 TAO（本地已完整），OVT-B 记为候选、不下载。

## 复用决定

- 评估协议（TETA + TAO `validation_ours_v1.json`）：采用官方定义与转换思路，
  自写等价评估/转换脚本，不复制官方代码（OVTR 是 MIT、OVTrack 是 Apache-2.0，
  协议本身可核验后实现）。
- 模型机制：所有上述架构仅阅读借鉴，未复制代码；Dance repair 与
  specification encoder 为本项目 clean reimplementation。
- 未复用：COVTrack MCF（机制已公开，避免撞车）、TempRMOT（无 license）。

