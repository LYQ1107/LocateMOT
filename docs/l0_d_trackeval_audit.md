# Stage L0-D TrackEval Audit

生成时间：2026-08-07

## 官方 TrackEval

- 项目：JonathonLuiten/TrackEval
- 官方 URL：https://github.com/JonathonLuiten/TrackEval
- 本地路径：`references/TrackEval-official`
- branch/commit：master @ `12c8791b303e0a0b50f753af204249e622d0281a`（2022-11-29）
- 许可证：MIT
- 已核对实现文件：
  - `trackeval/metrics/hota.py`（HOTA/DetA/AssA/LocA）
  - `trackeval/metrics/clear.py`（MOTA/MOTP/FP/FN/Frag/MT/PT/ML）
  - `trackeval/metrics/identity.py`（IDF1/IDP/IDR/IDSW）
  - `trackeval/eval.py`（Evaluator、COMBINED_SEQ 聚合）
  - `trackeval/datasets/mot_challenge_2d_box.py`（MOT Challenge 数据读取/preproc/similarity）

## 已核对的关键实现细节

### HOTA（hota.py）

- 逐帧按 GT/tracker 对计算 global alignment score，再在每帧做 Hungarian（`linear_sum_assignment(-score_mat)`）。
- alpha 阈值网格 `np.arange(0.05, 0.99, 0.05)`；报告值取 alpha=0.05 的 `HOTA(0)`。
- `DetA = TP / (TP+FN+FP)`，`AssA` 由 GT/tracker ID 对的 Jaccard 平均，`HOTA = sqrt(DetA*AssA)`。
- `LocA` 用匹配对 similarity 的加权平均。
- 多序列聚合：`combine_sequences` 直接对 TP/FN/FP 求和；`AssA/AssRe/AssPr` 按 `HOTA_TP` 加权平均；`LocA` 按 `HOTA_TP` 加权。即官方 COMBINED_SEQ 不是“先算每条序列再平均”。

### CLEAR（clear.py）

- MOTA = 1 - (FN+FP+IDSW)/GT；MOTP 按匹配 similarity 平均；MT/PT/ML 按轨迹被跟踪比例（>=0.8 / >0.2 / <=0.2）。
- 多序列聚合同样先聚合计数（`combine_sequences`）。

### Identity（identity.py）

- IDF1 基于全局 ID 匹配（最长公共子序列/Hungarian），IDSW 计数；`combine_sequences` 按官方实现聚合。

### MOT Challenge 2D Box 数据格式（mot_challenge_2d_box.py）

- GT 与 tracker 均为文本：`frame, id, x, y, w, h, mark, class`（xywh；mark=1 有效）。
- `_calculate_similarities` = box IoU（xywh 解释）。
- preproc 会按 distractor/zero_marked 过滤；本阶段两帧诊断数据全部使用有效类（class=1）、mark=1、无 crowd ignore，因此 preproc 不删除任何检测。

## 本地其他 TrackEval 副本

- `/data1/LWR/vranlee/SERVER_ONLY/avis/TrackEval-master`（无 git 元数据，未固定 commit）
- `/data1/LWR/vranlee/SERVER_ONLY/avis/trackeval`（无 git 元数据）
- MOTIP/MOTIP-2/MeMOTR 内嵌 TrackEval（旧版）

结论：本阶段正式结果一律使用 `references/TrackEval-official`（固定 commit `12c8791b`），不把本地无元数据副本作为正式来源；不自行实现 HOTA/AssA 数学公式。

## 两帧诊断的特殊性

- 每条 held-out pair 构造为独立 2-frame sequence（frame 0 = reference 初始化，frame 1 = current）。
- 所有序列一次性交给官方 Evaluator，报告 `COMBINED_SEQ`（官方 `combine_sequences` 聚合），不按 pair 平均。
- 结果命名统一为 **Two-frame held-out association TrackEval diagnostic**，不得称为 MOT17/DanceTrack/正式 long-video 结果。
