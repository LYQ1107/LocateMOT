# Stage L1-C Association-Controlled Protocol

日期：2026-08-09。目的：让不同 association 方法在完全相同的 detection 输入
下只改变 track ID，从而把 AssA/IDF1/IDSW 的差异归因于 association，而不是
detection/lifecycle。

## 1. 固定候选集（Fixed Candidate Manifest）

- 来源：L1-A DanceTrack cache（LocateAnything-3B 冻结，commit 783f656d，
  protocol `person`）+ L1-B 多数据集 cache（protocol `pilot`）。
- 产物：`outputs/l1_c/fixed_candidate_manifest/{dataset}_{split}.jsonl` +
  `.manifest.json`。
- 每帧记录：candidate boxes（xyxy）、gen_score、GT ids/boxes、
  matched_candidates、cache 路径、协议、image_size。
- 完整性：每帧 `entry_sha256 = sha256(video, frame, boxes, scores)`；
  每个 manifest 有 `total_sha256`。任何方法不得修改 boxes/scores。
- 输出置信度：TrackEval 输出使用候选 gen_score（不允许固定 1.0，
  否则等分场景下按 ID 顺序破平局会引入 DetA/LocA 抖动）。
- 已生成 manifest：

| dataset/split | frames | videos | sha256(前16) |
|---|---|---|---|
| dancetrack_train | 34,046 | 32 | c76fe7eadca102a0 |
| dancetrack_calibration | 8,024 | 8 | 1bc5111ace2c07f4 |
| dancetrack_val | 25,508 | 25 | 412720885a0bb5d3 |
| bdd100k_train | 8,001 | 200 | 823ba931197ef6df |
| tao_amodal_train | 4,200 | 105 | ab90ef1727b14fb2 |
| mot17_train | 240 | 3 | af890d8d2b393d42 |
| mot20_train | 160 | 2 | 41fbc6ed73a136f5 |

## 2. 有效性条件

同一 protocol 下，任意两个 association 方法必须满足：

- box coordinates hash 一致；
- score hash 一致；
- prediction count（输出 track 数/帧）一致；
- frame count 一致。

此时 TrackEval 的 DetA / LocA / FP / FN 必须相同（只允许浮点舍入差异）。
若不一致，该 association-controlled 实验非法，不能用于方法比较。

## 3. 共享 Lifecycle 与 NEW 规则

- 所有方法共用 `OnlineTracker` birth/lifecycle shell：
  `min_hits=3`（TENTATIVE→ACTIVE）、`max_age=30`、未匹配 candidate → NEW
  （新 track_id，顺序递增），未匹配 track → lost_age++ → TERMINATED。
- 这些规则对所有方法（C0–C4、UA、UAL）一致；差异只来自 assignment。

## 4. 主评估表（DanceTrack val，TrackEval official）

| variant | 定义 |
|---|---|
| C0 | IoU Hungarian（阈值 0.3） |
| C1 | OC-SORT 风格 motion（7 维 Kalman + OCM 第二轮） |
| C2 | raw PBD cosine（box-end last，Hungarian） |
| C3 | IoU + raw PBD 固定融合（权重/阈值需 calibration） |
| C4 | 冻结 B6 local kernel（L0-D checkpoint） |
| UAF | Frozen LocateAnything + Unified Association Decoder |
| UAL | LoRA LocateAnything + 同构 UA Decoder |

主指标排序：AssA > IDF1 > IDSW > HOTA；同时报告
IDSW/1000 GT detections。

## 5. 校验脚本

- 构建：`tools/build_l1c_fixed_manifest.py`
- 运行方法：`tools/run_l1c_tracker.py`
- 官方评估：`tools/run_l1c_trackeval.py`
- 输出：`outputs/l1_c/association_controlled_main.csv`、
  `association_controlled_trackeval.json`、
  `association_controlled_per_seq.csv`
