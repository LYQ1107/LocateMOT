# L82 失败分解

## 最终状态

`rank_representation_gate_fail`。这是 L82-A 的 development rank evidence stop，
不是 HOTA/TrackEval 结果，也不是 L82-B 训练结果。

## 先排除的原因

1. **不是缺少精确监督。** L82 matrix audit 找到 662 个 eligible frame groups、
   V1=328/V2=334、6,074 candidate-query flip triplets 和 1,998 target-bag
   query flips，超过预注册最低规模。
2. **不是接口或数值失败。** GroundingDINO candidate-reference contract 通过；
   native query/reference delta 为 0，detector frozen，表达式改变了 candidate-
   specific representation，所有 rows finite 且保留。
3. **不是 DDP/重载失败。** retry3 四卡运行正常结束；三个表示各有 5,240 条
   finite/nonzero-gradient trace，三份 package 的 CPU strict reload 均无 missing/
   unexpected keys。retry1/2 的 launcher/gather/flag 问题均已保留并隔离，不能被
   当作科学结果。
4. **不是零方差或空输出。** L82 interaction variance 为 `.53519`，score std 为
   `1.90083`，empty count 为 0；因此存在交互变化，但它没有转化为可靠的目标级
   排序。

## 仍然失败的证据

L82 candidate-reference 在 138 个 video-disjoint dev groups 上：

- hard violation `.8653846`：通过单项 ≤`.8666667`，且比 L81 `.9391026` 改善
  `.073718`；paired bootstrap 95% CI 为 `[.033303,.156685]`；
- query-swap accuracy/AUC `.742515/.742515`：accuracy 通过 `.70`，AUC 未达
  `.75`；
- target-bag Recall@1/5 `.288462/.772436`：均不能支撑稳定目标级选择；
- multi-positive target coverage `.426752`：远低于 `.7894444`；
- V1/V2 query-swap accuracy `.715084/.748481`，但 V2 hard violation `.905759`；
- inactive false acceptance `1.0`。L71/L82 rank probe没有独立 NULL emission head，
  所以该项不能被解释成已解决。

L59 同容量 control 的 hard violation `.8205128` 和 target-bag Recall@1
`.320513` 均优于 L82；因此 L82 也不满足“不能劣于 L59 control”的预注册条件。

## 根因判断

唯一第一 actionable root cause 是：

`query_candidate_correspondence_ceiling_insufficient_for_target_bag_generalization`

更具体地说，candidate-reference state 确实提供了比 L81 更强的某些 query-wise
变化，但跨视频的变化仍不足以同时区分同帧 hard negatives、把正确 target bag
排到第一并保留所有 multi-positive。当前结果不能证明 GroundingDINO frozen
reference 是无信息的，但证明了 L82 预注册的浅层 factorized rank probe 不足以
授权 task-composition/LoRA。

## 选择性停止

根据 L82 master prompt，rank gate 失败后不运行 Phase E，不读取 16/24 historical
validation labels，不做 threshold/NULL/top-k/NMS rescue，不进行三 seed、screening、
official test、TrackEval/HOTA，也不修改 ordinary MOT/OVMOT。唯一下一行动是
`STOPPED_PENDING_SUPERVISOR_REVIEW`，由监督者另行批准一个新的、单因素的结构或
数据解阻塞假设；本轮不自行发明并行分支。

## 保留证据

- 首次 launcher、gather、barrier 和 retry2 flag 目录均保留 `INCOMPLETE.md`/
  `INVALID.md`；
- 权威结果：`outputs/l82/train/frozen_rank_probe_retry3/`；
- 完整性：`outputs/l82/audit/final_integrity.json`；
- strict reload：`outputs/l82/train/frozen_rank_probe_retry3/reload_audit.json`。

```text
screening_gt_used=false
official_test_labels_read=false
ordinary_mot_ovmot_touched=false
hota_trackeval_run=false
candidate_deletion=false
candidate_truncation=false
token_span_region_alignment=UNALIGNED
static_motion_alignment=UNALIGNED
```
