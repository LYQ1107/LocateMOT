# Stage L4 — Cross-Spec Consistency Metric Audit

日期：2026-08-10。

## 1. 要求

比较两个 spec 视图（例如 ALL vs category）的身份一致性时：

- **permutation invariant**：不比较 raw track integer ID；全局一致重标号
  必须视为一致；
- 只比较 **common detections / common identities**（两视图都出现的
  候选与轨迹）；
- 对 **merge / split / switch** 敏感；
- 有 toy-case 验证；
- 与 GT-based AssA/IDF1 做 sanity check（TrackEval 仍是主指标）。

## 2. 已检索的候选指标

| 指标 | 优点 | 缺点 / 为何不作为主指标 |
|---|---|---|
| 直接比较 track ID | 简单 | 违反 permutation-invariant，全局重标号即误报 |
| HOTA-style AssA（per view） | 官方主指标 | 度量每个视图自身的关联质量，不直接度量两视图一致 |
| Pairwise co-identity agreement（最优 ID 映射后） | permutation-invariant，对 merge/split/switch 敏感 | 需要一个对齐步骤（匈牙利），对齐本身是诊断的一部分 |
| Partition F1 / Adjusted Rand Index | 成熟的聚类一致性 | 对候选集合差异敏感，需要额外规定 common set；可作诊断 |
| Trajectory partition consistency | 直观 | 与 pairwise co-identity 等价但实现更绕 |

结论：**采用「最优 ID 映射后的 pairwise co-identity agreement」作为
主诊断指标**，并同时报告：

- per-GT identity drift（每个 GT 身份在两视图间不一致的比例）；
- 每个视图自己的 TrackEval-consistent windowed AssA/IDF1/IDSW。

## 3. 实现

`tools/l4_restriction_audit.py`：

1. P0（Track-All-Then-Filter）：对全候选流跑 frozen U0，再按 spec 过滤
   轨迹；
2. P1（Pre-Filter）：只对 spec 候选流跑同一个 U0；
3. 在公共候选帧上收集 `(frame, tid_P0, tid_P1, gid, cat)`；
4. 用 Hungarian 在 `(tid_P0, tid_P1)` 共现计数矩阵上求最优 ID 映射；
5. `agree_rate` = 映射后一致的比例；`drift_rate = 1 - agree_rate`；
6. `per_gt_drift`：按 GT id 聚合的一致率；
7. 每视图用 `windowed_metrics`（AssA/IDF1/IDSW，公式与官方 TrackEval
   一致）计算。

## 4. Toy-case 验证（2026-08-10 实际运行）

| Case | agree_rate | 判断 |
|---|---:|---|
| 全局一致重标号（A→B 每帧同一映射） | 1.0000 | 正确：partition 未变 |
| 两个身份在 B 中合并 | 0.5000 | 正确：merge 被捕获 |
| 一个身份在 B 中分裂 | 0.6667 | 正确：split 被捕获 |
| 5 个身份全局一致置换 | 1.0000 | 正确：permutation-invariant |
| 全视频一致 switch（1↔2 互换） | 1.0000 | 正确：partition 等价 |

真实数据中的 drift（BDD 33–67%、DanceTrack 31–32%）不是全局置换，
而是随时间不一致的 merge/split/switch，因此被该指标捕获。

## 5. 与 TrackEval 的 sanity

同一代码路径的 `windowed_metrics` 已在 Stage L2 与官方 TrackEval
整视频数值对齐（`docs/l2_trackeval_objective_audit.md`）。本审计中
`ALL vs ALL` 的自检为 `agree_rate = 1.0` 且 P0/P1 指标完全相同，
排除实现层面的系统性偏差。

## 6. 不做的事

- 不比较 raw track integer ID；
- 不把「自定义 consistency」当主结果替代 TrackEval；
- 不使用 partition F1/ARI 作为主指标（两视图候选集合大小不同时
  需要额外规定 common set；可留作论文诊断，本阶段未实现）。
