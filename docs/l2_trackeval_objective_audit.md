# Stage L2 — TrackEval AssA / IDF1 Objective Audit

日期：2026-08-10。审计对象：`references/TrackEval-official`
（commit `HEAD` 以仓库内 `.git` 为准；本文依据实际源码推导，不凭印象）。

## 1. 为什么必须先审计

Stage L2 的 Future Trajectory Utility 必须与最终评测目标对齐。
评测协议是 Association-Controlled（固定 boxes/scores/帧数，只改 IDs），
因此真正受关联影响的是：

- HOTA AssA（关联质量）；
- IDF1（全局身份匹配质量）；
- IDSW（身份切换次数）；
- Frag（轨迹断裂次数）。

DetA 在 AC 协议下固定，不作为关联收益。

## 2. 数据预处理：ID 重标号

`trackeval/datasets/mot_challenge_2d_box.py::get_preprocessed_seq_data`：

1. 只保留 pedestrian 类，移除 distractor 匹配的 tracker det、ignore 区域 det、
   zero_marked GT det（MOTChallenge 规则）；
2. 对每个序列收集 `unique_gt_ids` / `unique_tracker_ids` 并重标号为
   **contiguous 整数**（`np.unique` 顺序）：

```python
unique_gt_ids = np.unique(unique_gt_ids)
unique_tracker_ids = np.unique(unique_tracker_ids)
...
new_id = dict(zip(unique_gt_ids, range(len(unique_gt_ids))))
```

含义：TrackEval 内部不关心外部 track ID 的具体数值，只关心
**每个预测 ID 与每个 GT ID 共现的帧集合**。任何关联策略的效用最终都
反映为“预测轨迹 ID 与 GT 轨迹 ID 的共现统计”。

## 3. HOTA AssA 的精确计算（`trackeval/metrics/hota.py`）

### 3.1 全局共现计数

对每一帧 t，用 IoU 相似度矩阵 `similarity` 计算归一化相似度
`sim_iou`（Jaccard 形式），累加到 `potential_matches_count[gt_id, trk_id]`；
同时累加每个 GT ID 的出现次数 `gt_id_count` 与每个 tracker ID 的出现次数
`tracker_id_count`。

### 3.2 全局对齐分数

```python
global_alignment_score = potential_matches_count / (
    gt_id_count + tracker_id_count - potential_matches_count)
```

即：每个 (gt_id, trk_id) 对的 **co-occurrence Jaccard 分数**。

### 3.3 逐帧匹配（Hungarian）

每帧的分数矩阵 = `global_alignment_score * similarity`，
`linear_sum_assignment(-score_mat)` 得到唯一匹配；仅当
`similarity >= alpha` 时计入该 alpha 的 TP。

### 3.4 AssA(alpha)

```python
matches_count = matches_counts[a]                     # (num_gt_ids, num_tracker_ids)
ass_a = matches_count / max(1, gt_id_count + tracker_id_count - matches_count)
AssA[a] = sum(matches_count * ass_a) / max(1, HOTA_TP[a])
```

等价解释：对每对 (gt_id, trk_id)，先算二者在整个序列中
**共同匹配的检测次数**与 Jaccard 分母的比值 `ass_a`，再按共同匹配次数
加权平均，分母是总匹配次数 `HOTA_TP[a]`。

注意：`HOTA_AssA(0)` 实际是 `array_labels[0] = 0.05`（不是 alpha=0），
即严格匹配阈值最低档。HOTA(0)/AssA(0)/DetA(0) 均取 alpha=0.05。

### 3.5 跨序列合并

`AssA` 跨序列用 `HOTA_TP` 加权平均；IDF1 的整数计数跨序列直接求和。
因此最终报告值是“按序列长度加权的整体统计”，不是序列平均。

## 4. IDF1 的精确计算（`trackeval/metrics/identity.py`）

1. 每帧统计 `similarity >= 0.5` 的 (gt_id, trk_id) 共现次数
   `potential_matches_count`；
2. 构造带 dummy 的全局分配矩阵，最小化
   `IDFP = tracker_id_count - matches`、`IDFN = gt_id_count - matches`
   （Hungarian，允许 gt 只分给一个 tracker id）；
3. `IDTP = total_gt_dets - IDFN`；
4. `IDF1 = IDTP / (IDTP + 0.5*IDFP + 0.5*IDFN)`。

含义：IDF1 是 **整个序列上的最优 ID 映射** 下的 F1，仍然只依赖
ID 共现计数。关联动作改变的是 tracker ID 的时间分配，因此 IDF1 能
反映轨迹级身份质量。

## 5. IDSW 与 Frag 的精确计算（`trackeval/metrics/clear.py`）

- 每帧先匹配（先最小化 IDSW，再最大化 MOTP 的 Hungarian）；
- `IDSW`：当某个 GT ID 在**任意历史帧**匹配过的 tracker ID 与当前帧
  不同时，计一次（`prev_tracker_id` 记录上次匹配的 tracker id）；
- `Frag`：GT 从匹配到不匹配再到匹配，每次“重新进入”计一次。

因此 IDSW 本质是 **轨迹身份在时间上的切换次数**，对局部单帧纠正非常
敏感：即使单帧分配在“局部 IoU 最优”意义下正确，只要它与该轨迹历史
身份不同，就会引入 IDSW。

## 6. 对 Stage L2 Future Utility 的设计约束

由上述源码可导出以下设计规则：

1. **Utility 必须以轨迹 ID 共现统计为对象**：
   不能只用单帧 pair accuracy；必须把“该 action 导致的 track ID 序列”
   与 GT ID 序列比较。
2. **Windowed utility 的排序应与 TrackEval AssA 一致**：
   在窗口内复现 `matches_count` → Jaccard → 加权平均的公式，即可得到
   `windowed_assA`。这与官方 AssA 在窗口边界上存在截断差异（官方是
   全序列匹配 + 逐帧 Hungarian），需要作为已知近似。
3. **IDF1 与 AssA 可能给出不同排序**：AssA 按检测加权、IDF1 按 ID 分配
   优化；两者都记录，但 AC 主指标以 AssA 为准（HOTA 的组成项）。
4. **本地正确 ≠ 未来效用**：单帧 GT-edge 正确不会自动转化为
   AssA/IDF1/IDSW 收益，因为后者是轨迹级统计。这正是 Stage L2 要
   用 Counterfactual Rollout 证明的核心机制问题。

## 7. 采用的设计（Stage L2 Oracle 阶段）

Primary utility：

```text
U_H(s_t, a) = windowed_assA(window = [t, t+H))
```

其中 windowed_assA 按官方公式实现：

```text
matches_count[g, p] = Σ_{frames in window, alpha-matched pairs}
    (gt_id=g, pred_id=p 的匹配计数)
ass_a[g,p] = matches_count[g,p] / (gt_count[g] + pred_count[p] - matches_count[g,p])
U = Σ matches_count * ass_a / Σ matches_count
```

同时记录 windowed IDF1、window IDSW、轨迹 purity（主导 GT ID 占比）
作为辅助证据，避免人工 reward 设计偏差。

约束：窗口内匹配采用与 TrackEval 相同的逐帧 Hungarian +
`global_alignment_score` 动态更新；窗口截断导致的边界效应在报告中
单独量化（headroom 验证时同时给整序列 TrackEval 数字）。

## 8. 已核对文件

- `references/TrackEval-official/trackeval/metrics/hota.py`
  （`eval_sequence`、`combine_sequences`、`_compute_final_fields`）
- `references/TrackEval-official/trackeval/metrics/identity.py`
  （`eval_sequence`、`_compute_final_fields`）
- `references/TrackEval-official/trackeval/metrics/clear.py`
  （`eval_sequence` 的 IDSW/Frag 计数）
- `references/TrackEval-official/trackeval/datasets/mot_challenge_2d_box.py`
  （`get_preprocessed_seq_data` 的 distractor 过滤与 ID 重标号）

## 9. 结论

AssA / IDF1 / IDSW 全部是 **轨迹级身份共现统计**，与单帧 pair
correctness 不同构。Stage L2 的 future utility 采用 windowed
association-aware 公式（以官方 AssA 为排序基准），并通过 oracle
headroom 实验验证窗口效用与整序列 TrackEval 的一致性。
