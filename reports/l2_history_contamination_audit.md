# Stage L2 — 历史身份污染审计

日期：2026-08-10。

## 1. 问题

L1-D 发现 EGRA 的局部修正（连续性 0.841→0.951）没有转化为 TrackEval
AssA 收益。本审计检查：这些修正是否主要发生在“已经被历史 IDSW 污染”
的预测轨迹上；若是，则局部正确但未来无效的机制成立。

## 2. 方法

用与 baseline 完全一致的 AC shell 重放 L1DK base 与 EGRA（L1DK_d03），
逐帧对比 base 与 EGRA 的 track 分配。对每个发生修正的 track，在修正前
记录：

- track 出生 GT（true identity）与最近匹配 GT；
- 轨迹 purity（历史匹配中主导 GT 占比）；
- 历史 IDSW 次数、fragment 数、age、hits；
- base 边与 EGRA 边的 candidate GT。

分类（按出生身份）：

- helpful：EGRA 把分配改到 true identity 的候选；
- harmful：base 已分配到 true identity，EGRA 改走；
- same_gt / other：其余。

## 3. 聚合统计（专用审计 `tools/l2_history_contamination.py`）

### DanceTrack val

- 修正事件：1,295；
- helpful：334（25.8%）；harmful：264（20.4%）；
- same_gt：213（16.4%）；other：484（37.4%）。

### BDD100K train（30 视频）

- 修正事件：1,799；
- helpful：163（9.1%）；harmful：247（13.7%）；
- same_gt：748（41.6%）；other：641（35.6%）。

## 4. 污染状态分布

### DanceTrack val（按修正前轨迹状态）

| purity bucket | n | helpful | harmful |
|---|---:|---:|---:|
| <0.5 | 257 | 52 | 30 |
| 0.5–0.8 | 400 | 92 | 75 |
| 0.8–0.95 | 288 | 75 | 54 |
| ≥0.95 | 350 | 115 | 105 |

| past IDSW | n | helpful | harmful |
|---|---:|---:|---:|
| 0 | 317 | 116 | 87 |
| 1 | 104 | 31 | 9 |
| 2–3 | 131 | 40 | 35 |
| ≥4 | 743 | 147 | 133 |

| age | n | helpful | harmful |
|---|---:|---:|---:|
| 1–3 | 82 | 22 | 22 |
| 4–10 | 107 | 37 | 29 |
| 11–30 | 173 | 49 | 43 |
| >30 | 933 | 226 | 170 |

### BDD100K train（30 视频）

| purity bucket | n | helpful | harmful |
|---|---:|---:|---:|
| <0.5 | 1,076 | 74 | 175 |
| 0.5–0.8 | 101 | 11 | 6 |
| 0.8–0.95 | 18 | 1 | 1 |
| ≥0.95 | 604 | 77 | 65 |

| past IDSW | n | helpful | harmful |
|---|---:|---:|---:|
| 0 | 1,207 | 121 | 201 |
| 1 | 264 | 14 | 20 |
| 2–3 | 191 | 19 | 17 |
| ≥4 | 137 | 9 | 9 |

| age | n | helpful | harmful |
|---|---:|---:|---:|
| 1–3 | 683 | 65 | 107 |
| 4–10 | 532 | 31 | 86 |
| 11–30 | 512 | 58 | 49 |
| >30 | 72 | 9 | 5 |

## 5. 解读

1. EGRA 的修正大多数 **不指向出生身份**（DanceTrack helpful 25.8%、
   BDD 9.1%），说明学习到的修正方向与 GT 身份不一致；
2. BDD 上 59.8% 的修正发生在 purity<0.5 的重度污染轨迹上，且这些
   轨迹上的修正 harmful（175）远多于 helpful（74）；
3. DanceTrack 上 57% 的修正发生在 past IDSW≥4 的轨迹上，helpful 率
   反而低于 IDSW=0 轨迹（19.8% vs 36.6%）；
4. 这解释了 L1-D 的“局部正确但 AssA 下降”：在 AC 协议下，
   ID 是持久符号，污染轨迹上的任何局部修复都只能增加碎片化，
   无法恢复已损失的全局 ID 统计。

## 6. 结论

历史污染审计支持 Stage L2 的核心机制判断：**已污染状态上的局部
correct action 与未来轨迹效用不同构**。但同时也说明，在固定 ID
符号的 AC 协议下，这类修正没有可恢复空间（与 oracle headroom 低
一致）。
