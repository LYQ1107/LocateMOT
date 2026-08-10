# Stage L2 — Local Correctness vs Future Utility Mismatch

日期：2026-08-10。

## 1. 定义

- 局部正确：action 中被匹配的 (track, candidate) 的 candidate GT 等于
  track 的出生 GT（true identity）；
- 未来最优：windowed AssA 最高的 action；
- 类别：
  - `local_correct_future_bad`：base 局部更正确，但未来效用反而更低；
  - `local_wrong_future_good`：base 局部更错，但未来效用更高；
  - `base_correct_future_suboptimal`：base 已局部全对，但未来仍可改进。

## 2. DanceTrack val（1,000 冲突事件）

| H | future_best ≠ base | local_correct_future_bad | local_wrong_future_good | base_correct_future_suboptimal |
|---|---:|---:|---:|---:|
| 4 | 75 | 30 | 39 | 4 |
| 8 | 135 | 69 | 49 | 9 |
| 16 | 185 | 98 | 58 | 11 |
| 32 | 219 | 128 | 60 | 19 |

## 3. BDD100K train（745 冲突事件）

| H | future_best ≠ base | local_correct_future_bad | local_wrong_future_good | base_correct_future_suboptimal |
|---|---:|---:|---:|---:|
| 2 | 134 | 40 | 46 | 10 |
| 4 | 268 | 90 | 74 | 23 |
| 8 | 397 | 137 | 110 | 38 |
| 16 | 460 | 173 | 110 | 39 |

## 4. 多 horizon 排序一致性

同一事件内 action 的效用排序随 horizon 增长逐渐稳定：

| 对比 | DanceTrack pair agreement | BDD pair agreement |
|---|---:|---:|
| 最短 vs 最长 | 27.6%（H4 vs H32） | 19.0%（H2 vs H16） |
| 中间 vs 最长 | 54.3%（H8 vs H32） | 42.0%（H4 vs H16） |
| 次长 vs 最长 | 73.7%（H16 vs H32） | 67.7%（H8 vs H16） |

MOT17/MOT20（小样本）：H16 vs H32 pair agreement 78.2% / 76.8%。

## 4b. MOT17 / MOT20 mismatch（H32）

| 域 | 事件 | future_best ≠ base | local_correct_future_bad | local_wrong_future_good |
|---|---:|---:|---:|---:|
| MOT17 | 120 | 85（70.8%） | 60 | 23 |
| MOT20 | 80 | 60（75.0%） | 47 | 7 |

MOT17/MOT20 的 mismatch 更偏向 “local correct but future bad”，
且单事件窗口 headroom 最大（+2.27/+1.76pp），但端到端实现后
MOT17 为 −2.32pp，进一步证明窗口效用与全局轨迹质量不同构。

## 5. 结论

1. **local 与 future 确实不同构**：DanceTrack H32 有 219/1000（21.9%）
   事件的未来最优 action 不同于 base；其中 128 个属于
   “base 局部正确但未来更差”，60 个属于“base 局部错但未来更好”；
   BDD H16 有 460/745（61.7%）、MOT17 H32 有 85/120（70.8%）、
   MOT20 H32 有 60/80（75.0%）事件未来最优不同于 base。
2. **但该 mismatch 的规模不足以支撑端到端收益**：这些差异的 windowed
   AssA 幅度平均只有 0.5–1.2pp，且端到端实现后互相抵消（见
   `reports/l2_oracle_headroom.md`）。
3. 科学含义：objective mismatch 真实存在（支撑论文核心问题），
   但在 L1DK base + AC 协议下 **不存在可学习的正收益空间**；
   这与 L1-D 的“局部修正成功但全局失败”完全一致。
