# Stage L1-C PBD Ambiguity

定义：m_pbd = top1 PBD cosine − top2 PBD cosine（track 参考特征对当前
candidates；参考 = 该 GT 上一帧匹配 candidate 的 box-end feature）。

## DanceTrack val（C2 事件）

| PBD margin bucket | events | PBD 选择正确率（IoU 方法） |
|---|---:|---:|
| <0.01 | 20,471 | 0.831 |
| 0.01–0.03 | 133,212 | 0.958 |
| 0.03–0.08 | 65,602 | 0.970 |
| ≥0.08 | 5,786 | 0.908 |

## 解读

1. PBD candidate-selection 整体很强（margin≥0.01 时正确率 ≥0.91）；
2. 真正 PBD 歧义（margin<0.01）只占 9%，正确率仍有 0.83；
3. 因此 raw PBD 的差 AssA 不是 candidate-selection 问题，而是
   full-video ID 连续性 / set-level 分配问题。
