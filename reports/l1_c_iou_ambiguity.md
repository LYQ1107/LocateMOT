# Stage L1-C IoU Ambiguity

定义：对每个 GT-valid association event，m_iou = top1 IoU − top2 IoU
（prediction-side candidates vs GT box）。bucket 为固定阈值
（calibration 与 val 分布接近，直接采用规格推荐 bucket）。

## DanceTrack val（225,071 events/方法）

| IoU margin bucket | events | 占比 | IoU 方法 acc | Motion acc | RawPBD acc | UAF acc |
|---|---:|---:|---:|---:|---:|---:|
| <0.02 | 1,509 | 0.7% | 0.472 | 0.482 | 0.427 | 0.400 |
| 0.02–0.05 | 2,278 | 1.0% | 0.563 | 0.565 | 0.499 | 0.462 |
| 0.05–0.10 | 4,062 | 1.8% | 0.652 | — | — | 0.533 |
| ≥0.10 | 217,222 | 96.5% | 0.960 | 0.961 | 0.906 | 0.856 |

## 解读

1. DanceTrack 绝大多数 association event 的 IoU margin 极高（96.5%
   ≥0.10），IoU 本身就是强 cue；
2. 在 IoU 歧义区（margin<0.05，仅 1.7% events），所有方法 acc 都只有
   0.40–0.58，是共同难点；
3. UAF 在 easy 区（≥0.10）明显低于 IoU（0.856 vs 0.960），说明
   from-scratch 学习破坏了强先验。
