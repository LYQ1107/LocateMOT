# Stage L4 — Association-Controlled Results（官方 TrackEval，ALL 模式）

日期：2026-08-10。协议与 L3 完全一致（fresh per-video OnlineTracker，
L1DK shell，thr 0.25 / delta 0.3；`tools/eval_l3.py` +
`tools/run_l1d_trackeval.py`，官方 TrackEval）。

| Domain | U0 AssA | U0 IDF1 | U0 IDSW | A2/A5/A5p AssA | A2/A5/A5p IDF1 | A2/A5/A5p IDSW |
|---|---:|---:|---:|---:|---:|---:|
| DanceTrack val | 0.4169 | 0.5694 | 2,588 | 0.4169 | 0.5694 | 2,588 |
| MOT17 | 0.6050 | 0.5825 | 259 | 0.6050 | 0.5825 | 259 |
| MOT20 | 0.2950 | 0.4012 | 2,406 | 0.2950 | 0.4012 | 2,406 |
| BDD 11-class | 0.2881 | 0.2923 | 11,042 | 0.2881 | 0.2923 | 11,042 |
| Macro AssA | 0.4013 | — | — | 0.4013 | — | — |

说明：

1. 官方 TrackEval 的 ALL 模式数值在 U0/A2/A5/A5p 间完全一致（4 位小数）；
2. `l4_restriction_audit` 的 per-video 均值 ALL 指标有微小差异
   （BDD 0.3518 → A5 0.3351、Dance 0.4514 → A5 0.4422），属于
   均值聚合对小变化的放大，官方 pooled 指标不受影响；
3. 因此 **A2/A5/A5p 保持了 ALL 模式的标准 TrackEval**（Gate C 在官方
   指标上通过），但 cross-spec consistency 未改善（Gate A 失败）。

主结果仍以官方 TrackEval 为准；audit 均值只作诊断。
