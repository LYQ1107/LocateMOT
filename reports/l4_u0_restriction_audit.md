# Stage L4-A — U0 Restriction Audit：P0 (Track-All-Then-Filter) vs P1 (Pre-Filter)

日期：2026-08-10。
协议：frozen U0（L1DAssociator，`outputs/l3/checkpoints/u0/final.pt`），
OnlineTracker L1DK shell（weights 0.4/0.2/0.4，thr 0.25，delta 0.3），
per-video fresh tracker，`output_all_candidates=True`。

## 1. BDD100K（200 视频 / 8001 帧 / 11 类 GT，PRIVILEGED_SPEC_ORACLE）

windowed AssA/IDF1/IDSW 按视频均值聚合（IDSW 为总和）；ALL 自检
agree=1.0 且 P0=P1，验证管线。

| Spec | Pairs | agree | drift | P0 AssA | P1 AssA | P0 IDSW | P1 IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL | 56,013 | 1.0000 | 0.0000 | 0.3518 | 0.3518 | 16,603 | 16,603 |
| car | 32,059 | 0.6709 | 0.3291 | 0.3950 | 0.3801 | 9,511 | 8,527 |
| bus | 634 | 0.5741 | 0.4259 | 0.2037 | 0.2656 | 231 | 42 |
| truck | 2,044 | 0.6032 | 0.3968 | 0.3438 | 0.4404 | 688 | 236 |
| pedestrian | 6,097 | 0.5122 | 0.4878 | 0.3151 | 0.3462 | 2,486 | 2,053 |
| rider | 59 | 0.5424 | 0.4576 | 0.0873 | 0.0927 | 20 | 3 |
| motorcycle | 39 | 0.5128 | 0.4872 | 0.0356 | 0.0400 | 15 | 3 |
| bicycle | 194 | 0.5155 | 0.4845 | 0.1070 | 0.1200 | 76 | 31 |
| train | 1 | 1.0000 | 0.0000 | 0.0050 | 0.0050 | 0 | 0 |
| trailer | 30 | 0.3333 | 0.6667 | 0.0022 | 0.0052 | 12 | 3 |
| other vehicle | 312 | 0.5256 | 0.4744 | 0.1057 | 0.1318 | 136 | 31 |
| other person | 62 | 0.5968 | 0.4032 | 0.0426 | 0.0446 | 25 | 20 |

除 car 外，P1 的 AssA 均 ≥ P0；所有样本充足的类别 P1 的 IDSW 都显著
更低（bus −82%、truck −66%、rider −85%、motorcycle −80%、bicycle −59%、
other vehicle −77%、trailer −75%）。

## 2. DanceTrack val（25 视频）

| Spec | Pairs | agree | drift | P0 AssA | P1 AssA | P0 IDSW | P1 IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL | 218,580 | 1.0000 | 0.0000 | 0.4514 | 0.4514 | 5,858 | 5,858 |
| person | 213,113 | 0.6775 | 0.3225 | 0.4505 | 0.4597 | 4,464 | 3,947 |
| inst:auto (top-2 GT) | 48,586 | 0.6888 | 0.3112 | 0.5592 | 0.8406 | 799 | 72 |

## 3. 结论

1. **候选集限制真实改变 persistent identity**：BDD category 的
   drift 33–67%，DanceTrack instance 31%；
2. 限制方向通常**改善**受限视图的关联质量（IDSW 大幅下降，
   多数 AssA 上升）——说明 distractor competition / set context
   是身份漂移的来源；
3. `L4_SPEC_RESTRICTION_SIGNAL_SUPPORTED`（BDD category 主证据 +
   DanceTrack instance 第二证据，跨两个 domain/spec 类型）。
