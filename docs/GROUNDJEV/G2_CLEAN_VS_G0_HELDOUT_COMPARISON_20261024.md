# G2 Clean vs G0 independent held-out comparison — 2026-10-24

结论：G2 微调后在注册的独立 MOT20-05 和 RMOT multi scope 上有明确提升；RMOT single 只提升 HOTA，IDF1 略降；TAO OVMOT 仍为全拒绝，尚未提升。比较使用相同输入、解析器、dense decoder、Simple tracker 和官方 scorer。

| Scope | G0 HOTA | G2 HOTA | Δ HOTA | G0 IDF1 | G2 IDF1 | Δ IDF1 | G0 IDSW | G2 IDSW | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MOT20-05 / 1657 frames | 0.342615 | 0.444739 | 0.102123 | 0.290533 | 0.400865 | 0.110332 | 13712 | 8464 | 提升：HOTA、DetA、AssA、IDF1 均上升，IDSW 下降 |
| RMOT multi / 800 frames | 0.314869 | 0.507575 | 0.192706 | 0.374223 | 0.565583 | 0.191360 | 123 | 107 | 提升：HOTA、DetA、AssA、IDF1 上升，IDSW 下降 |
| RMOT single / 800 frames | 0.053121 | 0.061876 | 0.008755 | 0.014388 | 0.012878 | -0.001511 | 2 | 3 | 混合：HOTA/AssA 上升，DetA/IDF1 略降，IDSW 上升 |
| TAO video105 / 40 frames | — | — | — | — | — | — | — | — | 无提升：G0/G2 均 100% 感知拒绝，TETA=0 |

相对变化（HOTA）：MOT20-05 `+29.80%`，RMOT multi `+61.20%`，RMOT single `+16.48%`。MOT20-05 的 IDSW 从 13,712 降至 8,464；RMOT multi 从 123 降至 107。RMOT single 的 IDF1 从 0.014388 降至 0.012878，不能表述为全面提升。

模型证据：G2 为 Clean V2 checkpoint step `18227`，bundle model SHA-256 `7c4a81349e333a304e32f42cf1999db289802d8c31eba415b85827038d14801f`，adapter SHA-256 `cf1b5a6f626810ead94eb4e16adaf3feb6c6da3f99ca5023b6b5c234726f5742`；G0 bundle model SHA-256 `3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2`。G2 TRAIN manifest 与四个 held-out scope 的 exact image overlap 均为空。

边界：这些是 GroundAnything 感知前端 + 固定 Simple tracker 的 G2/G0 结果，尚不是 OneJev/TrackJEV 三任务联合微调后的指标；OVMOT 必须先修复原始 token/schema 兼容和全拒绝问题。完整机器可读证据见同名 JSON。
