# Stage L1-D Correction Behavior

定义：每个 GT-valid 事件比较 base 与 L1D 分配给该 GT 候选的 track id；
“正确”= 与该方法上一帧给同一 GT 的 id 一致（帧间连续性，与
learnability probe 的 method_wrong 定义一致）。

## DanceTrack val（225,071 事件）

| 指标 | 数值 |
|---|---:|
| helpful（base 断、L1D 续） | 27,993 |
| harmful（base 续、L1D 断） | 3,187 |
| preserved（都续） | 186,081 |
| correction precision | 0.898 |
| correction coverage | 0.782 |
| preservation rate | 0.983 |
| base continuity acc → L1D acc | 0.841 → 0.951 |

## DanceTrack calibration（68,834 事件）

| 指标 | 数值 |
|---|---:|
| helpful / harmful / preserved | 5,281 / 679 / 61,433 |
| correction precision / coverage | 0.886 / 0.786 |
| preservation rate | 0.989 |
| base → L1D continuity acc | 0.902 → 0.969 |

## 解读

1. 残差模型确实学会了“保持 ID 连续”：修复了约 78% 的连续性断裂，
   只破坏 1.7% 的正确事件。
2. 但 TrackEval 官方 AssA/IDSW 不奖励这种连续性改善（val AssA
   −1.7pp、IDSW +21），说明该连续性定义与 CLEAR/HOTA 的匹配语义
   不完全一致（TrackEval 将大量断裂计入 Frag 而非 IDSW，
   val Frag 5,209→5,217 几乎不变）。
3. 结论：correction mechanism 真实存在且 precision 高，但不是
   官方指标的正向机制；不能作为统一部署模块。

