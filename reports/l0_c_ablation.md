# Stage L0-C：必要消融

## A. PBD coordinate-mean vs box-end

| 特征 | conditional acc | NO_MATCH F1 |
|---|---:|---:|
| coordinate-mean（B2） | 0.614 | 0.701 |
| box-end（B2 消融） | 0.643 | 0.719 |

box-end 略优（+2.9pp），但均低于 B4。

## B. Region only vs PBD only vs learned fused

| 配置 | conditional acc |
|---|---:|
| Region cosine（B1） | 0.544 |
| PBD cosine（B2） | 0.614 |
| learned fused（B4） | 0.627 |

学习融合 > PBD > region。

## C. Pairwise MLP vs Persistent Track Decoder

| 模型 | conditional acc | NO_MATCH F1 |
|---|---:|---:|
| B3 Pairwise MLP | 0.618 | 0.627 |
| B4 TrackDecoder | 0.627 | 0.735 |

B4 在 NO_MATCH 上明显更强（+10.8pp），assignment 仅 +0.9pp。

## D. query direction（1500 pairs 子集，同数据同步数）

| 方向 | calibration |
|---|---:|
| reference_query | 0.7206 |
| current_query | 0.7181 |

差异 <1pp，保留 reference-query。

## E. single vs multi target（B4 held-out）

| target count | conditional acc |
|---|---:|
| 1 | 0.796 |
| 2–4 | 0.589 |
| 5–8 | 0.229 |

多目标竞争是主要关联困难场景。
