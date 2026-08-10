# Stage L4 — Specification Paired-View Dataset

日期：2026-08-10。

## 1. 构造

`tools/build_l4_pairs.py` 对每个视频帧同时推进两个 base tracker
（L1DK：0.4 IoU + 0.2 PBD + 0.4 Kalman，thr 0.25）：

- full view：全部候选（spec=ALL）；
- restricted view：spec 保留的候选（category / instance）。

每对样本包含：

- full/rest 各自的 EGRA 特征（pair/track/cand/base）；
- 各自的 GT row/col label 与 base_correct；
- `common_cand`：(full_idx, rest_idx) 候选对齐（受限候选天然是
  full 候选的子集）；
- `common_track`：(full_track_idx, rest_track_idx) 轨迹对齐
  （按 birth 时 privileged GT 身份匹配）。

spec 类型编码（共享、非 dataset-specific）：

| idx | 类型 |
|---|---:|
| 0 | ALL |
| 1 | category |
| 2 | instance |

## 2. 规模

| Domain | Pairs | Spec 构成 |
|---|---:|---|
| BDD100K train | 7,645 | car 4,755 / pedestrian 1,274 / truck 953 / bus 340 / other vehicle 164 / bicycle 89 / motorcycle 21 / trailer 18 / rider 16 / other person 15 |
| DanceTrack calibration | 8,006 | inst:auto（top-2 GT） |
| MOT17 train | 180 | inst:auto |
| MOT20 train | 20 | inst:auto |
| **合计** | **15,851** | |

## 3. 用途

- 训练 A2（spec-conditioned、无一致性）与 A5（+ assignment/state
  consistency）；
- 可扩展用于 A3/A4 消融；
- 全部为 PRIVILEGED_SPEC_ORACLE（GT 身份只用于训练对齐，推理不使用
  GT membership）。
