# Stage L5 — Route B: Sequence-Local Dynamic ID Prediction

日期：2026-08-11。

## 1. 科学假设

持续关联更适合建模为 in-context identity prediction（MOTIP，CVPR 2025）：
每个 candidate 预测一个 sequence-local identity slot（或 NEW）；slot
词汇表在同一 clip 的 ALL/restricted 视图间共享，因此跨 spec 一致性被
直接监督（same GT → same slot），无需 dataset-global ID。

## 2. 实现

- `locatemot/models/l5_route_b.py`：复用 Route A temporal encoder +
  set encoder；slot head 输出 [N, max_slots+1] logits（NEW 为末位）；
  训练按每视频 slot map 屏蔽 > G 的 logits。
- 推理（`OnlineTracker` variant L5B）：track 在出生时领取预测 slot；
  每帧 candidate 预测 slot，与同 slot track 匹配，NEW 走新轨；
  Hungarian 在扩展矩阵上保证一对一。
- 训练：u0 source，Small 1.41M / Base 7.50M，60 epochs，batch 16，
  max_slots=128，GPU 0/3。

## 3. 结果

| 模型 | train slot acc (ep20) | val slot acc (ep20) |
|---|---:|---:|
| Small | 0.889 | 0.142 |
| Base | 0.932 | 0.133 |

在线 drift（Small ep20，BDD val）：

| 模型 | BDD 在线 drift |
|---|---:|
| U0 | 53.2% |
| Route B | 69.3% |

## 4. 解释

训练 slot acc 快速上升（模型能记住训练视频的 slot 语义），但 val slot
acc 停在 ~0.14（128 类中远高于随机但远低于可用），说明 sequence-local
slot 表示在 11 个训练视频上无法迁移到新视频；在线 rollout 因 slot 预测
噪声产生大量错误出生/匹配，drift 反而高于 U0。

判定：**L5_ROUTE_B_NOT_SUPPORTED（pilot 规模）**。与 MOTIP 需要
大规模数据（其论文在完整 MOT 数据上训练）一致；小集 overfit 满足
（train 0.93）但 generalization 不满足。
