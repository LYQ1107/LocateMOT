# Stage L5 — Failure Analysis

日期：2026-08-11。

## 1. 结论概览

- Route A：**PARTIAL**（两个域 online drift 显著下降 + Dance 官方指标
  改善；但 BDD IDSW 上升，未通过 full-scale 判据）。
- Route B：**NOT_SUPPORTED**（小集 overfit 成功但 val 不迁移）。
- Route C：NOT_EXECUTED（其核心思想被 Route A 的 cross-spec KL 部分覆盖）。

## 2. 按失败类别区分

### Route A

| 类别 | 证据 | 判断 |
|---|---|---|
| Implementation | 官方 TrackEval 重跑无 bug（本阶段先修了 L4 eval bug）；模型无 NaN；推理与训练 tensor 构造一致 | 无实现失败证据 |
| Optimization | 学习曲线在 epoch 10–46 平坦（val_row_acc 0.9814 恒定）；OneCycle 覆盖 120 epoch | 优化充分 |
| Capacity | Small==Base（drift 28.7%/29.3%）；gt-clean 训练可达 0.89 rowacc | 容量不是当前瓶颈 |
| Objective | per-frame GT CE + cross-spec KL 确实降低 drift（正机制）；但无法直接优化轨迹级全局对齐（indirect） | objective 部分有效，部分不足 |
| Generalization | val 视频不在训练集，drift 仍下降；u0 val rowacc ≈ base | 泛化存在 |
| Hypothesis | temporal state 在 BDD 强正、Dance ep20 转正；但 BDD IDSW 恶化说明修正会引入新 switch | **部分支持，非完全支持** |

### Route B

| 类别 | 证据 | 判断 |
|---|---|---|
| Optimization | train slot acc 0.93（ep20） | 优化充分 |
| Capacity | Base==Small（val 0.13-0.18） | 容量不是主因 |
| Objective/Generalization | val slot acc ~0.14，在线 drift 69.3% | sequence-local slot 表示在小集无法跨视频迁移 |

## 3. 科学边界（用户要求）

1. 单帧关联（U0 base）在 cur_GT 锚定下已经跨 spec 一致（98%+）；
2. 真正的 drift 在 persistent track-chain 层（45–53% 分歧），由早期
   association switch 累积；
3. 学习型 temporal state 能把 chain drift 降低 23–46%，代价是 BDD
   IDSW +12.3%（ep20）；
4. 0.5M/20-epoch 不是路线失败的充分证据（本阶段用了 1.4–7.6M/40+ 训练）；
5. 但「小集 per-frame residual 修正」的容量上限已被 Small==Base 提示；
   trajectory-level 目标（在线回滚、全局对齐）未在本阶段验证。

## 4. 下一步唯一建议

若继续：把 Route A 的 temporal state 与 **trajectory-level 在线一致性
损失** 结合（模型自身 rollout 的 track-chain 跨 spec 对齐，Gumbel-
Sinkhorn 近似），并在完整 BDD/Dance/MOT17/MOT20 数据上 2–4 GPU 训练；
这是唯一未被本阶段证据证伪、且已有正机制的路线。
