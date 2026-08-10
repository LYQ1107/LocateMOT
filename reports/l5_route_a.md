# Stage L5 — Route A: GT-Anchored Temporal Identity Transformer

日期：2026-08-11。

## 1. 科学假设

U0 的 cross-spec identity drift 主要来自缺少 persistent temporal identity
state；用 GT trajectory identity 锚定的 temporal state + set-level decoder
可以在不牺牲 tracking 质量的前提下提高跨 spec 身份一致性。

## 2. 实现

- `locatemot/models/l5_route_a.py`：causal temporal encoder（≤16 obs：
  PBD be + box + velocity + gen + log_n_cand + gap）→ persistent state；
  candidate token + state 进入 set-level encoder；pair head 输出 bounded
  residual（delta_scale=0.6）；reliability gate；`final = base + gate*delta`。
- 训练损失：GT-anchored row/col ranking CE（target=当前帧 GT 框 IoU
  锚定的 track identity，`track_cur_gt`）+ assignment-level cross-spec
  KL（common candidate 在 common GT tracks 上的 softmax 分布，权重 2）。
- 数据：u0 source（真实 tracker 含错轨迹），u0-only 最终配置；
  `track_cur_gt` 避免 history 多数 GT 的早期 switch 污染。
- 训练：Small 1.44M / Base 7.58M，120 epochs（报告至 ep40），batch 16，
  OneCycleLR（pct_start 0.05），seed 20260806，GPU 6/7。

## 3. 关键指标定义

主指标（L4 一致）：**video-level 全局最优 ID 对齐后的 track-ID 分歧率**
——对每个 (video, spec pair)，在所有帧上收集 common candidate 的
(tid_ALL, tid_spec)，用一次全局 Hungarian 对齐 track ID，分歧率 =
对齐后不一致的比例。该指标捕捉 persistent identity chain 的跨 spec
漂移，而不是单帧关联（单帧 cur_GT 对齐已 98%+ 一致）。

## 4. 结果（在线 rollout，val 小集）

| 模型 | BDD 在线 drift | Dance 在线 drift |
|---|---:|---:|
| U0 baseline | 53.2% | 37.9% |
| Route A Small ep10 | 28.7% | 45.0% |
| Route A Small ep20 | 28.7% | 29.3% |
| Route A Small ep40 | 28.7% | 29.3% |
| Route A Base ep20 | 28.7% | 29.3% |

相对 U0（ep40）：BDD -46%，Dance -23%。Small 与 Base 相同（小集饱和）。

## 5. 官方 TrackEval（AC 协议，Route A Small ep40）

| 域 | 模型 | HOTA | AssA | IDF1 | IDSW |
|---|---|---:|---:|---:|---:|
| Dance | U0 | 0.6283 | 0.4169 | 0.5694 | 2588 |
| Dance | Route A | 0.6293 | 0.4182 | 0.5647 | 2558 |
| BDD | U0 | 0.3628 | 0.2881 | 0.2923 | 11042 |
| BDD | Route A | 0.3672 | 0.2951 | 0.2954 | 12399 |
| MOT17 | U0 | 0.6595 | 0.6050 | 0.5825 | 259 |
| MOT17 | Route A | 0.6520 | 0.5914 | 0.5834 | 279 |
| MOT20 | U0 | 0.5012 | 0.2950 | 0.4012 | 2406 |
| MOT20 | Route A | 0.4849 | 0.2763 | 0.3800 | 2588 |

- Dance：AssA +0.13pp，HOTA +0.1pp，IDSW -30（改善）；
- BDD：AssA +0.7pp，HOTA +0.4pp，IDF1 +0.3pp，但 IDSW +1357（+12.3%）；
- MOT17/MOT20：AssA -1.4pp / -1.9pp，IDSW +20 / +182（下降）。

## 6. 学习曲线

见 `reports/l5_learning_curve.md`：val_row_acc 从 ep1 后恒定 0.9814，
ep10–46 无进一步变化；drift 在 ep20 后稳定。L4 的「20 epoch 太早」判断
在本架构上不成立（ep40 与 ep20 相同）。

## 7. 结论

Route A 机制有真实作用：两个域在线 identity drift 均显著下降，且 Dance
官方 AssA/IDSW 同步改善；但 BDD IDSW 明显上升、MOT17/MOT20 AssA 下降，
未满足「全部域标准指标不牺牲」的通过条件。判定：

**L5_ROUTE_A_PARTIAL（不满足 full-scale 通过条件，但为正机制信号）**
