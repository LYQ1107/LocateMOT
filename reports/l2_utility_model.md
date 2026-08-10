# Stage L2 — Trajectory Utility Model (TUM) 设计

日期：2026-08-10。本文记录模型设计；是否训练由 Oracle Headroom Gate
决定（见 `reports/l2_oracle_headroom.md`）。

## 1. 科学定位

TUM 不是一个新的 tracker，而是“当前关联 action → 未来轨迹效用”的
回归/排序模型：

```text
Q(s_t, a) ≈ U_H(s_t, a)
```

- 训练监督：counterfactual rollout 的 windowed AssA（privileged future）；
- 输入：严格 causal（不含未来帧）；
- 推理：从候选 action 集合中选 argmax Q 的 action。

## 2. 输入表示（全部来自 L1DK base 的在线状态）

每个决策事件：

- Track tokens（每轨迹 16 维，复用 L1-D `TRACK_FEATURES`）：
  box、velocity、gap、age、hits、IoU/PBD/base top1 与 margin、
  anchor-cos 等；
- Candidate tokens（每候选 12 维，复用 `CAND_FEATURES`）：
  box、gen、size、top1/margin 等；
- Pair features（19 维，复用 `PAIR_FEATURES`）：
  IoU、Kalman-pred IoU、PBD cos、中心距离、scale、margin、
  set context；
- Action 指示矩阵：component 内 (track, candidate) 是否被该 action 匹配；
- Base 矩阵（强先验）。

历史身份污染只用 prediction-side proxy（track age、gap、history
consistency、top1 margin 等），禁止输入真实 GT purity。

## 3. 模型

- TUM-small（pilot）：d_model=256、4 层 TransformerEncoder、8 heads、
  FFN 1024；track/cand token 投影后一起做 set self-attention；
  pair head 拼接 `[track_emb, cand_emb, pair_feats, base, action]` →
  每 pair logit，按 action 边加权池化 → per-horizon head。
- 参数量：约 1.5–3M（pilot 足够验证 signal）。
- 若 pilot 通过：TUM-base 6–8 层、d_model 384，参数量 5–15M，
  DDP 2–4 卡训练。

## 4. 训练目标

每个 event × action × H 有一个 oracle utility（windowed AssA）。

- 主 loss：utility 回归 MSE（同一 event 内 action 间归一化可选）；
- 辅助：listwise ranking loss（softmax CE over actions per event），
  使排序与 oracle 一致；
- Multi-horizon：共享 trunk + per-H head；训练时所有 H 联合。

## 5. 数据与 split

- 训练域：DanceTrack calibration + BDD100K train（30 视频 oracle 事件）；
- 评估域：DanceTrack val（主 AC 验证）+ MOT17/MOT20 train（跨域）；
- split 按 video 隔离，保证同一视频不跨 train/eval。

## 6. 评估指标（Gate 2）

- top1 future-best action 准确率；
- pairwise ranking AUC（action 对）；
- NDCG@k；
- regret = U(best) − U(predicted choice)；
- 对比：base 恒选、local-correctness probe（选 GT-local 最优 action，
  仅 oracle 分析用）、随机。

## 7. 与 EGRA 的区别

| | EGRA | TUM |
|---|---|---|
| 监督 | 当前帧 GT 身份 | counterfactual 未来 windowed AssA |
| 决策 | 每 pair 残差 | 集合级 action 效用排序 |
| 推理 | 单帧 Hungarian | online，同帧 action 选择 + Hungarian |
| 目标 | local correctness | trajectory utility |

## 8. 不采用的设计

- 不直接回归未来 IDSW（方差大、难归一化）；
- 不把未来 GT 身份作为输入（泄漏）；
- 不首版使用 RL（见 `docs/future_rl_reference.md`）。
