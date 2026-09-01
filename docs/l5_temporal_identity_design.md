# Stage L5 — Temporal Identity State 设计（Route A）

日期：2026-08-11。

## 1. 科学假设

**Route A**：U0 的 cross-spec identity drift 主要来自缺少 persistent
temporal identity state。L1-D 的 EGRA 只用「最后一帧 pair feature +
set transformer + bounded residual」，身份等价于单帧 appearance/motion
证据；当 candidate subset 改变时，竞争结构改变，单帧证据不足以稳定身份。

假设：把每个 track 的 observation history 因果压缩成一个 persistent
state h_i^t，并用 GT trajectory identity 监督（而不是 prediction
imitation），模型可以学到 specification-independent identity semantics，
同时保留 specification-dependent association evidence。

## 2. 架构

```
track observation sequence（≤16 obs: pbd_be + box + velocity + gen +
                              log_n_cand + gap）
        ↓
Temporal Identity Encoder（causal TransformerEncoder，per track）
        ↓
persistent state h_i^t

current candidates（pbd_be + 12-dim cand features）
        ↓ cand_proj
candidate tokens

[candidate tokens; track states] → Set-level Track-Candidate Interaction
        ↓
pair head: delta_ij = delta_scale * tanh(MLP(h_i, c_j, pair_feats_ij))
reliability gate: sigmoid(MLP(trk_out, row_sel))
        ↓
final = base + sigmoid(rel) * delta
```

设计依据（见 `docs/l5_reference_audit.md`）：

- MOTIP：trajectory feature + 相对时间交互 + candidate-as-query /
  trajectory-as-key-value；
- TrackFormer / MOTR / MeMOTR：persistent track query 生命周期；
- CAMELTrack / L1D EGRA：set-level competition + ranking CE + bounded
  residual + reliability gate（本项目的 L1D 就是 clean 实现）；
- SOTFormer（概念）：GT-primed persistent state。

## 3. 为什么不是 ReID

- L1-B 已证明 single-frame PBD → universal identity embedding →
  cosine matching 跨域不成立；
- Route A 的 state 是时间序列的因果压缩，包含 motion、appearance、
  candidate-set context、gap/uncertainty；
- 关联决策不是 state 之间的 cosine，而是 set-level decoder 在
  current-frame competition 下产生的 bounded residual；
- 关系损失（same/different GT）只是辅助，主损失仍是 GT-anchored
  row/column ranking。

## 4. 监督

1. **GT-anchored association**：每帧 row/col ranking CE，目标由
   candidate 的 GT identity 与 track 的 GT identity 决定
   （`gt` source 用 GT 轨迹；`u0` source 用 history 多数 GT）；
2. **Trajectory relation**：persistent state 对的
   same/different-GT BCE（辅助，权重 0.1）；
3. **Cross-spec relation-structure consistency**：同一 (video, frame)
   的 ALL 与 restricted 视图，在共同 GT identity 上的 state relation
   matrix 一致（MSE，权重 0.05）。两个视图分别对 GT 监督，因此不强制
   restricted 模仿 ALL 的错误。

## 5. 推理

与 L1D 完全一致：`final = base + gate * delta`，Hungarian 1-1 +
阈值 0.25 处理 NO_MATCH；不需要 ID 词汇表，不需要 retrospective
revision；online causal。

## 6. 容量阶梯

| 档位 | d_model | temporal layers | set layers | ffn | 预期参数量 |
|---|---|---:|---:|---:|---:|
| Small | 128 | 2 | 2 | 512 | ~1.5M |
| Base | 256 | 4 | 4 | 1024 | ~7.7M |
| Large | 384 | 6 | 6 | 1536 | ~25–35M |

Large 的 temporal encoder 结构更强（更多因果层），不是单纯把 MLP 乘 8。

## 7. 判定标准（overfit 阶段）

- small-set train drift 明显压到 <10–15%（相对 U0 65% 基线）；
- epoch 20 后继续改善；
- Base > Small；
- val drift 相对 U0 下降 ≥20–30%；
- P1-selected AssA / ALL AssA 不下降（后续 TrackEval 验证）。
