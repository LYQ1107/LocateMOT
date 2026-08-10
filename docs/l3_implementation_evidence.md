# Stage L3 — Implementation Evidence

日期：2026-08-10。

## 1. U0：共享 dense association core

- Module：`locatemot/models/l1d_association.py::L1DAssociator`
- Scientific purpose：多域共享 set-level 关联基线（local CE）。
- Reference：L1-D EGRA（本项目自研，基于 CAMELTrack/TDLP set-level
  竞争设计，commit 46a74bb / 50344b92 已审计）。
- Files inspected：`tools/train_l3.py`、`locatemot/models/l1d_association.py`
- Mechanism adopted：row/col CE + reliability BCE + base preservation。
- Mechanism changed：无（复用）。
- Why：U0 必须与 L1-D EGRA 可比。

## 2. U1：RegimeEncoder + FiLM 条件化

- Module：`locatemot/models/l3_unified.py`
  - `RegimeEncoder`：48 维 prediction-side 统计 → 32 维 z_regime；
  - `L3Associator`：FiLM（track/cand token、encoder 输出）+ z 注入
    pair head。
- Scientific purpose：验证“how to track”可由 latent regime 条件化。
- Reference（结构）：condition-aware routing / FiLM 常见于
  conditional vision（ICML 2026 Dual MoE 等，仅结构参考；
  无 MOT association 等价实现，见 `docs/l3_reference_audit.md`）。
- Files inspected：`locatemot/models/l3_unified.py`、
  `tools/analyze_l3_routing.py`
- Mechanism adopted：prediction-side stats（density/IoU/PBD ambiguity/
  motion/gap/age/hits/margin/competition），无 GT/future/dataset ID。
- Mechanism changed：无（首版即 FiLM；未做 MoE/hypernetwork，因
  pilot 已无正信号）。
- Why change：n/a。
- License：clean reimplementation（无外部代码复制）。

## 3. 评估管线

- `tools/eval_l3.py`：OnlineTracker AC shell + L1DK base 权重
  （0.4/0.2/0.4，thr 0.25，delta 0.3），输出同候选集只改 ID。
- `tools/run_l1d_trackeval.py`：官方 TrackEval。
- 协议：per-video fresh OnlineTracker（L1 定义），所有方法一致。

## 4. 关键实现决策记录

1. Regime 输入全部 causal：候选统计、历史统计、base 竞争；无未来。
2. 禁止 dataset ID：训练/推理均无 dataset 输入。
3. U1 与 U0 同数据、同步数、同 seed，保证对比公平。
4. 结果：U1 未过 Gate；z_regime 呈 dataset shortcut
   （domain classifier 96.6%），详见 `reports/l3_shortcut_audit.md`。
