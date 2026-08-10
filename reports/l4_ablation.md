# Stage L4 — Ablations

日期：2026-08-10。

## 1. 已执行

| tag | 定义 | Cross-spec drift（Dance inst） | ALL AssA（Dance/BDD） |
|---|---:|---:|---:|
| A0 = U0 | frozen shared core | 0.3112 | 0.4514 / 0.3518 |
| A1 = P1 | U0 pre-filter（评估方式，非独立模型） | — | — |
| A2 | spec-conditioned，无一致性 | 0.3272 | 0.4535 / 0.3277 |
| A5 | + row/col KL + state cosine | 0.3168 | 0.4422 / 0.3351 |
| A5p | + partition co-assignment MSE（一次最小修正） | 0.3314 | 0.4457 / 0.3364 |

## 2. Where to Inject Specification

- Late selection（P0，Track-All-Then-Filter）：一致性最好（同一模型
  只跑一次），但受限对象指标差、IDSW 高；
- Early conditioning（P1，pre-filter）：受限对象指标最好
  （Dance inst AssA 0.8406 / IDSW 72），但身份与 ALL 不一致；
- Proposed（共享 core + paired consistency）：目标是把 P1 的
  selected-object 质量与 P0 的稳定性合并；当前机制未达成。

## 3. 未执行

- A3（仅 assignment consistency）与 A4（仅 state consistency）：
  因 A5 已失败，按任务书不再细分；如需论文对照可在后续重设计后补。
