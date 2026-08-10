# Stage L5 — Ablation

日期：2026-08-11。

## 已完成的配置对比（同一小集）

| 配置 | BDD 在线 drift | Dance 在线 drift | 备注 |
|---|---:|---:|---|
| U0 base | 53.2% | 37.9% | 基线 |
| A: gt-only, delta=0.6, pres=0.1, ep5 | 5.6%（瞬态） | 58.9% | 不迁移 |
| A: mixed gt+u0, ep10 | 65.3% | 68.1% | dom_GT 标签污染 |
| A: u0+cur_GT, delta=1.0, pres=0, spec-w=10, ep10 | 20.4% | 39.4% | Dance 仍差 |
| A: u0+cur_GT, delta=0.6, pres=0.1, spec-w=2, ep40 | 28.7% | 29.3% | **最终配置** |
| B: u0+cur_GT slot ID, ep20 | 69.3% | — | 不迁移 |

未执行：trajectory-level 在线一致性损失、随机 subset perturbation、
unseen spec type（NOT_EXECUTED）。
