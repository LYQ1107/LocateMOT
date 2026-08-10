# Stage L4 — TAO / Open-World Results

日期：2026-08-10。105 视频 / 4200 帧（2256 帧有候选）；
cache 已通过 `cache_key` 覆盖恢复（`docs/l4_tao_cache_recovery_plan.md`）。
PRIVILEGED_SPEC_ORACLE；windowed metrics 按视频均值，IDSW 为总和。

| Spec | U0 drift | A2 drift | A5 drift | U0 P1 AssA | A5 P1 AssA | U0 P1 IDSW | A5 P1 IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL | 0.0000 | 0.0000 | 0.0000 | 0.4174 | 0.4376 | 870 | 860 |
| baby | 0.0675 | 0.1142 | 0.1003 | 0.2625 | 0.2768 | 69 | 53 |
| car_(automobile) | 0.2406 | 0.2293 | 0.2481 | 0.1315 | 0.1318 | 70 | 62 |
| dog | 0.1152 | 0.2166 | 0.1198 | 0.0285 | 0.0299 | 25 | 26 |
| cat | 0.0119 | 0.0119 | 0.0119 | 0.0436 | 0.0436 | 2 | 2 |
| inst:auto | 0.1430 | 0.1865 | 0.1955 | 0.4616 | 0.4532 | 217 | 266 |

结论：

1. TAO 同样证明 restriction 会改变身份（U0 car 24%、inst 14%）；
2. A5 一致性训练未降低 drift（inst 14.3% → 19.6%），P1 指标混合；
3. A5 的 ALL 模式 AssA 略升（0.4174 → 0.4376），但不足以抵消
   BDD/DanceTrack 的退化与 drift 恶化。
