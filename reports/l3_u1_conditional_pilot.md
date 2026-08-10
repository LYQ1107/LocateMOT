# Stage L3 — U1：Regime-Conditioned 条件化核心（Pilot）

日期：2026-08-10。

## 1. 定义

U1 = L3Associator：U0 同架构 + RegimeEncoder
（prediction-side 统计 → z_regime 32 维）+ FiLM 条件化
（track/cand token 与 encoder 输出）+ z 注入 pair head。
训练数据/步数与 U0 完全一致（30 epochs，seed 20260806）。

Regime 输入（causal）：候选数、IoU/PBD 歧义、运动代理、gap、
track age/hits、margin、base 竞争统计。无 dataset ID。

## 2. 结果（四域 AC，同协议）

| Domain | U0 AssA | U1 AssA | Δ | U0 IDF1 | U1 IDF1 | U0 IDSW | U1 IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|
| DanceTrack | 0.4169 | 0.4050 | −1.19pp | 0.5694 | 0.5618 | 2,588 | **2,528** |
| MOT17 | 0.6050 | 0.5859 | −1.91pp | 0.5825 | 0.5726 | **259** | 274 |
| MOT20 | 0.2950 | 0.2958 | +0.08pp | 0.4012 | 0.3971 | **2,406** | 2,436 |
| BDD | 0.2881 | 0.2792 | −0.89pp | 0.2923 | 0.2861 | **11,042** | 11,027 |

Macro AssA：U1 0.3915 vs U0 0.4013（−0.98pp）。

## 3. Gate 判定

任务书 Gate A：U1 在 ≥3 个 heterogeneous domains 总体优于 U0。

**不满足**：U1 只在 MOT20 微正（+0.08pp），DanceTrack/MOT17/BDD
均下降。

## 4. 原因分析

1. z_regime 学成了 dataset shortcut：域分类器在 z 上的准确率 96.6%
   （随机 25%），域质心距离远大于域内标准差（见
   `reports/l3_shortcut_audit.md`）；
2. Regime 特征（density/gap 等）与 dataset 天然相关（BDD 5fps 大 gap、
   DanceTrack 30fps），FiLM 条件化退化为 dataset-conditional 偏置；
3. 即使作为 dataset 偏置，也没有带来跨域收益——说明该 pilot 的
   regime 条件化对 association 学习无正信息。

## 5. 结论

```text
L3_REGIME_NOT_SUPPORTED（pilot）
REGIME_ROUTER_DATASET_SHORTCUT（z 与 dataset 强相关）
```

不进入正式训练；不堆容量/不换 MoE 强行挽救。
