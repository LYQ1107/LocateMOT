# Stage L4 — Specification-Equivariant Training Pilot

日期：2026-08-10。

## 1. 设定

- 共享核心：U0（L1DAssociator，0.49M），U0 checkpoint 初始化；
- 结构：`L4SpecEqAssociator` = U0 core + type-level spec embedding
  （ALL/category/instance，只注入 set-encoder token，非 dataset-specific）；
- 配对数据：15,851 pairs（BDD 7,645 + DanceTrack calib 8,006 +
  MOT17 180 + MOT20 20）；
- 训练：20 epochs，batch 64，1 GPU，lr 3e-4，OneCycle；
  lambda_assign=1.0，lambda_state=0.1。

变体：

| tag | 一致性损失 |
|---|---|
| A2 | 无（naive spec-conditioned） |
| A5 | row/col 对称 KL（birth-GT 轨迹对齐）+ track-state cosine |
| A5p（一次最小修正） | partition-level co-assignment MSE + state cosine |

## 2. 结果：Cross-Spec Drift（P0 vs P1，最优 ID 映射后）

### BDD100K（200 视频，均值聚合）

| Spec | U0 drift | A2 drift | A5 drift | A5p drift |
|---|---:|---:|---:|---:|
| car | 0.3291 | 0.3502 | 0.3262 | 0.3398 |
| bus | 0.4259 | 0.4590 | 0.4495 | 0.4479 |
| truck | 0.3968 | 0.4178 | 0.4022 | 0.4242 |
| pedestrian | 0.4878 | 0.5048 | 0.4793 | 0.4996 |
| bicycle | 0.4845 | 0.5103 | 0.4948 | 0.5309 |
| other vehicle | 0.4744 | 0.4744 | 0.4391 | 0.4519 |

结论：A2 全面变差；A5 只在少量类别小幅改善；A5p 全面接近或差于 U0。
ALL 模式 audit 均值：U0 0.3518 → A2 0.3277 / A5 0.3351 / A5p 0.3364
（官方 pooled TrackEval 三者与 U0 完全一致，见 `reports/l4_ac_results.md`）。

### DanceTrack val（25 视频）

| Spec | U0 drift | A2 drift | A5 drift | A5p drift |
|---|---:|---:|---:|---:|
| person | 0.3225 | 0.3367 | 0.3394 | 0.3346 |
| inst:auto | 0.3112 | 0.3272 | 0.3168 | 0.3314 |

A2/A5/A5p 均未降低 drift；ALL audit 均值：U0 0.4514 → A2 0.4535 /
A5 0.4422 / A5p 0.4457。

### TAO（105 视频，open-world）

| Spec | U0 drift | A2 drift | A5 drift | A5p drift |
|---|---:|---:|---:|---:|
| baby | 0.0675 | 0.1142 | 0.1003 | 0.1073 |
| car_(automobile) | 0.2406 | 0.2293 | 0.2481 | 0.2456 |
| dog | 0.1152 | 0.2166 | 0.1198 | 0.1751 |
| inst:auto | 0.1430 | 0.1865 | 0.1955 | 0.1677 |

## 3. Pilot Gate 判定

| Gate | 结果 |
|---|---|
| A. Cross-spec consistency 显著下降（≥25–30% relative） | FAIL：A5 多数类别无改善甚至变差 |
| B. Selected-object tracking ≥ naive pre-filter | 基本持平/略降（Dance inst P1 AssA 0.8300 vs U0 P1 0.8406） |
| C. ALL preservation（≤0.3–0.5pp 下降） | FAIL：BDD −1.67pp，Dance −0.92pp |
| D. Cross-domain | FAIL：TAO drift 反而上升 |

## 4. Stage Decision

```text
L4_PILOT_GATE_FAIL
（逐帧 assignment/state consistency 不足以消除 temporal identity drift）
```

一次最小修正（A5p：partition-level co-assignment consistency）正在
训练验证后同样失败（Dance inst drift 0.3314 > U0 0.3112；
BDD car 0.3398 > U0 0.3291）。按任务书停止堆模型，进入
failure analysis。
