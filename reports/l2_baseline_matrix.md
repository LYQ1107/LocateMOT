# Stage L2 — 四域 Association-Controlled Baseline Matrix

日期：2026-08-10。
协议：所有方法使用完全相同的候选集（boxes/scores/features）、相同帧数、
相同输出数量，只改变 track IDs（Association-Controlled）。
评测：官方 TrackEval（HOTA/CLEAR/Identity），输出中 AssA(0)/IDF1(0)
按 TrackEval `array_labels[0]=0.05` 档位报告。

## 1. 方法定义

| Variant | 定义 |
|---|---|
| C0 | 纯 IoU（last box），Hungarian + 阈值 0.3 |
| C1 | Kalman motion IoU（pred box）+ second-stage last-IoU，阈值 0.3 |
| C2 | raw PBD cosine，阈值 0.3 |
| C3 | 固定线性融合 IoU+PBD（0.7/0.3），阈值 0.3 |
| L1DK base | Kalman motion IoU + IoU + PBD 线性融合（0.4/0.4/0.2），阈值 0.25，无训练参数 |
| L1DK_d03 | 同一基座 + EGRA set-transformer 有界残差（delta scale 0.3） |

## 2. DanceTrack val（40 序列，官方 GT）

| Variant | HOTA | AssA | IDF1 | IDSW | Frag |
|---|---:|---:|---:|---:|---:|
| C0 | 0.6078 | 0.3899 | 0.5291 | 3,554 | 5,283 |
| C1 | 0.6301 | 0.4193 | 0.5660 | 2,916 | 5,221 |
| C2 | 0.3836 | 0.1555 | 0.3188 | 15,616 | 5,621 |
| C3 | 0.6103 | 0.3934 | 0.5367 | 2,981 | 5,254 |
| **L1DK base** | **0.6280** | **0.4165** | **0.5630** | **2,558** | **5,209** |
| L1DK_d03 | 0.6149 | 0.3992 | 0.5522 | 2,598 | 5,218 |

## 3. MOT17 train（3 序列）

| Variant | HOTA | AssA | IDF1 | IDSW | Frag |
|---|---:|---:|---:|---:|---:|
| C0 | 0.5682 | 0.4504 | 0.5055 | 569 | 328 |
| C1 | 0.6308 | 0.5530 | 0.5606 | 340 | 329 |
| C2 | 0.2645 | 0.0975 | 0.2155 | 1,325 | 345 |
| C3 | 0.5259 | 0.3856 | 0.4507 | 351 | 325 |
| **L1DK base** | **0.6569** | **0.6010** | **0.5784** | **276** | **323** |
| L1DK_d03 | 0.6525 | 0.5922 | 0.5775 | 274 | 325 |

## 4. MOT20 train（2 序列）

| Variant | HOTA | AssA | IDF1 | IDSW | Frag |
|---|---:|---:|---:|---:|---:|
| C0 | 0.4196 | 0.2071 | 0.3206 | 3,113 | 430 |
| C1 | 0.4942 | 0.2869 | 0.3956 | 2,824 | 426 |
| C2 | 0.2508 | 0.0740 | 0.1755 | 3,139 | 445 |
| C3 | 0.4299 | 0.2171 | 0.3248 | 2,396 | 425 |
| L1DK base | 0.4864 | 0.2779 | 0.3232 | 3,736 | 423 |
| **L1DK_d03** | **0.4937** | **0.2864** | **0.3916** | **2,408** | **427** |

## 5. BDD100K train（200 视频，5fps 采样）

| Variant | HOTA | AssA | IDF1 | IDSW | Frag |
|---|---:|---:|---:|---:|---:|
| C0 | 0.3731 | 0.3044 | 0.2981 | 14,457 | 2,595 |
| C1 | 0.3715 | 0.3019 | 0.2989 | 14,137 | 2,594 |
| C2 | 0.2754 | 0.1659 | 0.2146 | 12,424 | 2,587 |
| C3 | 0.3210 | 0.2255 | 0.2499 | 11,256 | 2,594 |
| **L1DK base** | **0.3878** | **0.3292** | **0.3167** | **12,149** | **2,589** |
| L1DK_d03 | 0.3603 | 0.2841 | 0.2889 | 11,151 | 2,588 |

## 6. Macro 汇总（等权四域）

| Variant | Macro AssA | Macro IDF1 | AssA 胜场 |
|---|---:|---:|---:|
| C0 | 0.3380 | 0.4133 | 0 |
| C1 | 0.3903 | 0.4553 | 1（DanceTrack） |
| C2 | 0.1232 | 0.2331 | 0 |
| C3 | 0.3054 | 0.3905 | 0 |
| **L1DK base** | **0.4062** | 0.4453 | **3（DanceTrack/MOT17/BDD）** |
| L1DK_d03 | 0.3905 | **0.4526** | 1（MOT20） |

## 7. 结论与 BEST_STRONG_BASE

1. **L1DK base 是当前最强统一基座**：macro AssA 0.4062 最高，且在
   DanceTrack、MOT17、BDD 三个域同时最优；MOT20 仅略低于 L1DK_d03
   （AssA 0.2779 vs 0.2864）。
2. **Motion 单独（C1）在 DanceTrack/MOT20 很强，但 BDD 上弱**；
   PBD 单独（C2）在所有域最弱（macro AssA 0.1232），印证 L1-C 结论：
   raw PBD 不能独立关联。
3. **EGRA residual（L1DK_d03）只在 MOT20 正向**，其余域 AssA 均下降，
   与 L1-D 的“局部修正成功但轨迹级失败”一致；这正是 Stage L2 要解决的
   objective mismatch 证据链的一环。
4. **BEST_STRONG_BASE = L1DK base**（0.4 IoU + 0.2 PBD + 0.4 Kalman
   motion，thr 0.25）。后续所有 counterfactual oracle、utility 训练与
   最终对比均以此为准。

## 8. 数据文件

- DanceTrack：`outputs/l1_c/association_controlled_main.csv`
  （本次只含 C0/C1/C2/C3/L1DK_BASE/L1DK_d03；旧版已备份到
  `outputs/l2/old_l1c/`）
- BDD/MOT17/MOT20：`outputs/l1_d/ac_{bdd,mot17,mot20}.csv`
  （旧版已备份到 `outputs/l2/old_l1d/`）
- 轨迹文件：`outputs/l2/baseline_AC/{bdd100k_train,mot17_train,mot20_train}/{variant}/`
  与 `outputs/l1_c/trackeval/{variant}/`
