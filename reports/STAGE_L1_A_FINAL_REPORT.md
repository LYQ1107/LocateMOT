# Stage L1-A Final Report

生成时间：2026-08-08 15:16  |  Stage decision: **L1_A_FAIL_TEMPORAL_VALUE_NOT_PROVEN**

## 1. Executive Summary

在固定 detections 下比较 T0 IoU → T6 trajectory-aware association。T6 vs T0：AssA -19.42 pp，IDSW 578.3%，HOTA 4.87 pp。LocateAnything Recall@0.5 = 0.9166。

## 2. Why Two-Frame B6 Was Not Enough

L0-D held-out：B6 conditional +3.5pp、hard +5.7pp、IDSW -19.2%，但 HOTA/AssA 基本持平且 5-8 targets 仍低于 IoU。因此本阶段把关联从两帧升级为全视频轨迹感知。

## 3. Scientific Question

在固定 detections 下，trajectory history + motion prediction + short-term/anchor memory + lost/reactivation 是否能在真实连续视频中显著减少 IDSW 并提升 HOTA/AssA/IDF1。

## 4. Frozen L0-D Basis

LocateAnything-3B (commit 783f656d) 与 B6 (outputs/l0_d/checkpoints/b6/best.pt) 全程冻结；B6 作为 local association kernel。

## 5. 2025-2026 Literature and GitHub Audit

完整审计见 docs/l1_a_reference_audit.md；主要参考 FDTA (CVPR 2026, MIT, b3b3b778)、MOTIP (CVPR 2025, MIT, ffc0e905)、MeMOTR (ICCV 2023)、OC-SORT (MIT, 8462e7e7)、MOTR；MATR (arXiv:2509.21715) 记录为 NO VERIFIED OFFICIAL CODE FOUND。

## 6. DanceTrack Protocol

train 40 / val 25 / test 35；本阶段固定 32 train + 8 calibration，official val 25 全程 held-out。

## 7. Dataset Split

seed 20260806，video-level disjoint；calibration 按 GT density 低/中/高 2/3/3 选取。

## 8. Detection Protocols

D-LA：LocateAnything-3B person query（calibration 固定 'person.'）；D-CTRL：ByteTrack 官方 YOLOX-X DanceTrack 权重 + OC-SORT 官方推理。

## 9. LocateAnything Detection Quality

| query | Recall@0.3 | Recall@0.5 | Recall@0.7 | Precision | cand/frame | FPS | peak VRAM |

|---|---:|---:|---:|---:|---:|---:|---:|

| d1 | 0.9411 | 0.9166 | 0.7911 | 1.0356 | 21.878 | 0.32 | 10.69 |

| d2 | 0.6018 | 0.537 | 0.4282 | 1.5721 | 14.411 | 0.24 | 10.69 |

| d3 | 0.9411 | 0.9152 | 0.7764 | 1.0403 | 21.778 | 0.33 | 10.69 |


## 10. Shared Birth/Lifecycle Infrastructure

Birth is shared evaluation infrastructure, not a proposed component. unmatched det -> tentative -> min_hits=3 -> ACTIVE；max_age=30。所有 T0-T6 相同。

## 11. T0 IoU

T0 IoU：HOTA 0.40，DetA 0.38，AssA 0.42，LocA 0.88，MOTA 0.37，MOTP 0.89，IDF1 0.41，IDSW 879.00。

## 12. T1 Motion Baseline

T1 Motion Baseline：HOTA 0.38，DetA 0.36，AssA 0.40，LocA 0.88，MOTA 0.34，MOTP 0.89，IDF1 0.39，IDSW 759.00。

## 13. T2 B6 Local

T2 B6 Local：HOTA 0.25，DetA 0.37，AssA 0.16，LocA 0.87，MOTA 0.35，MOTP 0.89，IDF1 0.22，IDSW 3329.00。

## 14. T3 Trajectory Context

T3 Trajectory Context：HOTA 0.25，DetA 0.39，AssA 0.17，LocA 0.87，MOTA 0.36，MOTP 0.89，IDF1 0.23，IDSW 3525.00。

## 15. T4 Motion-Aware Update

T4 Motion-Aware Update：HOTA 0.36，DetA 0.59，AssA 0.22，LocA 0.88，MOTA 0.56，MOTP 0.88，IDF1 0.31，IDSW 4107.00。

## 16. T5 Memory

T5 Memory：HOTA 0.36，DetA 0.60，AssA 0.21，LocA 0.88，MOTA 0.57，MOTP 0.89，IDF1 0.30，IDSW 4137.00。

## 17. T6 Lost/Reactivation

T6 Lost/Reactivation：HOTA 0.45，DetA 0.89，AssA 0.23，LocA 0.87，MOTA 0.84，MOTP 0.88，IDF1 0.35，IDSW 5962.00。

## 18. Architecture Summary

T3: TrajectoryEncoder(2-layer causal temporal transformer, K=8, raw-space fusion) -> frozen B6。
T4: + MotionPredictor(2-layer MLP, dx/dy/dw/dh, SmoothL1) + bounded motion residual。
T5: + MemoryFusion(anchor + EMA, 高可信写入)。
T6: + ReactivationResidualHead(lost>=2, trajectory/PBD similarity + motion-weighted IoU)。

## 19. Training Setup

仅训练 TrajectoryEncoder/MotionPredictor/MemoryFusion/residual heads + nm_bias；B6 冻结；AdamW lr=2e-4，bf16，SmoothL1 motion loss lambda=0.1。

## 20. Main Full-Video Results (D-LA val)

| variant | HOTA | DetA | AssA | LocA | MOTA | MOTP | IDF1 | IDP | IDR | IDSW | FP | FN | Frag | MT | PT | ML |

|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

| T0 | 0.40 | 0.38 | 0.42 | 0.88 | 0.37 | 0.89 | 0.41 | 0.74 | 0.28 | 879 | 1398 | 140477 | 1635.0 | 32 | 146 | 95 |

| T1 | 0.38 | 0.36 | 0.40 | 0.88 | 0.34 | 0.89 | 0.39 | 0.74 | 0.26 | 759 | 1328 | 145651 | 1504.0 | 28 | 135 | 110 |

| T2 | 0.25 | 0.37 | 0.16 | 0.87 | 0.35 | 0.89 | 0.22 | 0.40 | 0.15 | 3329 | 1157 | 142191 | 2313.0 | 25 | 163 | 85 |

| T3 | 0.25 | 0.39 | 0.17 | 0.87 | 0.36 | 0.89 | 0.23 | 0.41 | 0.16 | 3525 | 1192 | 139200 | 2371.0 | 29 | 158 | 86 |

| T4 | 0.36 | 0.59 | 0.22 | 0.88 | 0.56 | 0.88 | 0.31 | 0.42 | 0.25 | 4107 | 1721 | 93435 | 4351.0 | 66 | 164 | 43 |

| T5 | 0.36 | 0.60 | 0.21 | 0.88 | 0.57 | 0.89 | 0.30 | 0.41 | 0.24 | 4137 | 1816 | 91305 | 4384.0 | 69 | 161 | 43 |

| T6 | 0.45 | 0.89 | 0.23 | 0.87 | 0.84 | 0.88 | 0.35 | 0.37 | 0.33 | 5962 | 3374 | 26686 | 6145.0 | 203 | 67 | 3 |

## 21. Incremental Ablation

T6 vs T0: AssA -19.42pp, IDSW 578.3%, HOTA 4.87pp；T6 vs T1: AssA -17.03pp, IDSW 5203；T6 vs T2 至少两项更优: True。

## 22. Low-IoU Results

T0 iou_<0.1 acc=0.0，0.1-0.3 acc=1.0；T6 iou_<0.1 acc=0.0，0.1-0.3 acc=0.0。

## 23. Crowd/Density Results

T0 density low/med/high acc=0.994631971613138/0.9913358147229115/0.988786064235166；T6 low/med/high acc=0.9832807570977918/0.9787022193900411/0.9733797271753133。

## 24. Ambiguous Association Results

T0 ambiguous acc=0.9901499411749981 (n=79898)；T6 ambiguous acc=0.9754400021365809 (n=187215)。

## 25. Reactivation Results

T1 events=n/a；T6 events=3754，id_kept=0.06286627597229622，mean_gap=4.29914757591902。

## 26. Sequence-wise Results

见 outputs/l1_a/per_sequence_results_dla.csv。

## 27. LocateAnything vs Controlled Detection

D-CTRL 固定 YOLOX-X 只运行 T0/T1（T2-T6 需要 ObjectToken 特征）；结果见 main_results_ctrl.csv。

## 28. Why Not IoU?

结论由 low-IoU 子集与整体 IDSW 决定：详见第 22 节与第 20 节真实数值，不以口头解释代替实验。

## 29. Failure Cases

见 reports/l1_a_failure_analysis.md（若生成）。

## 30. Resource Usage

见 outputs/l1_a/tracker_runtime_*.json 与 cache meta（peak VRAM ~10.7GB/进程）。

## 31. Scientific Interpretation

以固定 detections 下 association-only 差异解释；不声称 detection 能力。

## 32. Claim Boundary

可以说：full-video trajectory context/motion/memory/reactivation 对 DanceTrack 固定检测的关联影响；不能说：LocateAnything 检测性能、open-vocabulary、跨数据集泛化。

## 33. Stage Decision
L1_A_FAIL_TEMPORAL_VALUE_NOT_PROVEN

## 34. Next Recommended Stage

依据 decision 决定：PASS -> Visual Prompt LoRA / candidate generation；否则 -> 先诊断 ObjectToken 判别力与 trajectory/memory 污染。

## 35. Important Paths

configs/stage_l1_a.yaml；outputs/l1_a/{main_results_dla.csv, per_sequence_results_dla.csv, detection_manifest.json, final_status.json}；reports/STAGE_L1_A_GPT_HANDOFF.md
