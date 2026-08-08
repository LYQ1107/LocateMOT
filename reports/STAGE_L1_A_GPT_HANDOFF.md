# Stage L1-A GPT 交接报告

生成时间：2026-08-08 15:16 | Stage decision: L1_A_FAIL_TEMPORAL_VALUE_NOT_PROVEN

## 1. Stage L0-D 结论

B6 两帧关联：conditional 0.7783 vs IoU 0.7432（+3.5pp），hard +5.7pp，IDSW -19.2%；但 HOTA 0.6592 vs 0.6607、AssA 0.8127 vs 0.8128 基本持平，5-8 targets 仍低于 IoU。

## 2. 为什么转 full-video

两帧模型只使用上一帧 token，无法利用 trajectory；需要全视频实验回答“为什么不用 IoU”。

## 3. GitHub 2025/2026 审计

FDTA (CVPR 2026, MIT, b3b3b778)、MOTIP (CVPR 2025, ffc0e905)、MeMOTR、OC-SORT、MOTR 已读；MATR 无官方代码。

## 4. DanceTrack 规模

train 32 (33,772 帧) / calibration 8 (8,024 帧) / official val 25 (25,508 帧)，video-level disjoint。

## 5. Detection recall
D-LA person query Recall@0.5 = 0.9166

## 6. T0-T6 架构

T0 IoU；T1 OC-SORT 风格 Kalman+OCM；T2 冻结 B6 local；T3 TrajectoryEncoder(K=8)；T4 + MotionPredictor；T5 + anchor/EMA memory；T6 + lost/reactivation。

## 7. HOTA/DetA/AssA/MOTA/IDF1/IDSW 完整表

| variant | HOTA | DetA | AssA | MOTA | IDF1 | IDSW |
|---|---:|---:|---:|---:|---:|---:|

| T0 | 0.40 | 0.38 | 0.42 | 0.37 | 0.41 | 879 |

| T1 | 0.38 | 0.36 | 0.40 | 0.34 | 0.39 | 759 |

| T2 | 0.25 | 0.37 | 0.16 | 0.35 | 0.22 | 3329 |

| T3 | 0.25 | 0.39 | 0.17 | 0.36 | 0.23 | 3525 |

| T4 | 0.36 | 0.59 | 0.22 | 0.56 | 0.31 | 4107 |

| T5 | 0.36 | 0.60 | 0.21 | 0.57 | 0.30 | 4137 |

| T6 | 0.45 | 0.89 | 0.23 | 0.84 | 0.35 | 5962 |

## 8. Low-IoU 结果

DanceTrack val 连续帧低 IoU 事件极少：T0 的 <0.1 仅 2 个、0.1-0.3 仅 1 个、0.3-0.5 仅 100 个（占总连续匹配 86,906 的 0.12%）。T0 vs T6：0.3-0.5 acc 0.92 vs 0.835820895522388；≥0.5 acc 0.9910486964736241 vs 0.9769434655582663。说明本 split+detector 组合下 IoU 失效场景很少，无法用 DanceTrack val 证明 trajectory 价值。

## 9. High-density 结果

T0 low/med/high acc：0.994631971613138/0.9913358147229115/0.988786064235166；T6：0.9832807570977918/0.9787022193900411/0.9733797271753133。T6 在高中低密度都略低于 T0；ambiguous acc T0 0.9901499411749981（n=79898）vs T6 0.9754400021365809（n=187215）。

## 10. Reactivation

T0 events=1545，id kept=0.37346278317152104，mean gap=60.97411003236246；T1 events=1411，id kept=0.4153082919914954；T2 events=2291，id kept=0.2422522915757311；T6 events=3754，id kept=0.06286627597229622，mean gap=4.29914757591902。T6 以 4.3 帧的平均 gap 高频 reactivate，但只有 6.3% 保持了原 ID，说明 reactivation 在抢夺身份而不是恢复身份。

## 11. T6 vs IoU

AssA -19.42pp，IDF1 -6.20pp，IDSW 578.3%，HOTA 4.87pp。

## 12. T6 vs Kalman/OC-SORT

T6 vs T1：AssA -17.03pp，IDSW 5203。

## 13. T6 vs B6-local

T6 vs T2：AssA 6.39pp，IDF1 12.93pp，IDSW 2633；至少两项更优：True。

## 14. 是否真正证明“为什么不用 IoU”
L1_A_FAIL_TEMPORAL_VALUE_NOT_PROVEN

## 15. 剩余最大瓶颈

1) Identity 一致性：T4-T6 的 IDSW 3,329→5,962 远高于 T0 879；AssA 0.163→0.227 低于 T0 0.422。2) Reactivation 无有效身份门控：T6 reactivation id-kept 仅 6.3%。3) DanceTrack val 低 IoU 场景极少（<0.5 仅 0.12%），trajectory 的价值场景没有被该协议覆盖。4) D-LA 检测碎片化（DetA 0.381 且 FN 140k）被 T6 大幅修复，但代价是 ID 混乱；D-CTRL YOLOX 下 T0/T1 的 IDSW 也高达 1,393-1,569，碎片化是比低 IoU 更主要的关联压力。

## 16. 下一阶段唯一建议

不要继续堆 memory/reactivation 复杂度。先做 identity-preserving association 修复：（a）reactivation 增加严格相似度/运动门控并限制每次复活消耗；（b）把 ID-consistency loss（跨帧同 ID 判别）纳入训练；（c）在 D-CTRL 强检测下重跑 T2-T6 以分离检测碎片与关联能力。在 AssA/IDF1/IDSW 三项至少两项明确超过 T0 之前，不进入 Visual Prompt LoRA。

## 17. Detection 质量（D-LA）

LocateAnything-3B person query 在 pilot 90 帧上 Recall@0.3=0.9411，Recall@0.5=0.9166，Recall@0.7=0.7911，candidates/frame=21.878，FPS=0.32，peak VRAM=10.69GB。d1（'person.'）与 d3（'people.'）接近且远好于 d2（'a person.'）。

## 18. D-CTRL 结果（YOLOX-X DanceTrack 官方权重）

D-CTRL 只运行 T0/T1（T2-T6 需要 ObjectToken 特征，未在 YOLOX 框上做特征迁移）：T0 HOTA=0.30，DetA=0.36，AssA=0.25，IDF1=0.30，IDSW=1569；T1 HOTA=0.29，AssA=0.24，IDF1=0.30，IDSW=1393。固定强检测下 IDSW 依然上千，说明检测碎片本身是关联压力的主要来源之一。

## 19. 训练设置与稳定性

TemporalBundle（TrajectoryEncoder K=8 两层因果 transformer + MotionPredictor MLP + MemoryFusion + Motion/Reactivation residual heads + nm_bias）在 32 个 train 视频、每视频 250 个 hard 样本上训练 2 epochs × 8,000 steps = 16,000 steps；AdamW lr=2e-4、weight_decay=1e-4、grad clip 1.0、cosine schedule；B6 全程冻结。loss 主体从 ~6 降到 ~1.2-2.5；部分困难样本出现有限大 spike（最高 ~1,500，BCE 对极端 logit），未产生 NaN。训练中修复了两个数值 bug：degenerate box 的 geometry inf（normalize_geom clamp）与轨迹注意力 padding 位置 NaN（对角线自注意力）。

## 20. 推理速度与资源

D-LA val 25,508 帧：T0 405-672 FPS、T1 233-615 FPS、T2 9-26 FPS、T3 5-16 FPS、T4 4-12 FPS、T5 5-10 FPS、T6 3-9 FPS（不同轮次波动）。缓存阶段 LocateAnything 3B 约 0.3 FPS/GPU、峰值显存 ~10.7GB；D-LA 全量缓存 67,304 帧约 14-20GB；系统内存全程控制在 65-117GB 可用，无新的 OOM。

## 21. 科学解释与边界

T4-T6 通过 motion/reactivation 显著降低了 FN（T6 FN=26,686 vs T0 140,477）并提高 MOTA/DetA，但代价是 AssA/IDF1/IDSW 全面劣化：learned 关联把‘多检测到’与‘保持同一身份’变成了竞争关系。这不能证明 trajectory modeling 无用，只能证明当前实现（尤其是 reactivation 的 ID 复用策略）没有做好 identity 一致性。结论边界：仅适用于 DanceTrack val + D-LA person 检测协议；不能外推到其他数据集或 open-vocabulary 场景；没有改动 LocateAnything/B6 权重。

## 22. 遗留问题与后续验证

（1）D-CTRL 下 T2-T6 未跑，无法完全分离 detection 与 association 的贡献；（2）低 IoU 事件不足，需要 MOT17/MOT20 或故意稀疏检测（drop 帧）协议来制造 trajectory 必须的场景；（3）reactivation 的相似度门控与 ID 一致性损失是第一优先修复项；（4）训练 loss 的大 spike 值得做 loss 截断/标签清洗后再训练。

## 23. D-LA 完整指标表（官方 TrackEval，val 25 视频）

| variant | HOTA | DetA | AssA | LocA | MOTA | MOTP | IDF1 | IDP | IDR | IDSW | FP | FN | Frag | MT | PT | ML |

|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

| T0 | 0.40 | 0.38 | 0.42 | 0.88 | 0.37 | 0.89 | 0.41 | 0.74 | 0.28 | 879 | 1398 | 140477 | 1635.0 | 32 | 146 | 95 |

| T1 | 0.38 | 0.36 | 0.40 | 0.88 | 0.34 | 0.89 | 0.39 | 0.74 | 0.26 | 759 | 1328 | 145651 | 1504.0 | 28 | 135 | 110 |

| T2 | 0.25 | 0.37 | 0.16 | 0.87 | 0.35 | 0.89 | 0.22 | 0.40 | 0.15 | 3329 | 1157 | 142191 | 2313.0 | 25 | 163 | 85 |

| T3 | 0.25 | 0.39 | 0.17 | 0.87 | 0.36 | 0.89 | 0.23 | 0.41 | 0.16 | 3525 | 1192 | 139200 | 2371.0 | 29 | 158 | 86 |

| T4 | 0.36 | 0.59 | 0.22 | 0.88 | 0.56 | 0.88 | 0.31 | 0.42 | 0.25 | 4107 | 1721 | 93435 | 4351.0 | 66 | 164 | 43 |

| T5 | 0.36 | 0.60 | 0.21 | 0.88 | 0.57 | 0.89 | 0.30 | 0.41 | 0.24 | 4137 | 1816 | 91305 | 4384.0 | 69 | 161 | 43 |

| T6 | 0.45 | 0.89 | 0.23 | 0.87 | 0.84 | 0.88 | 0.35 | 0.37 | 0.33 | 5962 | 3374 | 26686 | 6145.0 | 203 | 67 | 3 |

逐级解读：T2 在两帧 B6 迁移到全视频后 AssA 0.163/IDSW 3,329 明显差于 T0（0.422/879），说明 B6 的 no-match/ID 行为在连续视频中碎片化；T3 加入 trajectory 后仅把 DetA 从 0.372 提到 0.386，AssA 基本不变；T4 的 motion 把 MOTA 从 0.361 拉到 0.559（FN 从 139k 降到 93k），但 IDSW 升到 4,107；T5 memory 与 T4 持平；T6 reactivation 把 MOTA 推到 0.840、DetA 0.888、FN 26,686，但 AssA 0.227、IDF1 0.349、IDSW 5,962 全面劣于 T0。
