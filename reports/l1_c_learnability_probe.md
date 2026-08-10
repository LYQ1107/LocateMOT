# Stage L1-C Learnability Probe

目的（diagnostic only）：回答“PBD/IoU 失败模式能否由 prediction-side
features 预测”。模型：logistic regression（balanced）；训练：DanceTrack
calibration（C2_t0.0 事件，68,834）；评估：DanceTrack val（C2 事件，
200k 采样）。特征：iou_top1/iou_margin/pbd_top1/pbd_margin/
num_candidates/obj_size/gap。

| 目标 | 正样本率(train→val) | AUROC | PR-AUC |
|---|---:|---:|---:|
| PBD candidate-selection wrong | 8.5%→10.6% | 0.933 | 0.632 |
| IoU candidate-selection wrong | 0.01%→0.02% | 0.974 | 0.078 |
| raw-PBD ID-continuity wrong | 8.4%→10.7% | 0.915 | 0.575 |

## 解读

1. PBD 选择失败可预测（AUROC 0.93，PR-AUC 0.63，非 0.5–0.55）：
   具备进入“证据门控修正”的条件；
2. IoU 选择失败几乎不发生（0.02%），PR-AUC 低是类别极端不平衡导致，
   AUROC 高但实用意义有限；
3. raw-PBD ID 连续性失败可预测（AUROC 0.91）→ 可学习何时需要修正
   assignment；
4. 结论：**不直接支持“复杂 reliability gate 不值得做”**；相反，
   证据支持一个保守的 evidence-driven 修正（L1-D）。
