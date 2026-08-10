# Stage L1-C Baseline Semantic Mapping Audit

日期：2026-08-10。目的：解决历史文档中 C0–C4 命名漂移，确认每个编号对应
的实际方法，并核实“raw PBD cosine 是否就是 AssA≈0.419 的路线”。

## 1. 方法-编号映射（以本阶段代码为准）

| 编号 | 实际方法 | 关联代码 |
|---|---|---|
| C0 | IoU Hungarian（阈值 0.3） | `OnlineTracker._associate_t0` |
| C1 | OC-SORT 风格 motion（7 维 Kalman + OCM 第二轮） | `OnlineTracker._associate_t1` |
| C2 | raw PBD cosine（box-end last，Hungarian；阈值校准后仍不敏感） | `OnlineTracker._associate_pbd` |
| C3 | IoU + raw PBD 固定线性融合（权重/阈值在 calibration 校准） | `OnlineTracker._associate_iou_pbd` |
| C4 | 冻结 B6 local kernel（L0-D checkpoint） | `OnlineTracker._associate_learned`（T2 路径） |
| UA | 冻结 LocateAnything + Unified Association Decoder（50k 步，NEW-margin=3.5） | `locatemot/models/ua_decoder.py` |

## 2. 关键澄清：raw PBD 不是 0.419 路线

- AssA≈0.419 / IDF1≈0.566 / IDSW≈2,916 的路线是 **C1 motion**
  （OC-SORT 风格 Kalman + OCM），不是 raw PBD。
- raw PBD cosine（C2）在 DanceTrack val association-controlled 上只有
  AssA≈0.155 / IDF1≈0.319 / IDSW≈15,616。
- 该混淆源自早期文档中 C1/C2 命名被交换过；本审计以当前代码为准，
  后续所有报告使用 method name 作为主键。

## 3. 校准后的 DanceTrack val 基线（association-controlled）

校准规则：阈值/权重只在 calibration split（8 视频）调整，val 只做最终评估。

| method | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|
| IoU (C0) | 0.608 | 0.947 | 0.390 | 0.855 | 0.894 | 0.529 | 3,554 |
| Motion (C1) | 0.630 | 0.947 | 0.419 | 0.855 | 0.897 | 0.566 | 2,916 |
| Raw PBD (C2) | 0.384 | 0.947 | 0.155 | 0.842 | 0.836 | 0.319 | 15,616 |
| IoU+PBD (C3) | 0.610 | 0.947 | 0.393 | 0.849 | 0.896 | 0.537 | 2,981 |
| B6 (C4) | 0.383 | 0.948 | 0.155 | 0.851 | 0.836 | 0.308 | 16,456 |
| UAF (UA) | 0.355 | 0.947 | 0.133 | 0.850 | 0.787 | 0.270 | 26,804 |

注：DetA≈0.947 全方法一致（AC 协议有效，≤0.001 tie-break 差异）。

## 4. C2/C3 校准记录

- C2 raw PBD：calibration 上阈值 0.0/0.2/0.3/0.4 结果完全相同
  （AssA 0.140 / IDSW 4,190）——PBD 余弦在本协议下阈值不敏感，
  采用无阈值 Hungarian（threshold=0.0）。
- C3 IoU+PBD：calibration 网格
  w_iou∈{0.3,0.5,0.7} × w_pbd={1-w_iou} × thr∈{0.2,0.3}；
  最优 w_iou=0.7 / w_pbd=0.3 / thr=0.3（AssA 0.384 / IDSW 573）。
  目标：AssA 优先；与次优（0.7/0.3/0.2，AssA 0.378）差异 0.006。

## 5. 结论

- raw PBD 在 DanceTrack AC 协议下显著弱于 IoU/motion；
- motion 是最强传统基线（C1），不是 PBD；
- 简单 IoU+PBD 融合（C3）接近 IoU，IDSW 略优；
- UAF 低于全部传统基线，pilot gate 不通过（维持原结论）。
