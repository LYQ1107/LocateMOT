# Stage L7 RMOT Protocol（工作版，接口已审计）

## 主 benchmark 选择

Refer-KITTI / Refer-KITTI-V2（TempRMOT，arXiv 2406.05039）：

- 数据本地部分可用：`/data1/LWR/vranlee/MFT2025/REFER-MFT25/refer-kitti-v2`
  含官方 expression JSON（label: frame→object_ids、sentence、
  video_name="KITTI_N"）与 labels_with_ids（image_02/{seq_id}/）；
  KITTI 帧在 `REFER-MFT25/{train,test}/<SEQ>/img1`（SN/BT/MSK/PF 前缀序列，
  需要与 KITTI seq id 的映射核对后才能用）。
- 官方协议：HOTA / DetA / AssA / DetRe / DetPr / AssRe / AssPr / LocA
  （TempRMOT TrackEval 分支）；expression 条件化的多目标跟踪。
- 帧目录要求（TempRMOT）：`KITTI/training/image_02/{seq:04d}/{frame:06d}.png`。

## STORM-Bench（2026，备选，暂不执行）

- 官方 benchmark repo 只有数据（storm-bench.json），需要 VidOR 帧
  （本地没有，磁盘紧张不下载）；模型代码未发布（PAPER_ONLY）。

## 我们的接口（与 OVMOT 共享 WHAT/HOW 分解）

```text
referring text -> frozen language encoder (RoBERTa 风格) -> spec embedding
    -> candidate relevance (cosine / cross-attention) -> shared UIDM
```

- 首版只做标准 RMOT（属性/空间关系类 expression），不做 ReaMOT 级
  隐式推理；GT-selected oracle diagnostic 区分 grounding 与 identity
  两个 bottleneck。
- 语言编码器可参照 TempRMOT：frozen RoBERTa + 文本词特征与视觉
  cross-attention；我们只借鉴接口（TempRMOT 无 LICENSE，不复制代码）。

## 执行条件

OVMOT 出现正信号后自动进入；数据先解决 SN→KITTI seq 映射，否则下载
官方 KITTI tracking 帧（~5GB，磁盘可行）。

