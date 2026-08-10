# Stage L1-C Cue Disagreement Taxonomy

对每个 association event 离线标记：
A both_correct（IoU 选对且 PBD 选对）；B pbd_only（PBD 对、IoU 错）；
C iou_only（IoU 对、PBD 错）；D both_wrong。

## DanceTrack val（225,071 events，各方法一致）

| 类别 | 数量 | 占比 |
|---|---:|---:|
| both_correct | 201,197 | 89.4% |
| iou_only_correct | 23,827 | 10.6% |
| pbd_only_correct | 25 | 0.01% |
| both_wrong | 22 | 0.01% |

## 解读

1. **PBD 不互补于 IoU**：IoU 选错时 PBD 也错（25/23,849 ≈ 0.1%）；
2. IoU 是 DanceTrack 上压倒性的 candidate-selection cue；
3. association 的真正损失来自 ID 连续性/集合分配，而不是
   “该信 IoU 还是 PBD”的 cue 选择问题；
4. 因此 L1-D 不应做成简单的 pairwise reliability gate 选择 PBD/IoU；
   应重点修复 set-level 分配与 NEW/生命周期导致的 ID 断裂。
