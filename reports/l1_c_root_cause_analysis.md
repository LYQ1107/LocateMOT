# Stage L1-C Root-Cause Analysis

## 事实链

1. raw PBD candidate-selection 很强（R@1 0.92 / mAP 0.94，L1-B 也一致），
   但 full-video raw-PBD AssA 只有 0.155（C2）；
2. IoU 在 DanceTrack 上几乎总是选对 candidate（both_correct 89.4%），
   PBD 不互补（pbd_only 25 例）；
3. 96.5% 事件 IoU margin≥0.10，easy 区所有方法 acc 0.86–0.96；
   歧义区（1.7%）所有方法 acc 0.40–0.58；
4. UAF（from-scratch assignment）在 easy 区也低于 IoU
   （0.856 vs 0.960），说明 learned assignment 破坏了强 PBD/IoU 先验；
5. LoRA 适配后 PBD 判别力与 grounding 都下降。

## 根因结论

- association 失败的主因是 **set-level ID 分配与生命周期**，不是
  appearance/detection 判别力：
  - raw PBD 每帧能选对 candidate，但 Hungarian/argmax 在候选与 track
    数量变化时产生大量 NEW/ID 断裂；
  - UAF 的 K+1 CE 训练没能学到“保持正确先验、只在需要时修正”的行为，
    反而全面扰动。
- 次要因素：IoU 歧义（margin<0.05）存在但占比小，需要专门处理；
- 失败模式可由 prediction-side 特征预测（见 learnability probe）。

## 对 L1-D 的含义

- 必须保留强 base affinity（IoU/PBD/motion 融合）；
- 学习器只做“证据门控的 set-level 修正”（residual），不做 from-scratch
  assignment；
- NEW/连续性规则需要显式监督（ID 保持优先）；
- 不再继续 LoRA 主线（Frozen 为基座）。
