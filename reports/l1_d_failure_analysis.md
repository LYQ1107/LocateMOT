# Stage L1-D Failure Analysis

## 1. 为什么 residual 在 val 不迁移

1. base 身份正确率只有 44%（大量历史 swap 状态），需要修正的行中
   ~50% 需要 delta>0.6（真实候选 IoU≈0），超出 0.6*tanh 的修正能力；
2. 模型在 calibration 上把残余风险也学会了（+1.9pp AssA），但该
   分布与 val 的 swap/遮挡模式不同，val 上修正变为微害
   （AssA −1.7pp）；
3. reliability gate 的监督标签是“base 行是否身份正确”，在 val 上
   gate 输出与 AssA 收益不对齐。

## 2. 为什么连续性改善不等于 TrackEval 收益

- 校正审计（continuity acc +11pp on val）与 TrackEval IDSW/Frag 变化
  不一致：TrackEval 的断裂大多计入 Frag 而非 IDSW；Frag 在 base→L1D
  几乎不变（5,209→5,217）。
- L1D 改变的部分匹配把少量正确配对换成了连续性相同但身份不同的配对，
  损害 HOTA 的 association 匹配（AssA）。

## 3. 与 L1-C 证据链的关系

- learnability probe（AUROC 0.93）预测的是“PBD 选择/ID 连续性失败”，
  不是“修复后 AssA 提升”；
- 因此可预测性成立 ≠ 可修正性成立；本次实验证明后者不成立
  （在 DanceTrack/BDD 上）。

## 4. 保留什么

- 保留：L1DK base（校准 Kalman IoU+PBD 融合，val AssA 0.4165 /
  IDSW 2,558，优于 C1/C3）；
- 不保留：EGRA residual 作为统一部署模块；
- 保留为消融：MOT20 上 residual 的正向证据（说明 crowd 域可能
  需要不同的 gate 校准，但违背 one-checkpoint unified 目标）。

