# Stage L1-D Full Tracker Report

日期：2026-08-10。

## 结论

**未执行。** 任务书 §58 规定 Full Tracker Protocol 仅在
Association-Controlled 证明有效后执行；L1-D AC pilot 未通过
（residual < base on DanceTrack val），因此不混入 lifecycle/DetA
变化，避免归因混乱。

## 预备状态

- `OnlineTracker` 已支持 `L1D` variant（含 Kalman 生命周期与
  `--ac` 开关），去掉 `--ac` 即 full-tracker 模式（min_hits=3、
  max_age=30、只输出 ACTIVE 轨迹）；
- 建议 L2 以 L1DK base 先做 full tracker（无训练参数），再决定
  是否加入 residual。

