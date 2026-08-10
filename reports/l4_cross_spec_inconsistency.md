# Stage L4 — Cross-Spec Identity Inconsistency

日期：2026-08-10。

## 1. 问题

同一个 frozen U0，对同一视频跑 ALL 与受限候选流，公共对象上的
track identity 会因候选子集变化而改变：

- BDD：car 33%、truck 40%、bus 43%、pedestrian 49%、
  trailer 67% 的 common-object 观测在最优 ID 映射后仍不一致；
- DanceTrack：person 32%、top-2 instance 31% 不一致。

## 2. 为什么身份会漂移（机制判断）

1. **Hungarian 竞争**：全候选流中不同类别的 distractor 占用轨迹槽，
   受限流去掉了竞争后匹配结果不同；
2. **set context**：EGRA 的 pair/track/cand 特征包含
   `log_n_cand / margins / top1` 等集合级特征，候选集合变化直接改变
   模型输入；
3. **track-state 更新**：两视图的轨迹生命周期（birth/lost/terminate）
   不同步，公共对象在其中一个视图可能被错误匹配后污染后续状态；
4. **P0 过滤后产生缺失帧**：Track-All-Then-Filter 输出在受限对象上
   有 gap，TrackEval 的 IDSW 会把这些 gap 记为 switch。

## 3. P0 vs P1 方向

- P1 通常改善受限视图的 AssA/IDSW（尤其 instance subset）；
- 但 P1 的 identity 与 ALL 不一致，意味着「用户换一种询问方式」会得到
  一套不同的轨迹编号语义；
- 因此需要的不是选 P0 或 P1，而是让 shared identity process 对候选
  子集稳定（restriction-equivariant）。

## 4. 训练必要性

该不一致不是 evaluation bug（ALL vs ALL 自检一致、toy-case 指标验证
通过），因此进入 Stage L4-B：paired-view spec-equivariant training。
