# G2 训练加速审计（2026-10-19）

状态：`ACCELERATION_A_B_COMPLETE_MAIN_RUN_PROTECTED`。主 G2 没有迁移、重启或改变训练协议。

为避免干扰正式任务，在空闲 GPU6/7 上运行了两个真实 G2 训练器 4-step 对照。两者都使用同一 exact-once manifest、seed、token budget、world size=2 和 effective global batch=32，只改变 micro-batch 与梯度累积的分解：

| 对照 | micro-batch | grad accumulation | 平均 step（启动开销后） | 峰值显存/卡 | 结果 |
|---|---:|---:|---:|---:|---|
| `micro2_acc8` | 2 | 8 | 33.67s | 17.43 GiB | 完成 4 step |
| `micro4_acc4` | 4 | 4 | 31.00s | 39.62 GiB | 完成 4 step |

`micro4_acc4` 的步速约为 `micro2_acc8` 的 **1.09×**，但显存峰值约为 **2.27×**。正式 G2 每卡约 17.4 GiB，因此不能把 micro4 配置直接用于当前 GPU3/5；它会把每卡需求推到约 39.6 GiB。

正式主运行的最新静态观测是 step **3590/18227**（2026-10-19 17:32:43 +0800），最近 500 个间隔平均 **15.10s/step**，中位数 **15.00s**，p95 **16.00s**；当前协议下估算约 **2.12 global rows/s**。最新完整可恢复 checkpoint 仍为 step3000；主训练持续运行在 GPU3/5。

四卡迁移本轮没有强行启动：GPU5 属于受保护的正式 G2，且严格恢复 payload 的 `world_size=2` 与 trainer contract 不能无缝改成 4。只有正式任务结束、另建 world_size=4 合同并在空闲 GPU上做完整对照，才可评估四卡收益。两个隔离对照的完整 JSON receipt 和日志位于 `/data3/user/LocateMOT-runs/G2_perf_20261019/`。
