# GPU7 资源释放审计（2026-10-19）

结论：已安全释放。释放前 GPU7 使用 43,518 MiB、利用率 0%；释放后使用 0 MiB、利用率 0%，空闲显存 81,154 MiB。端口 `8102`、`8109`、`8112` 均已关闭。

| 服务 | PID / PGID | 用途 | 释放前状态 | 释放后 |
|---|---|---|---|---|
| 8102 | 1510982 / 1510979 | GroundAnything-DLM SGLang 历史标准评测服务 | alive | exited |
| 8109 | 2199499 / 2199497 | MOT20-05 G0 解码器 | alive | exited |
| 8112 | 1529604 / 1529602 | MOT20-05 原生视觉特征 RPC | alive | exited |

释放前核验：MOT20-05 G0 结果为 `VIDEO_INFERENCE_COMPLETE_WITH_REJECTED_PERCEPTION`，1658 帧，结果 SHA256 为 `58c6ba1fc3f36e01e09036f2aa74dae363b20d8653163d4a6243d70f989542e3`；expression RMOT owner 为 `EXPRESSION_RMOT_PILOT_COMPLETE`，24/24 查询完成；8112 receipt 的 `dataset_cache_created=false`，没有全量缓存。进程父 PID 均为 1，未发现活跃 `run_video`、RMOT 队列或自动 supervisor 消费端口。

执行方式：保存释放快照后，对三个已确认属于本项目的进程组定向发送 SIGTERM；在有界等待内全部退出，没有使用 `killall`、`pkill`、SIGSTOP、GPU reset，也没有触碰其他用户 GPU0/1/2 作业。完整机器快照保存在 `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/gpu7_release_audit_20261019.json`。

重启参数仍保留在各自 `launch.json`、8112 receipt 和进程审计中；若后续确需使用，应从新 owner 启动并重新生成独立 receipt，不依赖已经关闭的端口。
