# lwr → user Migration Handover

Migration:
lwr -> user

Project:
/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT

## Previous state（迁移前最后阶段）

Stage L1-B 已归档：`L1_B_IDENTITY_SIGNAL_NOT_SUPPORTED`。
“继续”阶段：road multi-class 方向验证（BDD+TAO 扩展缓存 → LODO 训练/评估）。
迁移前最后操作：启动 8 个 road cache 分片（BDD 8,000 帧 + TAO 4,000 帧），
随后用户要求暂停新任务，已写 PROGRESS.md 并结束上一轮。

## Active processes inherited

迁移审计结果：PROGRESS.md 中记录的 8 个缓存进程
（PID 16318/16377/18115/18173/18231/18282/18340/18406）**已不存在**。

- 状态判定：INTERRUPTED（日志无 finished 行、GPU 已空、进程消失）。
- 进度证据：BDD 8,001/8,000 帧完成；TAO 1,022/4,000 帧完成；
  日志停在 runs/l1b_road_cache_shard{0..7}.log（~1080–1100/1500）。
- 恢复方式：cache 工具按 `.complete` 断点续跑，无需重算已完成帧。

## Completed work inherited

- L1-B0：方法审计（OG-ReID/VICP/UPCL/UniTrack）、数据集审计（含 BDD100K
  masa box_track_20 更正）、统一 schema、存储计划。
- Pilot v1/v2：cache（2,932 帧）、R0–R3 raw 基线、Identity Adapter
  （InfoNCE full+pbd）训练与 R4 评估。
- 结论：v2 full adapter 在 DanceTrack +6.5pp / BDD +3.0pp / TAO +8.7pp，
  dense/deformable 下降，macro +1.0pp → gate 未通过。
- Road 扩展配置：configs/l1_b/road_v2_videos.json（BDD 200 视频 8,000 帧；
  TAO 100 视频 4,000 帧）。

## Environment

- 继续使用 lwr 的 conda 环境（user 可读可执行，经审计等价可用）：
  /home/lwr/anaconda3/envs/locatemot/bin/python
- torch 2.5.1+cu124 / transformers 4.57.1 / python 3.12.2 / CUDA True
- PATH 已包含 /home/lwr/anaconda3/bin（无需切换）。

## Path compatibility

- 项目与数据路径（/data1/LWR/vranlee/...、/data3/testdata/vranlee/...）
  保持不变，user 可读可写。
- /home/lwr/... 引用：仅环境（anaconda）与缓存；继续沿用原路径，未做
  全局替换，未重建环境。
- git safe.directory 已添加（仅 git 配置，不影响仓库）。

## Git state

- HEAD: d5817aa6e76b8e60c4e7d0907b9284cf55eacc1d（Stage L1-A 提交）
- 未提交修改：AGENTS.md、reports/LATEST_GPT_HANDOFF.md（M）
- 未跟踪 L1-B 成果：PROGRESS.md、configs/l1_b/、docs/l1_b_*、
  locatemot/models/identity/、reports/l1_b_*、research_log.md、
  tools/l1b_*（保留，不 reset/checkout/clean）

## Next executable step

1. 恢复 road cache（TAO 剩余 ~2,978 帧；8 卡 nohup + 单阻塞命令等待）。
2. LODO 训练：A_bdd（只训 BDD）、A_tao（只训 TAO）、A_road（合并），
   然后跨数据集 Same-Category R@1 评估。
3. 更新 reports/l1_b_lodo_report.md 与最终报告/状态文件。

## Update 2026-08-09 04:20（GPU 阻塞）

恢复尝试 resume1/2/3 均被环境或 GPU 事件中断；缓存进度
BDD 8,001/8,000、TAO 1,390/4,000（.complete 断点保留）。
当前 GPU 不可用：/dev/nvidia* 缺失，nvidia-smi 无法与驱动通信
（/proc/driver/nvidia/version 仍显示 525.105.17）。需主机/容器恢复 GPU
设备挂载后，以已批准命令 `bash scripts/resume_l1b_road_cache.sh` 续跑，
再执行 LODO 训练/评估。
