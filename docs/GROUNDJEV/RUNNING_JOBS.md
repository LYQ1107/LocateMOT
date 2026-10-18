# RUNNING_JOBS（2026-10-18 03:00 CST）

## 受保护的长任务

| 任务 | 进程 | GPU | 进度 | 状态 |
|---|---|---|---|---|
| DT1203 J0 零样本评测 | `run_video --backend trackjev --arm B4` | 7 | 523 / 1,203 帧 | 运行中，**保护** |
| MOT20-03 恢复导出 | `run_video --resume-from mot20_MOT20-03` | 5（解码器 8107 + 特征 8117） | 357 帧（含恢复的 321） | 运行中，**保护** |
| MOT20-05 恢复导出 | `run_video --resume-from mot20_MOT20-05` | 7（解码器 8109 + 特征 8112） | 37 帧 | 运行中，**保护** |
| 三任务 OneJev SFT | torchrun 2 ranks | 4, 6 | 见下 | 运行中 |

## 已终止（有记录）

| 任务 | 结果 |
|---|---|
| `trackjev_J1_ddp3_r6`（MOT17-only LoRA，3 卡） | 在 step 60 终止；`save-every=200` 意味着**从未写出 checkpoint**。记录：`/data3/user/LocateMOT-runs/TrackJEV_FullDataV2/trackjev_J1_ddp3_r6/STOPPED_WITHOUT_CHECKPOINT.json`。无可用于推理/初始化的权重。 |

## 已完成、不重跑

- MOT20-02 候选导出：1,392 帧（DONE）
- MOT17 七段候选导出 v2：2,288 帧 → 131,214 训练行（`native_G0`）
- DanceTrack 0001：120 帧；0006：62 帧（子集导出）
- GroundAnything 九任务 retention：9/9 完成（历史证据）

## GPU 归属

- 0–3：其他用户（PRISM / FSTT_DA），不触碰
- 4、6：三任务 SFT（2 ranks）
- 5、7：MOT20 导出 + DT1203 评测

## 2026-10-18 21:15 CST 复核

| 任务 | 当前状态 |
|---|---|
| MOT20-05 G0 导出 | 受保护，`mot20_MOT20-05_resumed` 到 1,558 / 1,658 帧；继续使用 GPU7 上的 8109/8112 服务 |
| TrackJEV J1 v4 birth | GPU4+6，真实三任务 rows.jsonl，step 260 / 1,200；step 250 有完整 FP32/Adam/RNG checkpoint |
| G1 detector adaptation | 已完成 200 / 200；G0 对照为负结果，保留证据，不重跑 |
| expression RMOT pilot | CPU 计划已完成，24 queries / 12 legal videos / 4,416 frames；等待 MOT20 释放共享服务 |

GPU0–3、GPU5 为其他用户作业，未停止或复用。当前没有残留的 MOT-only 长训练进程。
