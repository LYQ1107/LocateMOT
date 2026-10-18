# EXECUTION_REDIRECTION_20261018

执行时间：2026-10-18 02:15 CST
依据：用户在 2026-10-18 下发的调度指令（终止 MOT-only 长训练、保护 MOT20 导出、推进三任务与检测器适配）

## 0. 审计后的真实状态（与历史描述不一致处已更正）

| 项 | 历史描述 | 实测 |
|---|---|---|
| T1 三卡 J1 LoRA | "已运行约 2 小时" | 实际 step 60 / 1,080 样本 / 2h16m；`save-every=200`，**从未写出 checkpoint** |
| T1 checkpoint | 有 | **无**（目录里只有 args.json、log.jsonl） |
| T2 MOT20 导出 | "运行约 7 小时" | **实际没有进程在跑**（我为跑 DT1203 评测停掉了 runner）。已完成 MOT20-02 1,392 帧；03 321 帧；05 26 帧；01 未开始 |
| T4 DT1203 评测 | — | 398/1,203 帧，GPU7 运行中，已保护 |

## 1. T1 处置：停止，且如实记录"无权重"

- 发出 SIGTERM 给 torchrun 父进程；三个 rank 全部退出，GPU4/5/6 回收至 4 MiB。
- **不存在可保留的 adapter / optimizer / RNG 状态**，因此不存在"删除 2 小时成果"的问题。
- 记录文件：`/data3/user/LocateMOT-runs/TrackJEV_FullDataV2/trackjev_J1_ddp3_r6/STOPPED_WITHOUT_CHECKPOINT.json`
- 可用的科学信息只有 loss 轨迹（2.66 → 1.80，slot_acc 0.08 → 0.54，60 步），标注为工程数据点。
- 状态登记：`MOT_ONLY_PARTIAL_BASELINE`，**无权重可用于推理或初始化**。

## 2. T2 处置：继续，且用前缀恢复而不是重跑

- MOT20-03：用 `--resume-from mot20_MOT20-03` 恢复到新目录 `mot20_MOT20-03_resumed`，保留已完成的 321 帧，随后继续。已在 GPU5 上运行（解码器 8107 + 特征 8117）。
- MOT20-05：同样方式恢复（26 帧前缀）到 `mot20_MOT20-05_resumed`，在 GPU7 上运行（解码器 8109 + 特征 8112）。
- MOT20-02：1,392 帧已完成，**不重跑、不覆盖**。
- 旧的 `candidate_export_v1` / `candidate_export_v2` 全部保留，作为 G0 证据。

## 3. 资源快照（02:15）

- GPU0-3：其他用户作业（PRISM/FSTT_DA），不触碰。
- GPU4、GPU6：空闲（4 MiB），留给三任务 SFT / G1。
- GPU5：MOT20-03 恢复导出（64%，21 GB）。
- GPU7：DT1203 J0 评测（100%，62 GB）+ MOT20-05 恢复导出。
- /data3 243 G 可用；/data2 891 G 可用。已把 AerialMind/SportsMOT/AnimalTrack 及其原始包移至 `/data2/user/LocateMOT_FullDataV2_spill/` 并用软链保持原路径。

## 4. 下一步（按指令优先级）

1. RMOT/OVMOT 训练行接通（teacher-forced 与 native-proposal 两种来源分别标记 `proposal_source`）。
2. 三任务 OneJev SFT 实际 optimizer updates（同一物理身份主干）。
3. G0/G200/G1 检测器对照与 G1 多域微调。
4. 多问题共享视觉计算的效率优化（不阻塞上述训练）。

## 5. 当前复核（2026-10-18 21:15 CST）

- DeepSeek Harness 本身已经完成最近一轮；其 detached 远程任务仍在继续。
- 旧 T1 MOT-only 任务没有残留进程。当前 `/data3/user/LocateMOT-runs/TrackJEV_FullDataV2/trackjev_3task_J1_v4_birth` 是真实三任务训练，不应按旧名称误停：43,533 行中 identity 21,168、RMOT 16,677、OVMOT 5,688；GPU4/6，step 260/1,200，step 250 的 FP32 master/Adam/RNG 状态已保存。
- MOT20-05 G0 导出仍受保护，`mot20_MOT20-05_resumed` 已到 1,558/1,658 帧；禁止重跑、覆盖或与其抢占 8109/8112 服务。
- G1 已有完整 200/200 projector 训练和 G0 对照，结果没有超过 G0，作为负结果保留，不重复占用 GPU5。GPU0–3、5 当前属于其他用户的 `/data2/OCD/TALON` 作业，未触碰。
- RMOT 的下一步已在 CPU 上准备完成：`candidate_export_rmot_expression_v1/plan.json`，24 个 legal TRAIN 查询、12 个视频、4,416 帧。它不覆盖失败的 pedestrian-only `candidate_export_rmot_v2`，等 MOT20 释放共享服务后再进行模型推理。

当前目标仍为 `IN_PROGRESS`。本段没有声称检测器提升、正式 HOTA 或统一任务完成。
