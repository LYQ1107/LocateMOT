# THREE_TASK_TRAINING_COVERAGE（首轮，2026-10-18 03:00）

## 数据接通状态（区分六个层级）

| 层级 | MOT | RMOT | OVMOT |
|---|---|---|---|
| `DATASET_DOWNLOADED` | ✅ | ✅ Refer-KITTI-V2 / AerialMind / RMOT26 | ✅ TAO / BURST |
| `SOURCE_INDEXED` | ✅ 2,664 帧 | ✅ 4,350 fit 查询 | ✅ 18,274 图 / 318,200 mask |
| `NATIVE_CANDIDATES_EXPORTED` | ✅ 2,288 帧（native_G0） | ❌ 尚未导出（首轮用 GT_teacher） | ❌ 尚未导出（首轮用 GT_teacher） |
| `SFT_ROWS_BUILT` | ✅ 131,214 行 | ✅ 91,935 行（fit+calibration） | ✅ 1,554 行（600 图） |
| `ACTUAL_OPTIMIZER_UPDATES` | ✅ | ✅ | ✅ |
| `HELDOUT_EVALUATED` | ⏳ DT1203 进行中 | ❌ | ❌ |

## 首轮三任务文件

`outputs/groundjev/full_data_v2/three_task_sft_v1/rows.jsonl`

| 任务 | 行数 | 占比 | proposal_source |
|---|---:|---:|---|
| MOT identity | 30,000 | 68.9% | `native_G0`（真实 GroundAnything 候选） |
| RMOT | 12,000 | 27.6% | `GT_teacher`（Refer-KITTI 合法 fit 标签） |
| OVMOT | 1,554 | 3.6% | `GT_teacher`（TAO Base 类别） |
| 合计 | 43,554 | 100% | — |

覆盖 37 个不同视频。**检测器归因必须按 `proposal_source` 分开报告**：MOT 行来自真实候选，RMOT/OVMOT 行来自 GT 教师区域。

## 训练

`trackjev_3task_J1_v1`：2 卡 DDP，每步 12 行（2 ranks × 6），LoRA r=32，`--max-len 32768`，起始 loss 4.16 → 1.25（前 5 步），slot_acc 0.17 → 0.33。首个 checkpoint 在 step 100。

## 已知缺口（下一轮补）

1. OVMOT 仅占 3.6%，需要把 TAO 图像扩到 2,000+，并接入 BURST 的 mask→bbox。
2. RMOT/OVMOT 仍无原生候选（GT_teacher），下一步对 Refer-KITTI 1,814 帧与 TAO 抽样跑 G0 导出。
3. AerialMind（93 段 / 24,617 官方查询）尚未进入行生成。

## 2026-10-18 21:25 CST — v2b birth-supervised continuation

The active trainer is no longer the early v1 mix described above. The
recoverable v2b input is
`outputs/groundjev/full_data_v2/three_task_sft_v2b/rows.jsonl` with 43,533
rows: MOT identity 21,168, RMOT 16,677, and OVMOT 5,688. Provenance is
`native_G0` for the 21,168 MOT rows and `GT_teacher` for the 22,365 RMOT/OVMOT
rows. Birth rows were added from real causal empty-track states before this
run; no held-out video was added. `trackjev_3task_J1_v4_birth` is active on
GPUs 4 and 6 at log step 280/1,200, with the latest complete FP32/Adam/RNG
restore state at step 250. This is an in-progress engineering milestone;
independent held-out MOT/RMOT/OVMOT scores remain pending.
