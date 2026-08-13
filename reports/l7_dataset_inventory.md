# Stage L7 Dataset Inventory（2026-08-14）

原则：不修改原始数据；已存在则软链接；不使用 MOTSynth；不使用带
`.MOTSynth.partial` 路径的数据。

## TAO（OVMOT 主基准）— 完整可用

- 根：`/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal`
- `annotations/`：train.json（1230 类 / 500 视频 / 2647 tracks）、
  validation.json（988 视频 / 5485 tracks）、validation_with_freeform.json、
  tao_test_annotations.json；含 category `frequency`（r/f/c）、track、video。
- `frames/`：train 59G + val 116G + test 179G ≈ 354 GB。
- `BURST_annotations/{train,val,test}`：visibility 文件。
- OVMOT 官方协议：OVTrack 的 `create_tao_v1.py` 把 validation.json 按 synset
  映射到 LVIS v1 类别得到 `validation_ours_v1.json`；LVIS 类别表
  `lvis_classes_v1.txt` 本地存在于 masa（见下）。Base/Novel 由 LVIS frequency
  决定，TETA 指标分 Base/Novel/All 报告。
- 决策：主用 TAO val（本地已有帧），可软链接到 `data/`。

## LVIS v1 类别/频率（Base/Novel 定义来源）

- `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/lvis/annotations/lvis_v1_train.json`
  （另一项目数据，只读/软链接使用，不修改）。

## Refer-KITTI / Refer-KITTI-V2（RMOT 候选）

- `/data1/LWR/vranlee/MFT2025/REFER-MFT25/refer-kitti-v2`（79 MB）存在，
  属于 MFT2025 项目；含 expression/ 与 labels_with_ids/。若做 RMOT 需核对
  其是否官方 TempRMOT 格式（expression JSON 结构），并以软链接使用。
- 帧来源需另行核对（Refer-KITTI 原始 KITTI 帧）。

## OVT-B — 服务器无数据

- 官方分发为百度网盘/Google Drive；磁盘紧张（/data1 剩 171G、/data2 89G、
  /data3 122G），本阶段不下载，记录为候选。

## C-TAO — 禁用

- 本地 `/data3/testdata/vranlee/.MOTSynth.partial/C-TAO/` 含 ctao_base.json 等，
  但位于 `.MOTSynth.partial` 且来源无法证实，按项目禁令不使用。
- COVTrack 官方 C-TAO 分发路径可另行核对，但本阶段主实验用官方 TAO 标注即可。

## RMOT26（QTrack）/ STORM-Bench / VidOR

- 本地无；STORM-Bench 需要 VidOR 帧（未下载），磁盘紧张，本阶段不优先。
- 若 RMOT 主实验选 Refer-KITTI，优先利用本地 refer-kitti-v2。

## 磁盘预算

- 当前可用：/data1 171G、/data2 89G、/data3 122G。
- 计划：不新增大型下载；checkpoint/日志放 `outputs/l7/`；TAO 帧软链接。

