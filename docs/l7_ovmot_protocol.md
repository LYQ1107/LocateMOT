# Stage L7 OVMOT Protocol（TAO 官方协议，已核对官方代码）

## 数据集与标注

- GT：官方 TAO val，LVIS v1 类别版本
  `tao_val_lvis_v1_classes.json`（MASA/HuggingFace dereksiyuanli/masa 发布，
  与官方 TETA repo 提供文件一致）。988 视频 / 36,375 图 / 1,203 类
  （c=461、f=405、r=337）。
- 候选：官方 Detic SwinB public detections（MASA 发布、`teta_50_internms`，
  每帧 ≤50 box，label 为 LVIS v1 id）。
- 帧：官方 TAO 帧（本地 354GB），1 fps 标注。

## Base / Novel / All 定义（官方 run_ovmot.py）

- Base = `frequency != "r"`（LVIS common + frequent）
- Novel = `frequency == "r"`（rare）
- All = 全部类别（COMBINED）

## 指标（官方 TETA，TETA repo b498aa8）

- 逐类 TETA50（alpha=0.5）：LocA、AssocA、ClsA（及 Re/Pr 分量）。
- Base / Novel 为对应类别 TETA50 的均值；All 为全体。
- 我们报告 Base / Novel / All 的 TETA、LocA、AssocA、ClsA。

## 预测格式

COCO-VID 风格 JSON list：`{image_id, video_id, track_id, category_id,
bbox [x,y,w,h], score}`。tracker 只负责 track_id（association），
category_id 首版为 frozen Detic label（perception 分离，便于归因）。

## 评估命令

`references/l7/TETA/scripts/run_ovmot.py --GT_FOLDER <v1.json>
 --TRACKERS_FOLDER <root> --TRACKERS_TO_EVAL UIDM --TRACKER_SUB_FOLDER data
 --SPLIT_TO_EVAL val --USE_PARALLEL False`

## 可比性

- 同候选/同 GT 下与 OVTrack（CVPR23，AssocA base 36.9 / novel 33.6）、
  COVTrack（ICCV25）对比为 apples-to-apples（REFERENCE_ONLY 若为论文数字）。
- OVTR 是端到端（含检测），不可直接比，标 REFERENCE_ONLY。
- 主因果比较：同一 public dets 下，冻结 UIDM core + 新语义前端 vs
  task-specific baseline。

