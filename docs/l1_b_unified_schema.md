# Stage L1-B Unified Identity Supervision Schema

目标：把不同数据集统一为 identity supervision 样本，而不是强制所有字段。

## 字段

- dataset_family: same-class dense | road multi-class | deformable/sparse
- dataset_name / split
- video_id（原始 id）
- frame_id / timestamp（优先真实时间；无则用帧序号并记录 fps 未知）
- instance_id（同一视频内稳定；跨视频不保证，除非数据集定义）
- category_id / category_name / category_known
- box（xyxy）
- mask_reference（VOS 数据集可用，路径）
- visibility / ignore（MOT17/20 有；其它无则空）
- annotation_exhaustive（MOT dense=True；YT-VOS/MOSE/TAO sparse=False）
- identity_available（bool；MOSE valid 隐藏 GT=False；BDD100K 已用
  masa box_track_20 tracking 标签，本地有图像时为 True）
- object_token_available（是否已缓存 LocateAnything ObjectToken）
- candidate_source / candidate_box / candidate_iou_to_gt（训练时由真实
  candidate→GT 匹配产生）
- feature_source: REAL_CANDIDATE_OBJECT_TOKEN | GT_ROI_FEATURE(diagnostic)
- temporal_gap（按真实时间换算；无 FPS 则帧差并标注）
- supervision_valid

## 监督三元组构造

- Positive: same instance_id, different frames（short 1–4 / medium 5–16 /
  long 17–64 / very-long >64，按各数据帧间隔换算）
- Hard negative: different instance_id, same category, 优先同视频、近空间、
  相似尺度、高 PBD/region 相似度
- Easy negative: different category

## 数据可用性判定

- TRAIN：identity_available=True 且 annotation 语义允许 identity supervision
- EVAL_ONLY：无训练身份标签或官方测试集（DanceTrack test、MOT17 test、
  MOSE valid、BDD100K 本地检测标签）
- NOT_USED：MOTSynth（规格禁止）
