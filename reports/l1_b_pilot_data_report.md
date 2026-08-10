# Stage L1-B Pilot Data Report

## Pilot cache

- 配置：`configs/l1_b/pilot_videos.json`（seed 20260806，video-level disjoint）
- 帧数：1,860（DanceTrack 360 / MOT17 240 / MOT20 160 / TAO-Amodal 240 /
  YT-VOS 360 / MOSE 500）
- Cache：/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1B/cache_dla
  （safetensors，每帧 1 文件，resume via .complete）
- 模型：LocateAnything-3B 冻结（commit 783f656d，checkpoint hash 已记录在
  meta），候选→GT 逐帧最大 IoU 匹配。

## Identity supervision

- 总观测：22,437；可用身份（≥2 观测）：704
  （DanceTrack 62 / MOT17 166 / MOT20 340 / TAO 23 / YT-VOS 44 / MOSE 69）
- 训练单元：positive（同身份不同帧）+ same-video hard negative +
  cross-video easy negative；dataset-balanced 每 epoch 上限 60 身份/数据集。
- 说明：TAO/YT-VOS 的 matched 身份较少（稀疏标注 + 查询类别覆盖有限），
  pilot 对这些数据的统计不稳定（q=23/44）。

## v2（加入 BDD100K）

- Cache 为超集：2,932 帧（BDD 240 新增；ytvos/mose 因配置再生成保留旧帧）。
- 观测 26,367；可用身份 1,064
  （BDD 275 / DanceTrack 62 / MOT17 166 / MOT20 340 / TAO 23 /
  YT-VOS 79 / MOSE 119）。
- BDD 本地数据：train 200 视频 / 39,418 帧 / 15,558 身份 / 11 类。

## 修复记录

- frame id str/int bug（dancetrack/mot17/mot20 GT 为空）→
  `tools/repair_l1b_cache_gt.py` 修复 760 帧 matched_candidates。
- gen_score 维度 (N,)→(N,1)。
