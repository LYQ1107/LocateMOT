# Stage L3 — 数据集审计

日期：2026-08-10。只审计本地实际存在的资源，不假设。

## 1. DanceTrack

- 本地：`/data1/LWR/vranlee/DATASETS/JDE/dancetrack`（train/val GT）；
- manifest：`outputs/l1_c/fixed_candidate_manifest/dancetrack_{calibration,val,train}.jsonl`；
- cache：`/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1A/cache_dla`
  （DanceTrack 专用）；
- 规模：val raw pkl 25 视频 / 25,508 帧（val split 40 序列，
  其中 25 有 cache）；calibration 8 视频 / 8,024 帧；
- regime：密集同类、外观歧义高、非线性运动、交叉；
- 标注：box + GT ID，person only，dense；
- 角色：calibration=训练（oracle/U0/U1），val=主评估。

## 2. MOT17

- manifest：`fixed_candidate_manifest/mot17_train.jsonl`；
- cache：`LocateMOT_L1B/cache_dla/mot17`；
- 规模：3 视频（MOT17-02/04/09-SDP），240 帧（采样）；
- regime：标准 pedestrian，中等密度；
- 标注：dense box + ID；
- 角色：训练 + 跨域评估。

## 3. MOT20

- manifest：`fixed_candidate_manifest/mot20_train.jsonl`；
- cache：`LocateMOT_L1B/cache_dla/mot20`；
- 规模：2 视频，160 帧；
- regime：极端 crowd / 遮挡；
- 标注：dense box + ID；
- 角色：训练 + 跨域评估。

## 4. BDD100K（多类）

- manifest：`fixed_candidate_manifest/bdd100k_train.jsonl`；
- cache：`LocateMOT_L1B/cache_dla/bdd100k`；
- 规模：200 视频 / 8,001 帧（5fps 采样）；
- **GT 已含 11 类**：bicycle/bus/car/motorcycle/other person/
  other vehicle/pedestrian/rider/trailer/train/truck；
- `matched` 字段含全类候选匹配（5,191/8,001 帧 ≥2 个匹配）；
- 现有 `outputs/l1_d/raw/bdd100k_train.pkl` 的 `cand_gt` 即全类；
- regime：多类驾驶、ego-motion、5fps 大时间间隔、尺寸分布广；
- 角色：多类训练 + 多类评估（按类过滤 + macro，或官方 BDD eval）。

## 5. TAO / C-TAO

- manifest：`fixed_candidate_manifest/tao_amodal_train.jsonl`；
- 规模：105 视频 / 4,200 帧；有候选帧 2,256；
- cache：`LocateMOT_L1B/cache_dla/tao_amodal/`，但为**旧布局**
  （`train/{BDD,AVA,YFCC100M,HACS,LaSOT}/...`），与 manifest 的
  `cache_key` 不匹配；未发现 `.complete` 标记；
- `/data3/testdata/vranlee/.MOTSynth.partial/C-TAO/` 有 C-TAO
  base/novel 类别文件（清单级）；
- 结论：TAO 为 sparse/federated annotation，评测须用官方 TETA/TAO
  协议；**当前 cache 不可直接复用，Stage L3 先文档记录，延迟到
  A 的四域 pilot 之后**。

## 6. 其他（本地状态）

- YTVIS/MOSE：cache_dla 下有缓存目录，但无 L3 manifest，未纳入；
- DAVIS/BURST：仅 GLEE_PMOT 项目的评估 CSV（其他项目只读参考），
  无本项目缓存；
- STORM-Bench / RMOT26：官方基准已 clone（`references/l3/`），
  数据未下载（体积大），referring 仅作诊断级；
- MOTSynth：禁止使用。

## 7. 结论

L3 主实验最小集合：

- DanceTrack calibration（训练）+ val（评估）；
- MOT17 / MOT20（训练 + 评估）；
- BDD multi-class（训练 + 多类评估）；
- TAO：cache 修复后纳入 open-world 证据。
