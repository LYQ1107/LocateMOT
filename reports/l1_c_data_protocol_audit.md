# Stage L1-C Data / Protocol Audit

审计时间：2026-08-09。目的：复用 L1-A/L1-B 已有 cache，禁止重复生成昂贵
cache；确定哪些数据集可进入 pilot 训练、association-controlled 评估、
official metric 评估。

## 1. 已有 Cache 盘点

### L1-A DanceTrack（LocateAnything-3B 冻结，free-form PBD detection）

- 根：`/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1A/cache_dla`
- 总帧数：67,578（train 34,046 / calibration 8,024 / val 25,508）
- 视频数：65（train 32 / calibration 8 / val 25），split 互斥
- 每帧条目：`dancetrack/{vid}/{frame}/person.safetensors + .meta.json + .complete`
- 字段：pbd_coord_mean_last / pbd_box_end_last / region / geometry / gen_score /
  boxes / normalized_boxes；meta 含 GT boxes、matched_candidates、split。
- 协议：`person`（DanceTrack 单人检测 prompt）

### L1-B 多数据集 Cache（LocateAnything-3B 冻结，free-form PBD detection）

根：`/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1B/cache_dla`

| dataset | frames | videos | unique IDs（采样帧内） | cands/frame | gt/frame | match rate |
|---|---|---|---|---|---|---|
| bdd100k | 8,001 | 200 | 14,370 | 7.0 | 9.66 | 0.628 |
| tao_amodal | 4,200 | 105 | 549 | 1.79 | 3.04 | 0.378 |
| dancetrack | 360 | 6 | 25 | 10.05 | 10.14 | 1.0 |
| mot17 | 240 | 3 | 130 | 25.97 | 28.78 | 0.99 |
| mot20 | 160 | 2 | 271 | 46.32 | 50.89 | 0.975 |
| ytvos | 690 | 23 | 5 | 2.92 | 3.0 | 0.95 |
| mose | 1,000 | 20 | 12 | 5.08 | 5.07 | 0.809 |

- 字段：pbd_coord_mean_last/penultimate、pbd_box_end_last/penultimate、
  pbd_full_block_mean_last、region、geometry、fused、gen_score、boxes、
  normalized_boxes；meta 含 GT、matched_candidates、query。
- 说明：
  - TAO 目录含 `train/.../*.broken` 权限受限的残缺帧（不影响 complete 计数，
    已用 `repair_l1b_cache_gt.py` 处理历史问题；本轮统计基于 .complete）。
  - TAO/BDD 是稀疏/部分标注：match rate 0.378/0.628 表示 LocateAnything
    检出的候选只有一部分能匹配到 GT，不能用 match rate 当检测质量。

## 2. 数据格式与协议

- Cache key：`{dataset}/{video_id}/{frame:05d}/{protocol}`，
  `.complete` 标记原子完成；读接口 `locatemot.data.token_cache`。
- 候选 boxes 为 LocateAnything PBD 输出（像素 xyxy），PBD 特征为该 box
  block 的 hidden state（box_end / coord_mean / full_block_mean 及其
  penultimate 变体），region 为 MoonViT 对应网格特征。
- 冻结性：模型 commit 783f656d，checkpoint hash 存于 meta，
  可用于 fixed manifest 的完整性校验。

## 3. 各数据集角色与资格

| dataset | train 资格 | association-controlled eval | official metric | 角色/原因 |
|---|---|---|---|---|
| DanceTrack | 是（L1A train 32v / L1B 6v） | 是（L1A val 25v 全帧） | TrackEval HOTA | same-class dense；主 pilot 域 |
| MOT17 | 是（train 3 seqs） | 是（同 train 帧做 controlled eval 仅作诊断，不冒充 test） | TrackEval（需 test 数据，本地无） | standard pedestrian |
| MOT20 | 是（train 2 seqs） | 是（诊断） | TrackEval（需 test） | extreme crowd stress |
| BDD100K | 是（train 200v，8,001 帧采样） | 是（同批帧 controlled，做分类正确性分析） | 官方 BDD tracking metric（本地无官方 eval 包，先报告 per-frame 指标） | multi-class road；semantic cue |
| TAO / C-TAO | 是（train 105v，4,200 帧采样；sparse supervision 需按真实标注语义处理） | 是（controlled 诊断；不做 dense GT 转换） | TETA（本地有 OVTR 的 teta 代码，可评估检测/关联分类子项） | long-tail/open-world |
| YT-VOS / MOSE | 否（本阶段不作为主训练） | 关系诊断（deformable stress，不进入主表） | 不适用 | 未来扩展 |

- 已禁用：MOTSynth（规格明确禁止）。
- Split 卫生：L1A 三 split 互斥；L1B 全部取自各数据集 train split，
  不会与 test/val 混合。pilot 评估统一使用 L1A DanceTrack val 25 视频；
  其余数据集 controlled eval 使用与其训练互斥的保留视频（若 cache 内视频
  有限，则明确标注为 in-sample diagnostic）。

## 4. Fixed Candidate Manifest 计划

从上述 cache 直接构建 `outputs/l1_c/fixed_candidate_manifest/`：

- 每帧记录：dataset/video/frame、candidate boxes（xyxy）、score、
  PBD features（box_end_last / coord_mean_last 为准，dim 记录）、
  region feature、geometry、query/semantic、GT ids/boxes、match 映射。
- 冻结 hash：按 dataset/video 对 boxes + scores 计算 sha256，
  存入 manifest 元数据；后续 association 方法必须复用同一 box set。
- Pilot 覆盖：DanceTrack（val 25v）+ BDD（train 200v 子集）+ TAO
  （train 105v 子集）+ MOT17/20（诊断）。

## 5. 与 L1-A 协议问题的关系

L1-A 已确认：T4–T6 因输出 boxes 数量不同导致 DetA/MOTA 变化，
无法归因 association。L1-C 的 association-controlled protocol 将固定
box/score/count，只允许 track ID 变化；DetA/LocA/FP/FN 必须一致，
否则该 controlled 实验非法（详见后续
`reports/l1_c_association_controlled_protocol.md`）。
