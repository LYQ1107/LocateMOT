# Stage L1-B Storage Plan

原则：只缓存 pilot 需要的样本；高维特征用 safetensors / npz / mmap，禁止 JSON。

## 每样本估计

- LocateAnything ObjectToken：PBD box-end + coordinate + MoonViT region 拼接
  后约 1–4KB/样本（fp16）；原始 region feature 缓存约 4KB/样本。
- 预计 pilot 样本：identity units 50k–150k → cache 约 0.2–0.6GB（特征级）。
- GT ROI feature（diagnostic）：每样本约 0.5–1KB → pilot 10k 样本约 10MB。

## 分数据集 pilot 预算（先缓存最小子集）

- DanceTrack train：pilot 12 个视频 / ~5k units
- MOT17 train：pilot 4 个序列 / ~3k units
- MOT20 train：pilot 2 个序列 / ~2k units
- YT-VOS train：pilot 200 视频 / ~2k units
- MOSE train：pilot 200 视频 / ~2k units
- C-TAO：pilot 100 视频 / ~2k units（若许可证允许；COVTrack 派生数据待核）

## 磁盘

/data1 剩余约 40GB；pilot 缓存 <2GB，安全。全量 cache 在 pilot 通过后再评估
（预计 <30GB，按 dataset-balanced 采样而非全帧）。
