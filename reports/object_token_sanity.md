# Stage L0-B：ObjectToken 特征 sanity check 报告

日期：2026-08-06

## 声明

这是小样本 sanity check，不是正式 MOT benchmark；不据此声称 MOT 性能。GT 仅用于构造 reference prompt、判断 LocateAnything 候选是否匹配 GT、以及形成身份对；GT 没有用于筛选当前帧候选。

## 数据

- 数据集：YouTube-VOS 2019 train（只读路径 `/data3/testdata/vranlee/.MOTSynth.partial/YouTube-VOS-2019`）
- 视频数：8
- 每视频帧数：2
- GT 目标总数：36
- 正对：24（同视频同目标跨帧）
- 负对：120（同视频不同目标 + 跨视频目标）
- 时间间隔：0–5 帧（采样固定）
- 查询：按目标类别运行 "Locate all the instances ... category."
- 图像 token 预算：processor `in_token_limit=4096`（官方允许的运行时参数；720p 帧在 A100 sdpa/eager 下避免 OOM）

## Candidate recall（LocateAnything 候选匹配 GT）

| 指标 | 值 |
|---|---:|
| recall@IoU 0.3 | 75.0% |
| recall@IoU 0.5 | 75.0% |
| recall@IoU 0.7 | 66.7% |
| GT 对象数 | 36 |

## 特征余弦相似度与 AUC

| Feature | Positive mean | Positive median | Negative mean | Negative median | AUC | Valid pairs |
|---|---:|---:|---:|---:|---:|---:|
| PBD box-end | 0.925 | 0.993 | 0.882 | 0.935 | 0.768 | 24/120 |
| PBD coordinate mean | 0.792 | 0.881 | 0.650 | 0.695 | 0.777 | 24/120 |
| PBD full-block mean | 0.873 | 0.938 | 0.768 | 0.805 | 0.769 | 24/120 |
| MoonViT region | 0.907 | 0.888 | 0.812 | 0.801 | 0.633 | 24/120 |
| Simple fused (untrained projection) | 0.906 | 0.944 | 0.826 | 0.852 | 0.788 | 24/120 |

## 解读

1. 所有特征的正负分布高度重叠（negative mean 0.65–0.88），说明原始 hidden/region 特征整体相似度高，不能直接做全局匹配。
2. PBD coordinate-mean AUC 0.777 略优于 box-end 0.768 与 full-block 0.769，但差异在 24 个正对的规模下不显著。
3. MoonViT region feature AUC 0.633 最弱（本小样本下）。
4. 随机投影 fused 的 AUC 0.788 只是接口自检，未训练，不能当作融合有效性的证据。
5. Candidate recall@0.5=75% 意味着约 25% 的 GT 对象没有匹配候选；这是后续关联上限的重要瓶颈信号。

## 结论

- PBD hidden state 包含初步身份判别信息（AUC≈0.77），值得进入 Track Decoder 训练。
- region feature 在本小样本上较弱，但样本太少，不能据此否定；保留在 fused token 中作消融。
- 进入 Stage L0-C 前应先用更大、更规范的 pair 集合（20–50 对/类）复测。
