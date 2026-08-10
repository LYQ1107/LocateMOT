# Stage L1-B Same-Category Analysis（Raw Baselines）

## 设置

单类别数据集（DanceTrack/MOT17/MOT20/MOSE）天然 same-category；TAO/YT-VOS
按类别过滤 gallery 后单独计算 same-category 与 cross-category。

## Same-Category vs Cross-Category（TAO / YT-VOS，R0 PBD-box-end）

| dataset | same-cat R@1 | same-cat mAP | cross-cat R@1 | cross-cat mAP |
|---|---:|---:|---:|---:|
| tao_amodal | 0.870 | 0.841 | - | - |
| ytvos | 0.750 | 0.895 | - | - |

说明：pilot 中跨类别负样本数量很少（TAO 23 queries、YT-VOS 44 queries），
cross-category 数字统计不稳定，暂以 same-category 为核心指标；全量数据阶段
再补 cross-category 表。

## 关键观察

1. same-category 是更难的协议（去掉“不同类别就是不同实例”的捷径后，
   region 与 fused 的 R@1 大幅下降），证明 raw region 主要编码类别语义。
2. PBD 特征在 same-category 下仍保持强判别（R0/R1），说明 LocateAnything
   的 PBD 已携带部分 instance 信息，Identity Adapter 的核心价值在于把
   这些弱信号在 deformable/sparse（YT-VOS/MOSE）上稳定化。
3. 不存在明显的 category shortcut 证据（raw 特征 same-cat 与混合指标接近）。
