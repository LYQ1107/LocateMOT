# Stage L0-C：训练报告

## 数据

- train pairs：6858（400 videos）
- calibration pairs：1383（80 videos）
- held-out pairs：2556（150 videos）
- 主特征：PBD coordinate-mean last + MoonViT region + geometry + generation score
- Reference 来源：reference candidate token（若匹配到 GT）或 GT crop region token

## B3 Pairwise MLP

- 结构：FeatureProjector(256) + PairwiseMLP（1030→512→256→logit）
- 训练：AdamW lr=2e-4、wd=1e-4、cosine、warmup 5%、bf16、batch 32
- 结果：calibration 最佳 0.547（step 2200）；训练 loss 0.76→0.69（早期）

## B4 Persistent Track Decoder

- 结构：d=256、4 层、8 heads、FFN 1024，reference_query
- 训练：同样配置，batch 32
- 结果：calibration 最佳 0.609（600 steps，全量数据）；loss 1.22→0.19

## 方向检查（1500 pairs 子集，同一数据/步数）

| 方向 | calibration 最佳 | best step |
|---|---:|---:|
| reference_query | 0.7206 | 200 |
| current_query | 0.7181 | 400 |

差异 0.25pp < 1pp → 保留 reference-query 作为主模型。

## 训练稳定性问题（已修复）

1. `nn.MultiheadAttention` 的 key_padding_mask 语义（True=忽略）用反；修复为 `~mask`。
2. 空样本（0 candidate）全 mask 导致 attention NaN；预计算时给空样本补 dummy 有效 key（loss 仍被 candidate_missing 屏蔽）。
3. contrastive 使用 F.normalize 对零向量产生 NaN；改为带 eps 的 cosine margin loss。

修复后训练 loss 稳定下降，无 NaN。
