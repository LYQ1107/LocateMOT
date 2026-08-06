# Stage L0-C：存储规划与实际使用

## 选择路径

- 缓存根：`/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L0C/cache`
- owner：lwr；所在盘 /data3 剩余 167GB
- 原因：lwr 自有、容量充足，且不修改其他用户目录；不使用 sudo。

## 实际占用

| 项 | 大小 |
|---|---:|
| ObjectToken cache（3780 shards，float16 safetensors + meta json） | 429 MB |
| 预计算训练张量 train_subset.pt（1500 pairs） | 1.5 GB |
| 训练 checkpoint（B3/B4 等） | ~0.3 GB |
| 输出/报告 | <10 MB |

## 估算

- 平均每 frame+protocol shard ≈ 114 KB（含特征与元数据）。
- 全量 cache（若未来需要 6066+1071 视频）估算：20–40GB JSON/二进制混合；本项目只构建中等规模 cache。
- 训练输出：checkpoint 每个 ~30–75MB。

## 建议

- 当前 /data3 容量充足；无需改动 `/data3/testdata/vranlee/LocateMOT` 权限即可完成 L0-C。
