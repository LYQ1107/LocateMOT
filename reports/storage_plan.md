# Stage L0-B 存储规划

日期：2026-08-06

## 当前磁盘

| 挂载 | 剩余 | 可用率 |
|---|---|---:|
| /data1 | 67 GB | 99% |
| /data3 | 167 GB | 96% |
| /home | 1.5 TB | 57% |

/data3/testdata/vranlee 属主为 testuser，当前用户 lwr 无法写入 `/data3/testdata/vranlee/LocateMOT`；本阶段输出全部在项目目录（/data1）内。

## 当前 L0-B 产物

- `outputs/l0_b_token_debug/`：37 MB
- 9 张图、36 个查询、149 个 ObjectToken
- 平均每图调试产物 ≈ 4.1 MB（events + tokens JSON，含 2048/4608 维特征序列化）

## 单帧/单目标估算

- 每 token 特征：PBD 3×2048 + region 4608 + fused 256 ≈ 11,008 floats ≈ 44 KB（fp32 二进制）；JSON 文本约 200–300 KB。
- 每帧候选生成：events 若干条（~2–10 KB）+ 5–20 个 ObjectToken ≈ 1–6 MB JSON。
- 保守估计：2 MB/帧（JSON）。

## 全量 cache 容量估算（Stage L0-C 之后）

假设 YouTube-VOS train 3471 视频 + MOSE train ~3667 视频，每视频 2 帧、每帧平均 8 个候选 token：

- 帧数 ≈ 7140 × 2 = 14,280
- JSON 体积 ≈ 14,280 × 2 MB ≈ 28 GB（上限 ~60 GB）
- 二进制（npz/parquet）≈ 14,280 × 0.5 MB ≈ 7 GB

建议：

1. 输出根目录恢复为 `/data3/testdata/vranlee/LocateMOT`（167 GB 可用）；
2. cache 使用二进制格式（npz/parquet）而非 JSON，目标 ≤10 GB；
3. 若 /data3 权限无法解决，至少需在 /data1 腾出 40–60 GB 或改用 /home。

## 安全并发

- 本阶段单 GPU 推理峰值 8.5–17 GB（依赖图大小）；建议单进程 1 worker、batch=1。
- 全量缓存阶段再按“单个 worker 峰值内存”决定并发，遵守 AGENTS.md 的 MemAvailable 安全线。
