# Stage L5 — Evaluation Integrity Audit（L4 官方 TrackEval 之谜）

日期：2026-08-11。

## 1. 可疑现象

L4 最终报告声称 A2/A5/A5p 在官方 TrackEval ALL 模式与 U0 完全一致
（AssA/IDF1/IDSW 全部 4 位小数相同），但 `l4_restriction_audit` 的
per-video 均值又显示 ALL 指标有变化。二者矛盾。

## 2. 审计方法

对 3 个视频（DanceTrack `dancetrack0004`、BDD `0000f77c-6257be58`、
MOT17 `MOT17-02-SDP`），用 U0/A2/A5/A5p 四个 checkpoint 在 ALL 模式
重放 OnlineTracker，比较：

- 输出行 hash；
- 每帧候选→track map；
- GT-matched 候选分配差异；
- final affinity / base / delta 差异；
- 官方 TrackEval 数据目录是否被旧文件污染。

工具：`tools/l5_eval_integrity.py`；结果：
`outputs/l5/eval_integrity.json`。

## 3. 发现 1：L4 输出与 U0 确实不同（BDD/MOT17 差异巨大）

| Video | U0 vs A2 cand diff | U0 vs A5 cand diff | U0 vs A5p cand diff |
|---|---:|---:|---:|
| dancetrack0004 | 1/4,491 (0.02%) | 1/4,491 (0.02%) | 1/4,491 (0.02%) |
| BDD 0000f77c… | 175/220 (79.5%) | 167/220 (75.9%) | 175/220 (79.5%) |
| MOT17-02-SDP | 972/1,783 (54.5%) | 1,134/1,783 (63.6%) | 1,108/1,783 (62.1%) |

GT-matched 差异率：BDD 74–78%，MOT17 56–64%，DanceTrack 0.02%。
affinity 的 row-argmax 变化率在 BDD/MOT17 上为 1–60%。

## 4. 发现 2（Root Cause）：官方 TrackEval 数据目录被旧 U0 文件污染

`tools/run_l1d_trackeval.py::build_data` 只在 `not os.path.exists(dst)`
时复制 tracker 输出。L4 的 `eval_l4_ac.py` 复用了 L3 的 split 标签
（`dance_l3`/`bdd_l3`/`mot17_l3`/`mot20_l3`），而这些标签的数据目录
（`outputs/l1_d/trackeval_data_*_l3`）在 L3 已存在且包含 U0 文件。
因此 L4 新输出**从未进入 TrackEval**，官方表实际评估的是旧 U0 文件。

## 5. 修复

`tools/eval_l4_ac.py` 改为 per-tag split（如 `a5_dance_l3`），每次创建
全新 TrackEval 数据目录；旧 L3 目录保留不动。重新运行官方 AC 评估。

## 6. 重新评估结果（2026-08-11，per-tag 全新数据目录）

每个模型在 4 个域各跑一次官方 TrackEval（DanceTrack val 25 seq /
BDD100K train 200 seq / MOT17 train 3 seq / MOT20 train 2 seq，
AC 协议，与 L3/L4 报告一致）。

| 模型 | 域 | HOTA | AssA | IDF1 | IDSW |
|---|---|---:|---:|---:|---:|
| U0 (L3) | Dance | 0.6283 | 0.4169 | 0.5694 | 2588 |
| U0 (L3) | BDD | 0.3628 | 0.2881 | 0.2923 | 11042 |
| U0 (L3) | MOT17 | 0.6595 | 0.6050 | 0.5825 | 259 |
| U0 (L3) | MOT20 | 0.5012 | 0.2950 | 0.4012 | 2406 |
| A2 (L4) | Dance | 0.6288 | 0.4176 | 0.5638 | 2571 |
| A2 (L4) | BDD | 0.3503 | 0.2686 | 0.2787 | 11253 |
| A2 (L4) | MOT17 | 0.6502 | 0.5880 | 0.5778 | 281 |
| A2 (L4) | MOT20 | 0.4909 | 0.2831 | 0.3886 | 2408 |
| A5 (L4) | Dance | 0.6210 | 0.4075 | 0.5575 | 2647 |
| A5 (L4) | BDD | 0.3521 | 0.2714 | 0.2803 | 11127 |
| A5 (L4) | MOT17 | 0.6568 | 0.6008 | 0.5822 | 265 |
| A5 (L4) | MOT20 | 0.4955 | 0.2885 | 0.3963 | 2383 |
| A5p (L4) | Dance | 0.6220 | 0.4087 | 0.5597 | 2548 |
| A5p (L4) | BDD | 0.3531 | 0.2729 | 0.2812 | 11272 |
| A5p (L4) | MOT17 | 0.6655 | 0.6165 | 0.5919 | 266 |
| A5p (L4) | MOT20 | 0.4930 | 0.2857 | 0.3903 | 2388 |

要点：

- A2/A5/A5p 与 U0 的官方数字**确实不同**，但差异远小于逐帧审计中
  BDD/MOT17 的候选分配差异（54–80%）；这是因为 TrackEval 只统计与
  GT 匹配的部分，且 AC 协议下多数候选对结果的影响被生命周期/阈值吸收。
- 没有任何 L4 模型在全部域上同时保持或改善 U0：A5p 改善 MOT17
  （AssA +0.0115），但 BDD/Dance/MOT20 均下降。
- DanceTrack 的 L4 数字与 U0 接近（AssA 差异 ≤0.01），与逐帧审计
  “DanceTrack 仅 0.02% 分配差异”一致。

## 7. 结论

- L4 的「官方 TrackEval 与 U0 完全一致」是**评估管道 bug 造成的假象**；
- 需要重新评估 A2/A5/A5p 的真实官方指标；
- L4 小模型（0.488M、20 epoch、frame-level paired consistency）未在
  官方指标上带来跨域一致改善；
- `l4_restriction_audit`（进程内直接计算）不受此 bug 影响，其数字可信。
