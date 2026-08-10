# Stage L5 — Clip-Level GT-Anchored 数据集

日期：2026-08-11。

## 1. 设计原则

L4 的 frame-level paired 数据有两个问题：

1. 只保留最后一帧的 track 状态（last box + ref PBD），模型无法学习
   temporal identity process；
2. 监督用 birth-GT 对齐，会把已经被 tracker 污染的 identity 当作
   正确答案。

L5 数据改为：

- **GT-anchored track（`gt` source）**：每个 track 对应一个真实 GT
  identity，history = 该 GT identity 在视图内过去最多 16 帧的候选观测
  （由 manifest 的 `matched[gid].candidate` 决定）。目标行/列标签完全由
  GT identity 定义。
- **U0 rollout track（`u0` source）**：用冻结 L1DK base tracker 按视图
  重放，history 包含真实关联错误（输入证据允许错误）；track 的监督标签
  取 history 中占多数的 GT identity（GT-anchored，不用 tracker 整数 ID）。

## 2. 每个视频保存什么

```
video record:
  image_size
  cands: [{frame, box[N,4], pbd[N,2048] float16, gen[N], gt[N],
           gt_box, matched}]
  views: {spec: {"gt": [frame_sample...], "u0": [frame_sample...]}}

frame_sample:
  frame           位置下标
  frame_id        绝对帧号
  keep            视图内候选在全集中的下标
  base            L1DK base affinity [T,N]
  pair_feats      [T,N,19]
  track_feats     [T,16]
  cand_feats      [N,12]
  row_label / col_label   GT 监督的一对一目标（-1 = 无）
  base_correct    该帧 base argmax 是否正确
  track_gt        每 track 的 birth GT（诊断用）
  track_dom_gt    每 track 的 history 多数 GT（监督锚点）
  track_hist      [(abs_frame, pos, cand_idx, box, pbd, gt, gen, log_ncand)]
  track_tid       u0 的 tracker 整数 id（仅诊断）
```

## 3. 视图（specification）

- BDD100K：ALL + 每个视频实际出现的类别（cat:car / truck / bus /
  pedestrian / rider / …）；
- DanceTrack / MOT17 / MOT20：ALL + 每视频最长 2 条 GT 实例
  （inst:<gid>）。

## 4. 文件

| 文件 | 内容 |
|---|---|
| `outputs/l5/clips/small_bdd_train.pkl` | 8 个多类别 BDD 视频 |
| `outputs/l5/clips/small_bdd_val.pkl` | 2 个多类别 BDD 视频 |
| `outputs/l5/clips/small_dance_train.pkl` | 3 个 DanceTrack calibration 视频 |
| `outputs/l5/clips/small_dance_val.pkl` | 1 个 DanceTrack calibration 视频 |
| `outputs/l5/clips/mot17_train.pkl` / `mot20_train.pkl` | MOT17/20 全部 |

## 5. Baseline drift（u0 source，val 小集）

`tools/l5_drift_eval.py --scorer base`：

- 整体 drift rate：65.0%（common candidate 在 ALL 与 restricted 视图
  中被分配给不同 track GT 的比例）；
- BDD：46.5%（370 common / 172 drift）；
- DanceTrack：68.1%（2228 common / 1517 drift）；
- 事件分类：Type1 787、Type2（P0 wrong/P1 correct）1554、
  Type3 2、Type4 122、Type5 133。

这确认 L4 的问题信号在小集数据上依然成立，且存在大量「P1 更正确」的
hard case。

## 6. Hard-case 采样

训练 Dataset 按 group（video, frame）加权，权重与 `base_correct` 错误率
正相关（`1 + 3 * wrong_fraction`），并以 60% 概率按该分布采样；
group 化同时保证 ALL 与 restricted 视图在同一 batch 内出现，以计算
cross-spec relation consistency loss。

## 7. 限制

- PBD 只保存 `pbd_box_end_last`（2048 维 float16），未保存 region
  （4608 维）以控制体积；如后续证据表明 region 必要再扩展。
- DanceTrack calibration 只有 8 个视频；full 训练时使用
  `dancetrack_calibration` 全部视频。
- MOT20-02 的 inst:176/178 视图 u0 样本为 0（该实例候选在 tracker
  生命周期中未产生有效样本），gt 样本正常；full 训练时以 gt 为主。
