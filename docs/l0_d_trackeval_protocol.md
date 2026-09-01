# Stage L0-D Two-Frame TrackEval Protocol

## 目的与命名

把两帧 association 修复结果转换成标准 tracking metric 语言（HOTA/DetA/AssA/LocA/MOTA/MOTP/IDF1/IDP/IDR/IDSW/FP/FN/Frag/MT/PT/ML）。这是 **Two-frame held-out association TrackEval diagnostic**，不是 MOT17/DanceTrack/正式 benchmark 结果。

## 序列构造

每条 held-out pair 是独立 2-frame sequence：

### Frame 0（t0，reference 初始化）

- GT：reference targets 的 GT box（xyxy 转 MOT xywh），track ID = 该 pair 内稳定序号（1..M）。只有 reference targets 进入 GT。
- tracker：与 GT 完全相同的 box + 相同 track ID（合法初始化，不参与关联学习）。

### Frame 1（t1，current）

- GT：当前帧中仍然可见的 reference targets：
  - `assignment_targets` 对应的 ref（候选存在）；
  - `candidate_missing_targets` 对应的 ref（GT 存在但 LocateAnything 无候选）；
  - `true_no_match` 的 ref 不在 GT 中。
- tracker 输出仅由 association 模型预测构造：
  - ref 被分配 candidate j → 输出 candidate box，track ID = ref ID；
  - ref 被判 NO_MATCH → 不输出；
  - ref 属于 candidate_missing → 不输出（反映为 FN）；
  - 未被任何 ref 使用的候选 → 输出为新 track ID（反映为 FP 的来源，允许 false candidate 形成 FP）。

## 硬性保证

- 每个 candidate box 至多被一个 ref 使用（Hungarian 一对一）。
- current GT 不参与 prediction 构造（只用于打分）。
- 一个 ref 至多输出一个框；NO_MATCH 正确时不凭空输出框。
- 不做 ignore regions（两帧数据无 crowd/ignore）。
- 全部 box 在 MOT 格式下为 `frame, id, x, y, w, h, 1, 1`（mark=1，class=1）。

## 评估与聚合

- 使用 `references/TrackEval-official`（commit `12c8791b`）官方 Evaluator + HOTA/CLEAR/Identity metrics。
- 单类（object）；所有 pair 序列一起评估。
- 报告 `COMBINED_SEQ`：HOTA 的 TP/FN/FP 求和、AssA/LocA 按 HOTA_TP 加权；CLEAR 与 Identity 用官方 `combine_sequences`。
- 不按 pair 先算再平均（除非官方聚合本身如此）。

## 分层

在整体 COMBINED 之外，按以下子集分别运行官方评估：

- target count：1 / 2–4 / 5–8
- dataset：YouTube-VOS / MOSEv2
- protocol：category_known（category_guided） / generic
- hard competition：easy / hard（`configs/l0_d_hard_subset.json` 冻结定义）

分层结果同样命名为 two-frame diagnostic。

## 实现位置

- `locatemot/evaluation/two_frame_trackeval.py`：数据适配（内存读入，复用官方 metric 与聚合）
- `tools/l0d_trackeval.py`：构造序列 + 调用官方 Evaluator + 输出 CSV/JSON
