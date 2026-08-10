# Stage L4 — Implementation Evidence

日期：2026-08-10。每个核心模块的官方参考、实际阅读文件、采用/不采用
理由，按证据要求记录。

## Module: Restriction Audit（P0 vs P1）

- Scientific purpose：证明 specification/candidate-set restriction
  真实改变 persistent identity，且不是 evaluation bug。
- Official references inspected：TrackEval formulas（
  `references/TrackEval-official`，commit 12c8791b）、MOTIP ID prediction
  （`references/identity_decoding/MOTIP`，ffc0e905）、Path Consistency
  （`references/association_2025_2026/PathConsistency`，f4b7d26d）。
- Repository commits：见上。
- Files inspected：`tools/run_l2_oracle.py`（windowed_metrics）、
  `tools/build_l1d_dataset.py`（base simulator）、`tools/eval_l3.py`、
  `locatemot/tracking/online_tracker.py`。
- Observed implementation：U0 在同一 AC shell 上对全候选/受限候选分别
  推理；最优 ID 映射（Hungarian on co-occurrence）后比较 common
  objects 的 co-identity agreement。
- Parts adopted：windowed AssA/IDF1/IDSW；per-video fresh tracker；
  threshold 0.25 / delta 0.3。
- Parts intentionally not adopted：不比较 raw track ID；不用 GT
  membership 过滤主推理结果（全部标 PRIVILEGED_SPEC_ORACLE）。
- Reason for final design：需要 permutation-invariant 且对
  merge/split/switch 敏感的诊断指标；TrackEval 仍是主结果。

## Module: Paired Spec Views

- Scientific purpose：构造 `T(R_s(X))` 与 `R_s(T(X))` 的可训练配对。
- Official references inspected：TDLP clip 构造
  （`references/association_2025_2026/TDLP`，50344b92）、Path
  Consistency 路径构造（f4b7d26d）、CAMELTrack 轨迹状态采样
  （`references/l1_d/CAMELTrack`，46a74bb）。
- Observed implementation：同一 L1DK base 对 full/restricted 候选流
  各自推进 tracker；按 frame 对齐候选，按 birth GT 对齐轨迹。
- Parts adopted：base simulator（L1DK Kalman）、EGRA 特征、
  Hungarian+threshold、birth/lifecycle。
- Parts intentionally not adopted：不用未来帧；不把 restricted view
  的 GT 身份当推理输入。

## Module: L4SpecEqAssociator（shared identity core + spec conditioning）

- Scientific purpose：一个 checkpoint 统一多域 + 多 spec；spec 只决定
  WHAT to track，shared core 决定 HOW to track。
- Official references inspected：CAMELTrack GAFFE set-level interaction
  （46a74bb）；V2-SAM visual prompt matcher + contrastive alignment
  （`references/l4/v2-sam`，31c3babf）；NOVA class split / hybrid prompt
  （`references/l4/nova`，4358a627）。
- Observed implementation：U0（L1DAssociator）核心不变；type-level
  spec embedding（ALL/category/instance）只注入 set-encoder token，
  不改变 pair head 结构；U0 权重完整初始化。
- Parts adopted：EGRA set transformer + bounded residual + reliability
  gate；spec 作为共享、非 dataset-specific、有界条件。
- Parts intentionally not adopted：不用 category one-hot 进 pair MLP；
  不做 dataset-specific MoE/router；不引入大 VLM。

## Module: Assignment / State Consistency Loss

- Scientific purpose：让 common objects 在两视图的 identity
  decision/state 一致（permutation-invariant）。
- Official references inspected：Path Consistency loss
  （f4b7d26d）；V2-SAM `get_contr_loss` 双向对比（31c3babf）；
  UniTrack 轨迹一致正则（afdd9869）。
- Observed implementation：row/col softmax 在 common candidates /
  common tracks 上的对称 KL；common track token 的 cosine 一致性。
- Parts adopted：permutation-invariant 的对齐方式（common set 内
  重归一化）；对称 KL；state cosine 作为轻正则（lambda=0.1）。
- Parts intentionally not adopted：不做 MSE 两个不同大小矩阵；
  不要求 raw ID 相等；不用 teacher-student 的 stop-grad（symmetric）。

## Module: TAO Cache Recovery

- Scientific purpose：恢复 open-world 长尾域证据，不重跑
  LocateAnything。
- Observed implementation：manifest key 与 cache 路径差一层
  `train/<SOURCE>/`；通过 `cache_key` 覆盖修复。
- Parts adopted：`tools/fix_tao_manifest.py` + `build_candidates`
  cache_key 优先。
- Parts intentionally not adopted：不改共享缓存；不复制 safetensors；
  不使用 `.broken` 帧。
