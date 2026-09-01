# Stage L5 — 2025/2026 官方代码与文献深度审计

日期：2026-08-11（Asia/Shanghai）。
原则：只记录实际 clone + 阅读的官方代码，或明确标注 paper-only /
非官方复现；不根据摘要或博客转述实现细节；每个仓库记录
URL / commit / license / inspected files / adopted component。

## 1. L5 要回答的问题

1. 是否存在 2025/2026 官方方法证明「persistent temporal identity state +
   set-level association + GT trajectory supervision」能提升异构 MOT 的
   identity consistency；
2. 是否存在「同一视频、不同 specification/candidate subset 下身份稳定」
   的直接等价工作；
3. 哪些官方实现提供了 Route A / B / C 的可迁移设计。

## 2. 新检索范围（2026-08-11）

关键词（GitHub / arXiv / CVF / IEEE）：

- temporal identity learning MOT、trajectory identity transformer、
  persistent identity state、track memory transformer 2026、
  sequence identity decoder、clip-level association transformer、
  trajectory-level identity supervision、dynamic ID prediction；
- multi-path tracking consistency、subset perturbation consistency、
  candidate dropout / distractor invariant MOT、specification invariant、
  query invariant identity、prompt invariant identity；
- unified MOT 2026、multi-dataset single checkpoint、state-space MOT、
  differentiable assignment / Sinkhorn / soft Hungarian、trajectory
  tokenization、autoregressive identity、generative MOT、open-vocabulary
  unified tracking、learned lifecycle / birth / death。

## 3. 新核实方法（paper-only / 无官方代码，不作为实现依据）

| 方法 | 出处 | 官方仓库 | 判定 |
|---|---|---|---|
| MambaTrack（MOT baseline with SSM） | 2025 | 未开源（第三方 `JackWoo0831/Mamba_Trackers`，非官方） | 仅概念参考，不采用 |
| Gated Temporal Fusion Transformers | WACV 2026 | 未找到官方 repo | encoder-level tracklet memory 概念；无代码，不采用 |
| AssociaTR（Long-Horizon Set Prediction） | 2026 | 未找到官方 repo | 直接输出完整轨迹；与本项目在线因果约束冲突，不采用 |
| NOOUGAT（unified online/offline graph tracking） | arXiv 2509.02111, IJCV | 未找到官方 repo | 强设计参考：learned ALT 层 + subclip + 无 heuristic 关联；paper-only |
| Dual-Path Temporal Decoder | NeurIPS 2025 | 未找到官方 repo | appearance/identity 双路径解耦；无代码，不采用 |
| DecoderTracker / FixDT | Pattern Recognition 2026 | 官方入口指向 `liaopan-lp/MO-YOLO`（AGPL-3.0） | 已 clone 阅读；decoder-only + fixed query memory；AGPL 不允许复制代码，仅记录设计 |
| CATB Identity Vault（VOT） | 2025/26 | 未找到可信官方 repo | 不采用 |
| APML（Sinkhorn matching loss） | 2025 | `apm-loss/apml` | Sinkhorn 可微匹配；与本项目 Hungarian 推理约束不直接兼容，仅理论参考 |

## 4. L5 新 clone 并阅读的官方仓库

### 4.1 SOTFormer（CVPR 2026，official）

- 官方 URL：https://github.com/zhongpingDong12/SOTFormer
- Commit：`bb28e62d596c7bf107269ae1e9749fc30c48052f`
- License：MIT
- 已读文件：`Code/SOFTFomer_model.py`、`README.md`
- 实际机制：
  - GT-Primed Initialization：训练/推理前 K 帧用 GT box 与 query 的
    IoU 交换 slot-0，消除 cold-start drift；
  - constant-memory：temporal attention 只保留 detached refinement
    state，长序列 O(1) 内存；
  - 多任务头：detection + identity consistency + 短程轨迹预测。
- 与 Route A 关系：single-object tracking（SOT），不是 MOT；
  「GT-primed persistent state」与我们的 GT-anchored track state 概念
  相近，但本项目是 TBD 式 MOT（不重新训练检测器），不复制其 DETR 主干。
- 采用：无代码复用；概念上支持「state 由 GT 轨迹锚定初始化/监督」。

### 4.2 MO-YOLO（含 DecoderTracker）

- 官方 URL：https://github.com/liaopan-lp/MO-YOLO
- Commit：`029e23a776ad916d87f27335f804bdb0064d1466`
- License：AGPL-3.0（不可复制进本项目）
- 已读文件：`README.md`、`MOTR/` 目录结构
- 实际机制：decoder-only 端到端 MOT，固定 query memory，弱监督
  缩短训练时间。
- 采用：仅记录设计思想；不复制任何代码。

## 5. 既有参考仓库的 Route A 复核（已 clone，重新精读相关文件）

| 方法 | 官方 repo / commit | License | 关键机制 | Route A 采用/不采用 |
|---|---|---|---|---|
| MOTIP | MCG-NJU/MOTIP `ffc0e905` | Apache-2.0 | 历史轨迹 feature + ID embedding；candidate 为 query、轨迹为 key/value 的 cross-attention；相对时间位置编码；ID 分类 K+1（newborn）；推理扩展矩阵 Hungarian；无显式 NO_MATCH 类，阈值等价 | 采用：track-candidate cross-attention 方向、相对时间 PE、GT-anchored sequence-local ID 目标（Route B 主依据；Route A 的结构近亲） |
| MOTIP-2 | GISer-WB/MOTIP-2 `012856c1` | Apache-2.0 | 独立复现仓库；IDDecoder 第一层无 self-attn；可学习相对时间偏置；ensemble | 仅确认 MOTIP 设计可复现，不采用其实现 |
| TrackFormer | timmeinhardt/trackformer `e468bf15` | Apache-2.0 | track query 跨帧复用；输出顺序分离 tracks/new；matcher 对 track 强制匹配自身 GT | 采用概念：persistent query = persistent track state；不采用其 DETR 检测联合训练 |
| MOTR | megvii-research/MOTR `8690da33` | MIT | QIM 更新 track query；消失槽位 matched_gt_idx=-1；memory bank 可选 | 采用概念：NO_MATCH 的槽位语义；不采用其实现 |
| MeMOTR | MCG-NJU/MeMOTR `eb7a177b` | MIT | short/long memory 分层；long memory EMA；memory attention | 采用概念：持久 state 更新；Stage L5 先不做完整 long bank |
| CO-MOT | ICLR 2025 `1e0618a` | 见仓库 | coopetition label assignment + shadow set | 不采用（与统一身份目标无直接关系） |
| SambaMOTR | `f1c139a` | 见仓库 | Mamba temporal association | 概念参考；无官方完整训练代码可复用 |
| TDLP | `50344b9` | MIT | 下一帧 link prediction 轨迹 | 概念参考：轨迹级 link prediction 与 Route C 一致 |
| CAMELTrack | references/l1_d `46a74bb` | 见仓库 | GAFFE set-level 交互 + ranking 训练 | 采用：set-level competition 的 training 范式（已在 L1D 中 clean 实现） |
| Path Consistency | amazon-science `f4b7d26d` | Apache-2.0 | 多观测路径关联一致性 | Route C 数学近亲；但路径=时间子采样，非 spec 子集；采用其「一致性监督」思想，不复制 |
| HATReID-MOT | `3eb440c` | 见仓库 | ReID + MOT | 不采用（L1-B 已证明单帧 ReID 路线失败） |
| NOVA | IROS 2026 `4358a627` | Apache-2.0 | 3D open-vocab autoregressive association | 不采用核心（3D + LM），借鉴 class-split 严谨性 |
| V²-SAM | CVPR 2026 `31c3babf` | README MIT（无 LICENSE 文件） | visual prompt 对比对齐 | 不复制；表示对齐思路与 Route A relation consistency 相关 |
| GLEE / OVTR / OVTrack / QTrack / AnyTrack / SAM2MOT / SAM3 / Grounded-SAM2 | l3 各 commit | 见 l3 审计 | prompt 类统一 | 均已审计：无跨 spec 身份等价目标 |
| TrackEval-official | references/TrackEval-official | MIT | 官方评估 | 本项目唯一评估入口（AC 协议） |

## 6. 为什么 Route A 不是现有方法的重述

- MOTIP 在同一 spec 内做 ID 预测，没有「同一视频不同 candidate subset 的
  身份稳定性」目标；其 ID 词汇表是 dataset-global 的（Route B 会参考其
  sequence-local 变体，但 Route A 不要求固定词汇表）。
- SOTFormer 只跟踪单目标；我们处理多目标 set-level competition。
- NOOUGAT 是 offline/online 图关联，不处理 spec restriction 下的身份
  一致性，且无官方代码。
- 本项目的可验证差异点：GT trajectory identity 作为唯一监督锚点 +
  cross-spec relation-structure consistency + 一个 checkpoint 覆盖
  Dance/MOT17/MOT20/BDD/TAO，不做 dataset-specific 参数。

## 7. 结论

1. 没有发现与本项目「specification-restriction identity consistency」
  直接等价的 2025/2026 官方实现（L4 审计结论保持一致）；
2. Route A 结构（temporal track encoder + persistent state + set-level
   decoder + GT-anchored 监督）的每个组件都有 2025/2026 官方先例可依，
  但组合方式与科学目标没有直接撞车；
3. 代码全部 clean reimplementation，不复制任何第三方实现。
