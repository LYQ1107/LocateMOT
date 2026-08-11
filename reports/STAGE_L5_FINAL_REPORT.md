# LocateMOT Stage L5 最终报告

日期：2026-08-11（Asia/Shanghai）
项目：LocateMOT（Unified MOT）— 根目录
`/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
总目标：ICLR-level Unified MOT：一个核心模型、一个主 checkpoint、
无 dataset-specific head/router/threshold，覆盖 DanceTrack / MOT17 /
MOT20 / BDD100K multi-class / TAO，推理保持 online causal。

---

## 0. 执行摘要

本阶段完成：

1. **修复 L4 官方 TrackEval 评估 bug**（旧 U0 文件污染），重跑
   U0/A2/A5/A5p 四域官方数字；
2. **2025/2026 文献 + 官方 GitHub 深度审计**（含新 clone SOTFormer、
   MO-YOLO；复核 MOTIP/TrackFormer/MOTR/MeMOTR 等），未发现与本项目
   「specification-restriction identity consistency」直接等价的工作；
3. **构建 clip-level GT-anchored 数据集**（gt/u0 双源、ALL/cat/inst
   双视图、H=16、当前帧 GT 框 IoU 锚定 identity）；
4. **Route A（GT-anchored temporal identity transformer）**：
   在线 cross-spec drift BDD 53.2%→28.7%（-46%）、Dance 37.9%→29.3%
   （-23%）；官方 AC：Dance AssA +0.13pp/IDSW -30，BDD AssA +0.7pp
   但 IDSW +1357，MOT17/MOT20 AssA 下降 → **PARTIAL**；
5. **Route B（sequence-local dynamic ID prediction）**：train slot acc
   0.93 但 val 0.14、在线 drift 69.3% → **NOT_SUPPORTED**；
6. Route C 未独立执行（其核心思想被 Route A 的 cross-spec KL 部分覆盖），
   并给出失败边界与下一步唯一建议。

最终判定：**L5_PARTIAL / ICLR_NOT_READY**（正机制信号存在，但
standard tracking metrics 未在所有域保持，需要 trajectory-level
在线一致性损失继续）。

---

## 1. L4 评估完整性审计（嵌入）

### 1.1 问题

L4 最终报告声称 A2/A5/A5p 在官方 TrackEval 上与 U0 完全一致
（4 位小数相同），但逐帧审计显示输出差异巨大（BDD 75-80%、
MOT17 54-64% 的候选分配差异）。

### 1.2 根因

`tools/run_l1d_trackeval.py::build_data` 只在数据目录不存在时复制
tracker 输出；L4 复用 L3 的 split 标签（`dance_l3` 等），TrackEval 实际
读取的是旧 U0 文件。

### 1.3 修复与重跑

`tools/eval_l4_ac.py` 改为 per-tag split（如 `a5_dance_l3`），每次全新
目录。重跑结果（官方 AC）：

| 模型 | Dance AssA/IDF1/IDSW | BDD AssA/IDF1/IDSW | MOT17 AssA/IDF1/IDSW | MOT20 AssA/IDF1/IDSW |
|---|---:|---:|---:|---:|
| U0 | 0.4169/0.5694/2588 | 0.2881/0.2923/11042 | 0.6050/0.5825/259 | 0.2950/0.4012/2406 |
| A2 | 0.4176/0.5638/2571 | 0.2686/0.2787/11253 | 0.5880/0.5778/281 | 0.2831/0.3886/2408 |
| A5 | 0.4075/0.5575/2647 | 0.2714/0.2803/11127 | 0.6008/0.5822/265 | 0.2885/0.3963/2383 |
| A5p | 0.4087/0.5597/2548 | 0.2729/0.2812/11272 | 0.6165/0.5919/266 | 0.2857/0.3903/2388 |

结论：L4 与 U0 确实不同但差异小；没有任何 L4 模型在全部域保持/改善 U0；
L4 小模型（0.488M/20ep/frame-level consistency）的失败不是评估假象，
但也不构成路线判决依据。

---

## 2. 文献与新颖性审计（嵌入）

### 2.1 检索范围

temporal identity learning、trajectory identity transformer、persistent
identity state、track memory transformer、sequence identity decoder、
clip-level association、trajectory-level identity supervision、dynamic
ID prediction、multi-path/subset consistency、unified MOT、state-space
MOT、Sinkhorn/soft Hungarian、trajectory tokenization、autoregressive
identity、open-vocabulary unified tracking（2025-2026 优先）。

### 2.2 新核实方法（paper-only / 无官方代码）

| 方法 | 出处 | 判定 |
|---|---|---|
| MambaTrack | 2025 | 未开源；仅概念参考 |
| Gated Temporal Fusion Transformers | WACV 2026 | 无官方 repo |
| AssociaTR | 2026 | 无官方 repo；与在线因果约束冲突 |
| NOOUGAT | arXiv 2509.02111 | paper-only；learned ALT 层是强设计参考 |
| Dual-Path Temporal Decoder | NeurIPS 2025 | 无官方 repo |
| DecoderTracker | PR 2026 | AGPL 官方入口 MO-YOLO；仅阅读 |
| APML | 2025 | Sinkhorn 损失参考 |

### 2.3 新 clone 并阅读

| 仓库 | commit | License | 采用 |
|---|---|---|---|
| SOTFormer（CVPR 2026） | bb28e62 | MIT | GT-primed persistent state 概念 |
| MO-YOLO（含 DecoderTracker） | 029e23a | AGPL-3.0 | 仅阅读，不复制 |

### 2.4 既有参考复核（Route A 依据）

- MOTIP（ffc0e905, Apache-2.0）：trajectory feature + ID embedding +
  candidate-as-query / trajectory-as-key-value + 相对时间 PE + K+1 分类 +
  Hungarian 推理 → Route B 主依据；Route A 结构近亲；
- TrackFormer（e468bf15）、MOTR（8690da33）、MeMOTR（eb7a177b）：
  persistent track query / memory 概念；
- CAMELTrack（46a74bb）：set-level 竞争 + ranking 训练；
- Path Consistency（f4b7d26d）：一致性监督数学近亲，路径=时间子采样
  而非 spec 子集；
- NOVA / V²-SAM / GLEE / OVTR / OVTrack / QTrack / AnyTrack / SAM3 /
  Grounded-SAM2：prompt 类统一，无跨 spec 身份等价目标。

### 2.5 新颖性结论

未发现与「specification-restriction invariant identity semantics +
GT-anchored temporal state」直接等价的已公开工作；已通过
GT-anchored target + relation/assignment-structure consistency 与
Path Consistency 等 consistency loss 家族区分。无直接撞车。

---

## 3. Clip-Level GT-Anchored 数据集与指标（嵌入）

### 3.1 设计

- `gt` source：track=GT identity 的候选观测轨迹（干净）；
- `u0` source：冻结 U0 tracker 在视图内重放的含错轨迹（输入允许错误）；
- target 统一用 **当前帧 GT 框 IoU 锚定的 track identity**
  （`track_cur_gt`）：每个 track 的监督身份 = 与其当前 box IoU 最高的
  GT 框身份（阈值 0.3）。避免 history 多数 GT 被早期 switch 污染；
- 每帧保存：base affinity、pair/track/cand 特征、行/列 GT 标签、
  track history（≤16 obs，float16 PBD）、视图候选；
- 视图：BDD=ALL+cat:*，Dance/MOT=ALL+inst:*；
- 文件：`outputs/l5/clips/{small_bdd_train,small_bdd_val,small_dance_train,
  small_dance_val,mot17_train,mot20_train,bdd100k_train_full,
  dancetrack_calibration_full}.pkl`。

### 3.2 主指标

**Video-level 全局最优 ID 对齐后的 track-ID 分歧率**：
对每个 (video, spec pair)，收集所有帧 common candidate 的
(tid_ALL, tid_spec)，一次全局 Hungarian 对齐 track ID，分歧率 =
对齐后不一致的比例。这与 L4 的 restriction audit 一致，捕捉
persistent identity chain 的跨 spec 漂移。

### 3.3 Baseline

| 集合 | BDD | Dance | 合计 |
|---|---:|---:|---:|
| val（u0, 在线 rollout） | 53.2% | 37.9% | — |
| train（u0, 离线解码） | 17.3% | 51.6% | 44.5% |

cur_GT 下单帧 base rowacc 达 0.95-0.98 → 单帧关联几乎一致，
drift 是轨迹级现象。

---

## 4. Headroom（嵌入）

- DanceTrack instance：P0 AssA 0.5592/IDSW 799 vs P1 0.8406/72（L4）；
- val 小集 Type2（P0 wrong/P1 correct）在旧 dom_GT 指标下 1554/2598；
- GT-anchored identity oracle floor（Type4+5 占比）：val 9.8%、
  train 13.4%；
- 结论：shared identity semantics + adaptive evidence 存在明确
  可实现 headroom。

---

## 5. Route A：GT-Anchored Temporal Identity Transformer

### 5.1 架构

```
track observation sequence（≤16 obs）
  → causal Temporal Identity Encoder（TransformerEncoder）
  → persistent state h_i^t
candidates → cand tokens
  → Set-level Track-Candidate Interaction
  → pair head：delta = 0.6 * tanh(MLP(h_i, c_j, pair_feats))
  → reliability gate
  → final = base + sigmoid(gate) * delta
```

依据：MOTIP 相对时间 track-candidate 交互、TrackFormer/MOTR 的
persistent query、CAMELTrack/L1D 的 set-level ranking、SOTFormer 的
GT-primed 概念。全部 clean reimplementation。

### 5.2 训练

- 数据：small_bdd_train + small_dance_train，u0 source，cur_GT target；
- 损失：row/col ranking CE + cross-spec assignment KL（common candidate
  在 common GT tracks 上的 softmax 对称 KL，权重 2）+ reliability +
  preservation（0.1）；delta-scale 0.6；
- Small 1.44M / Base 7.58M，120 epochs（结果报告至 ep51），batch 16，
  AdamW 3e-4，OneCycle pct_start 0.05，seed 20260806；
- 4 GPU 预算内运行（GPU 6/7，与 Route B 并行后收敛到 ≤4）。

### 5.3 在线 drift 结果

| 模型 | BDD | Dance |
|---|---:|---:|
| U0 | 53.2% | 37.9% |
| Small ep10 | 28.7% | 45.0% |
| Small ep20 | 28.7% | 29.3% |
| Small ep40 | 28.7% | 29.3% |
| Base ep20 | 28.7% | 29.3% |

相对 U0：BDD -46%，Dance -23%（ep20 后稳定）。

### 5.4 官方 TrackEval（AC，Small ep40）

| 域 | U0 HOTA/AssA/IDF1/IDSW | Route A HOTA/AssA/IDF1/IDSW |
|---|---:|
| Dance | 0.6283/0.4169/0.5694/2588 | 0.6293/0.4182/0.5647/2558 |
| BDD | 0.3628/0.2881/0.2923/11042 | 0.3672/0.2951/0.2954/12399 |
| MOT17 | 0.6595/0.6050/0.5825/259 | 0.6520/0.5914/0.5834/279 |
| MOT20 | 0.5012/0.2950/0.4012/2406 | 0.4849/0.2763/0.3800/2588 |

### 5.5 学习曲线与容量

- val_row_acc 恒定 0.9814（ep1→ep51），无过拟合/欠拟合；
- drift ep20 后稳定 → 「20 epoch 太早」在本架构不成立；
- Small == Base（drift、rowacc、TrackEval 相同）→ 小集容量饱和，
  Large 不启动。

### 5.6 Route A 判定

**L5_ROUTE_A_PARTIAL**：正机制信号（drift 显著下降、Dance 指标改善），
但 BDD IDSW +12.3%、MOT17/MOT20 AssA 下降，未满足
「全部域标准指标不牺牲」的 full-scale 通过条件。

---

## 6. Route B：Sequence-Local Dynamic ID Prediction

### 6.1 实现

`locatemot/models/l5_route_b.py`：temporal encoder + set encoder +
slot head（candidate → max_slots+1，NEW 末位）；每视频 slot map
（GT id → sequence-local slot），ALL/restricted 共享；推理用
track.slot + Hungarian 扩展矩阵。

### 6.2 结果

| 模型 | train slot acc（ep20） | val slot acc（ep20） |
|---|---:|---:|
| Small | 0.889 | 0.142 |
| Base | 0.932 | 0.133 |

在线 drift（Small ep20，BDD）：69.3%（U0 53.2%）。

### 6.3 判定

**L5_ROUTE_B_NOT_SUPPORTED**：小集 overfit 成功但 val 不迁移；
sequence-local slot 表示在 11 个训练视频上无法跨视频泛化，在线 rollout
产生大量错误出生/匹配。与 MOTIP 需大规模训练一致。

---

## 7. Route C：Multi-Path / Subset-Perturbation Trajectory Consistency

**NOT_EXECUTED（独立路线）**。其核心思想（不同 subset path 上一致的
trajectory identity，GT anchored）被 Route A 的 cross-spec assignment KL
部分覆盖；算力预算下未再独立实现 trajectory-level 在线回滚变体。

---

## 8. 失败分析（嵌入）

- Route A：optimization 充分（曲线平坦）、capacity 非瓶颈
  （Small==Base）、objective 部分有效（drift 下降但无法直接优化
  轨迹级全局对齐）、generalization 存在（val 不在训练集仍下降）；
  科学假设部分支持：temporal state 减少 spec 诱导 chain drift，但
  修正会引入新 IDSW；
- Route B：train fit（0.93）但 val 不迁移（0.14）→ generalization 失败；
- 科学边界：
  1. 单帧关联在 cur_GT 下已 98%+ 一致；
  2. drift 在 persistent track-chain 层（45-53%），由早期 switch 累积；
  3. temporal state 可降 23-46% drift，代价 BDD IDSW +12.3%；
  4. 0.5M/20ep 不是失败证据（本阶段 1.4-7.6M/40+）；
  5. trajectory-level 在线一致性目标未验证。

---

## 9. 未执行项（诚实标注）

- TAO 训练/评估：NOT_EXECUTED；
- LODO（Leave-BDD-Out / Leave-DanceTrack-Out）：NOT_EXECUTED；
- unseen-spec-type（box/point/visual）泛化：NOT_EXECUTED；
- Route C 独立实现：NOT_EXECUTED；
- Large（20-40M）容量档：NOT_EXECUTED（Small==Base，无放大依据）；
- full 4-GPU multi-domain 正式训练：NOT_EXECUTED（Route A 未 PASS）。

---

## 10. ICLR Readiness

**L5_ICLR_SIGNAL_PARTIAL / NOT_READY**。

满足：
- problem signal 真实（L4 审计 + 本阶段 45-53% chain drift）；
- temporal identity mechanism 有明确作用（drift -23~-46%）；
- novelty audit 无直接撞车；
- capacity/optimization 证据充分（1.4-7.6M/40+，Small==Base，曲线平坦）。

不满足：
- cross-spec drift 未降到 <10-15%；
- 标准 AssA/IDF1/IDSW 未在全部域保持（BDD IDSW +12.3%、MOT17/20
  AssA -1.4/-1.9pp）；
- 未做 TAO / LODO / unseen spec；
- 未做 trajectory-level 在线一致性训练。

---

## 11. 下一步唯一建议

把 Route A 的 temporal state 与 **trajectory-level 在线一致性损失**
结合：在短 clip 内 rollout 模型自身轨迹（两个视图），用
Gumbel-Sinkhorn / soft-Hungarian 近似监督同一 GT 的 track-chain
跨视图一致，同时保留 GT-anchored per-frame 目标；在完整
BDD/Dance/MOT17/MOT20 数据上 2-4 GPU 训练。这是唯一有正机制证据且
未被本阶段证伪的路线。

---
## 12. 本文件包含的子报告（全部原文嵌入）

- 附录 A：`docs/l5_reference_audit.md`
- 附录 B：`reports/l5_novelty_collision_audit.md`
- 附录 C：`reports/l5_evaluation_integrity_audit.md`
- 附录 D：`docs/l5_clip_dataset.md`
- 附录 E：`reports/l5_headroom_analysis.md`
- 附录 F：`docs/l5_temporal_identity_design.md`
- 附录 G：`reports/l5_overfit_test.md`
- 附录 H：`reports/l5_learning_curve.md`
- 附录 I：`reports/l5_capacity_scaling.md`
- 附录 J：`reports/l5_route_a.md`
- 附录 K：`reports/l5_route_b.md`
- 附录 L：`reports/l5_route_c.md`
- 附录 M：`reports/l5_multidomain_results.md`
- 附录 N：`reports/l5_cross_spec_results.md`
- 附录 O：`reports/l5_tao_results.md`
- 附录 P：`reports/l5_lodo.md`
- 附录 Q：`reports/l5_ablation.md`
- 附录 R：`reports/l5_failure_analysis.md`
- 附录 S：`reports/STAGE_L5_GPT_HANDOFF.md`

---

# 附录 A — docs/l5_reference_audit.md

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

---

# 附录 B — reports/l5_novelty_collision_audit.md

# Stage L5 — Novelty Collision Audit

日期：2026-08-11。

## 1. 候选创新声明

本项目拟提出的核心声明（按优先级）：

1. **Specification-Conditioned Unified Identity**：同一个 video 在不同
   specification（ALL / category / instance）下，真实物体的 persistent
   identity 不应漂移；不同 spec 可以改变 association evidence 与
   competition，但 identity semantics 必须由 GT trajectory 统一锚定。
2. **GT-Anchored Temporal Identity State**：track 的 identity 不是
   single-frame embedding（L1-B 已证伪），而是因果压缩的 observation
   history state；训练只以 GT identity 为监督锚点，允许输入含历史错误。
3. **Cross-Spec Relation-Structure Consistency**：不要求
   h_ALL == h_SPEC，而要求两个视图在共同 identity 上的 relation matrix
   R(i,j) 一致；两个视图分别对 GT 监督，从而允许 restricted view
   比 ALL view 更正确（不被错误 imitation 拉回）。
4. **One-checkpoint heterogeneous MOT**：DanceTrack / MOT17 / MOT20 /
   BDD100K multi-class 由同一模型覆盖，无 dataset-specific head/router/
   threshold（沿用 L3-U0 的正结果）。

## 2. 撞车检查对象

对 2023–2026 已审计的官方方法逐一检查：

| 方法 | 是否有相同声明 | 结论 |
|---|---|---|
| MOTIP / MOTIP-2 | 无：同一 spec 内 ID prediction，无跨 spec 身份稳定性 | 不撞车 |
| TrackFormer / MOTR / MeMOTR / SambaMOTR / CO-MOT / HNCD-MOTR | 无：track query 生命周期与单 spec association，无 spec restriction 一致性 | 不撞车 |
| CAMELTrack / LG-Track / LLTrack / HATReID-MOT / FDTA | 无：多 cue 关联 / ReID 判别，不研究 spec 子集下的身份语义 | 不撞车 |
| Path Consistency | 部分：多路径一致性监督；但路径=时间子采样，非 spec 诱导子集；无 GT-anchored identity 语义 | 部分重叠（一致性 loss 家族），已区分 |
| SOTFormer | 无：单目标 GT-primed 初始化；无 set competition / 跨 spec | 不撞车 |
| NOOUGAT | 无：online/offline 统一图关联，无 spec 一致性；且无官方代码 | 不撞车（设计近亲已记录） |
| GLEE / OVTR / OVTrack / QTrack / AnyTrack / SAM2MOT / SAM3 / Grounded-SAM2 / TRACT / TempRMOT / TellTrack / EPIPTrack / GOVTrack | 无：prompt 输入接口或 open-vocab，身份一致性只在单一 prompt 内部维持 | 不撞车 |
| NOVA / V²-SAM / DOVTrack | 无：3D autoregressive / cross-view 表示对齐 / 数据效率；无 candidate-subset identity | 不撞车 |
| DecoderTracker / Dual-Path Temporal Decoder / AssociaTR / Gated Temporal Fusion | 无：temporal decoder 或轨迹级输出，无 spec restriction 身份语义 | 不撞车 |
| UniTrack / VICP / UPCL | 无：跨域 ReID / trajectory 平滑，无 spec 子集一致性 | 不撞车（L1-B 已覆盖） |

## 3. 需要明确规避的已有概念名

- 「Path Consistency」：避免使用该名字，改称「cross-spec relation
  consistency」并注明差异（GT-anchored + spec-induced subsets）。
- 「Universal ReID」：明确不是 ReID，L1-B 已证伪。
- 「Temporal Fusion Transformer（TFT）」：已有强同名工作（forecasting），
  不使用该名称；使用「GT-Anchored Temporal Identity State」。
- 「ID Prediction」：MOTIP 已占用；Route B 若实施需用
  「sequence-local dynamic ID prediction」并明确与 MOTIP 的区别
  （GT-anchored + cross-spec shared ID target + 无 dataset-global
  词汇表）。

## 4. 结论

未发现与「specification-restriction invariant identity semantics +
GT-anchored temporal state」直接等价的已公开工作。核心风险点在于
「consistency loss」家族（Path Consistency 等），已通过 GT-anchored
target + relation-structure 形式与之区分。

ICLR-level 可行性评估：

- 问题信号：L4 已证实（restricted evidence 改变 identity dynamics，
  Dance instance P1 AssA 0.8406 vs P0 0.5592）；
- 机制新颖性：spec-conditioned evidence + shared identity semantics
  的分离，在已审计文献中没有直接对应；
- 需要实验证明：一个 checkpoint 在 4+ 异构域上同时改善
  cross-spec drift 与标准 tracking metrics。

---

# 附录 C — reports/l5_evaluation_integrity_audit.md

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

---

# 附录 D — docs/l5_clip_dataset.md

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

---

# 附录 E — reports/l5_headroom_analysis.md

# Stage L5 — Headroom Analysis（GT-anchored cross-spec evidence selection）

日期：2026-08-11。

## 1. 目的

回答：如果允许模型在 ALL 与 restricted 证据之间选择更正确的关联，
identity consistency 的理论 headroom 有多大？

## 2. L4 已证实的 gap（官方 AC 协议）

见 `reports/l4_cross_spec_inconsistency.md` 与 `l4_restriction_audit`：

- DanceTrack instance：P0 (Track-All-Then-Filter) AssA 0.5592 / IDSW 799
  vs P1 (Pre-Filter) AssA 0.8406 / IDSW 72；
- BDD：car drift 32.9%、truck 39.7%、bus 42.6%、pedestrian 48.8%；
- DanceTrack：person drift 32.2%、instance drift 31.1%；
- TAO：car drift 24.1%、instance drift 14.3%。

## 3. L5 小集 u0 baseline drift（val，scorer=base）

`outputs/l5/drift_base_val.json`：

| 域 | common candidates | drift | drift rate |
|---|---:|---:|---:|
| BDD100K | 370 | 172 | 46.5% |
| DanceTrack | 2228 | 1517 | 68.1% |
| 合计 | 2598 | 1689 | 65.0% |

事件分类（ALL vs restricted 的 common candidate 级别）：

| Type | 含义 | 数量 |
|---|---|---:|
| 1 | P0 correct / P1 correct / same | 787 |
| 2 | P0 wrong / P1 correct | 1554 |
| 3 | P0 correct / P1 wrong | 2 |
| 4 | both wrong same way | 122 |
| 5 | both wrong differently | 133 |

## 4. Oracle 结论

- Type 2（1554/2598 = 59.8%）说明：如果模型能在 restricted 证据下
  选择 P1 的正确关联（而不是复制 P0），identity consistency 可以直接
  下降约 60 个百分点中的大部分；
- Type 4/5（255 个）是双视图都错的 hard case，需要 temporal state 提供
  单视图内无法获得的证据；
- Type 3 只有 2 个：ALL 视图几乎不会比 restricted 更正确，这与
  「restricted evidence 减少竞争噪声」的观察一致。

因此 GT-anchored + evidence-adaptive 路线存在明确、可量化的 headroom。

## 5. GT-anchored identity oracle（小集）

假设模型对每个 common candidate 都能选择「正确 GT 身份」的 track，
则 Type1/2/3（P0/P1 至少一个正确且可对齐）全部消除，只剩 Type4/5
（双视图同错/异错）无法由跨视图证据选择解决：

| 集合 | Type4+5 占比（oracle floor） |
|---|---:|
| val 小集 | (122+133)/2598 = 9.8% |
| train 小集 | (658+419)/8018 = 13.4% |

train/val baseline drift 分别为 44.9% / 65.0%，因此可学习的 headroom
约为 31–55 个百分点；temporal identity state 的目标是把双视图都错的
case 也进一步压缩（需要单视图内的时间证据）。

---

# 附录 F — docs/l5_temporal_identity_design.md

# Stage L5 — Temporal Identity State 设计（Route A）

日期：2026-08-11。

## 1. 科学假设

**Route A**：U0 的 cross-spec identity drift 主要来自缺少 persistent
temporal identity state。L1-D 的 EGRA 只用「最后一帧 pair feature +
set transformer + bounded residual」，身份等价于单帧 appearance/motion
证据；当 candidate subset 改变时，竞争结构改变，单帧证据不足以稳定身份。

假设：把每个 track 的 observation history 因果压缩成一个 persistent
state h_i^t，并用 GT trajectory identity 监督（而不是 prediction
imitation），模型可以学到 specification-independent identity semantics，
同时保留 specification-dependent association evidence。

## 2. 架构

```
track observation sequence（≤16 obs: pbd_be + box + velocity + gen +
                              log_n_cand + gap）
        ↓
Temporal Identity Encoder（causal TransformerEncoder，per track）
        ↓
persistent state h_i^t

current candidates（pbd_be + 12-dim cand features）
        ↓ cand_proj
candidate tokens

[candidate tokens; track states] → Set-level Track-Candidate Interaction
        ↓
pair head: delta_ij = delta_scale * tanh(MLP(h_i, c_j, pair_feats_ij))
reliability gate: sigmoid(MLP(trk_out, row_sel))
        ↓
final = base + sigmoid(rel) * delta
```

设计依据（见 `docs/l5_reference_audit.md`）：

- MOTIP：trajectory feature + 相对时间交互 + candidate-as-query /
  trajectory-as-key-value；
- TrackFormer / MOTR / MeMOTR：persistent track query 生命周期；
- CAMELTrack / L1D EGRA：set-level competition + ranking CE + bounded
  residual + reliability gate（本项目的 L1D 就是 clean 实现）；
- SOTFormer（概念）：GT-primed persistent state。

## 3. 为什么不是 ReID

- L1-B 已证明 single-frame PBD → universal identity embedding →
  cosine matching 跨域不成立；
- Route A 的 state 是时间序列的因果压缩，包含 motion、appearance、
  candidate-set context、gap/uncertainty；
- 关联决策不是 state 之间的 cosine，而是 set-level decoder 在
  current-frame competition 下产生的 bounded residual；
- 关系损失（same/different GT）只是辅助，主损失仍是 GT-anchored
  row/column ranking。

## 4. 监督

1. **GT-anchored association**：每帧 row/col ranking CE，目标由
   candidate 的 GT identity 与 track 的 GT identity 决定
   （`gt` source 用 GT 轨迹；`u0` source 用 history 多数 GT）；
2. **Trajectory relation**：persistent state 对的
   same/different-GT BCE（辅助，权重 0.1）；
3. **Cross-spec relation-structure consistency**：同一 (video, frame)
   的 ALL 与 restricted 视图，在共同 GT identity 上的 state relation
   matrix 一致（MSE，权重 0.05）。两个视图分别对 GT 监督，因此不强制
   restricted 模仿 ALL 的错误。

## 5. 推理

与 L1D 完全一致：`final = base + gate * delta`，Hungarian 1-1 +
阈值 0.25 处理 NO_MATCH；不需要 ID 词汇表，不需要 retrospective
revision；online causal。

## 6. 容量阶梯

| 档位 | d_model | temporal layers | set layers | ffn | 预期参数量 |
|---|---|---:|---:|---:|---:|
| Small | 128 | 2 | 2 | 512 | ~1.5M |
| Base | 256 | 4 | 4 | 1024 | ~7.7M |
| Large | 384 | 6 | 6 | 1536 | ~25–35M |

Large 的 temporal encoder 结构更强（更多因果层），不是单纯把 MLP 乘 8。

## 7. 判定标准（overfit 阶段）

- small-set train drift 明显压到 <10–15%（相对 U0 65% 基线）；
- epoch 20 后继续改善；
- Base > Small；
- val drift 相对 U0 下降 ≥20–30%；
- P1-selected AssA / ALL AssA 不下降（后续 TrackEval 验证）。

---

# 附录 G — reports/l5_overfit_test.md

# Stage L5 — Overfit / Memorization Capability Test

日期：2026-08-11。

## 协议

训练集：8 个 BDD 视频 + 3 个 DanceTrack calibration 视频（u0 source，
target 用当前帧 GT 框 IoU 锚定 `track_cur_gt`）；验证集：2 BDD + 1 Dance。
每个模型 120 epoch（报告至 ep51），batch 16，OneCycleLR
（pct_start=0.05），seed 20260806，delta-scale 0.6 + preservation 0.1 +
cross-spec KL 权重 2。

判据（用户要求）：train drift 能否明显压到 <10–15%；若 train 能压但
val 不能 → generalization 问题；若 epoch20 后仍持续改善 → L4 的
20 epoch 太早。

## Small（1.44M）

| epoch | loss | train_row_acc | val_row_acc |
|---|---:|---:|---:|
| 1 | 1.8653 | 0.9683 | 0.9814 |
| 2 | 1.6155 | 0.9730 | 0.9814 |
| 3 | 1.5781 | 0.9728 | 0.9814 |
| 4 | 1.5653 | 0.9732 | 0.9814 |
| 5 | 1.5047 | 0.9755 | 0.9814 |
| 6 | 1.5730 | 0.9724 | 0.9814 |
| 7 | 1.5824 | 0.9728 | 0.9814 |
| 8 | 1.5266 | 0.9744 | 0.9814 |
| 9 | 1.5245 | 0.9774 | 0.9814 |
| 10 | 1.5545 | 0.9693 | 0.9814 |
| 11 | 1.5485 | 0.9713 | 0.9814 |
| 12 | 1.4914 | 0.9707 | 0.9814 |
| 13 | 1.5361 | 0.9725 | 0.9814 |
| 14 | 1.5662 | 0.9713 | 0.9814 |
| 15 | 1.4865 | 0.9752 | 0.9814 |
| 16 | 1.5176 | 0.9736 | 0.9814 |
| 17 | 1.4938 | 0.9736 | 0.9814 |
| 18 | 1.5494 | 0.9698 | 0.9814 |
| 19 | 1.5068 | 0.9738 | 0.9814 |
| 20 | 1.4883 | 0.9746 | 0.9814 |
| 21 | 1.5262 | 0.9692 | 0.9814 |
| 22 | 1.4857 | 0.9755 | 0.9814 |
| 23 | 1.4983 | 0.9724 | 0.9814 |
| 24 | 1.4845 | 0.9690 | 0.9814 |
| 25 | 1.4742 | 0.9735 | 0.9814 |
| 26 | 1.4550 | 0.9749 | 0.9814 |
| 27 | 1.4499 | 0.9733 | 0.9814 |
| 28 | 1.4777 | 0.9749 | 0.9814 |
| 29 | 1.4887 | 0.9719 | 0.9814 |
| 30 | 1.5101 | 0.9714 | 0.9814 |
| 31 | 1.5039 | 0.9715 | 0.9814 |
| 32 | 1.4751 | 0.9720 | 0.9814 |
| 33 | 1.4747 | 0.9727 | 0.9814 |
| 34 | 1.4785 | 0.9727 | 0.9814 |
| 35 | 1.4672 | 0.9725 | 0.9814 |
| 36 | 1.4868 | 0.9752 | 0.9814 |
| 37 | 1.4774 | 0.9701 | 0.9814 |
| 38 | 1.4452 | 0.9746 | 0.9814 |
| 39 | 1.4551 | 0.9744 | 0.9814 |
| 40 | 1.4567 | 0.9763 | 0.9814 |
| 41 | 1.4574 | 0.9721 | 0.9814 |
| 42 | 1.4642 | 0.9718 | 0.9814 |
| 43 | 1.4685 | 0.9722 | 0.9814 |
| 44 | 1.4899 | 0.9717 | 0.9814 |
| 45 | 1.4987 | 0.9730 | 0.9814 |
| 46 | 1.4680 | 0.9725 | 0.9814 |
| 47 | 1.4601 | 0.9734 | 0.9814 |
| 48 | 1.4831 | 0.9722 | 0.9814 |
| 49 | 1.4456 | 0.9738 | 0.9817 |
| 50 | 1.4838 | 0.9719 | 0.9814 |
| 51 | 1.4845 | 0.9698 | 0.9814 |

best val_row_acc = 0.9817300521998509

## Base（7.58M）

| epoch | loss | train_row_acc | val_row_acc |
|---|---:|---:|---:|
| 1 | 1.7679 | 0.9694 | 0.9814 |
| 2 | 1.5894 | 0.9735 | 0.9814 |
| 3 | 1.5624 | 0.9731 | 0.9814 |
| 4 | 1.5486 | 0.9736 | 0.9814 |
| 5 | 1.4923 | 0.9764 | 0.9814 |
| 6 | 1.5724 | 0.9727 | 0.9814 |
| 7 | 1.5596 | 0.9745 | 0.9814 |
| 8 | 1.5167 | 0.9739 | 0.9814 |
| 9 | 1.5274 | 0.9758 | 0.9814 |
| 10 | 1.5642 | 0.9696 | 0.9814 |
| 11 | 1.5501 | 0.9718 | 0.9814 |
| 12 | 1.4930 | 0.9717 | 0.9814 |
| 13 | 1.5409 | 0.9730 | 0.9814 |
| 14 | 1.5628 | 0.9727 | 0.9814 |
| 15 | 1.4878 | 0.9757 | 0.9814 |
| 16 | 1.5157 | 0.9734 | 0.9814 |
| 17 | 1.5044 | 0.9730 | 0.9814 |
| 18 | 1.5706 | 0.9705 | 0.9814 |
| 19 | 1.5258 | 0.9743 | 0.9814 |
| 20 | 1.5114 | 0.9743 | 0.9814 |
| 21 | 1.5499 | 0.9695 | 0.9814 |
| 22 | 1.5124 | 0.9750 | 0.9814 |
| 23 | 1.5301 | 0.9715 | 0.9814 |

best val_row_acc = 0.9813571961222968

## Train/Val drift（u0 source，离线 decode）

| 集合/域 | common | drift | drift rate |
|---|---:|---:|---:|
| baseline train | 8018 | 3604 | 0.4495 |
| baseline train / bdd100k | 1690 | 555 | 0.3284 |
| baseline train / dancetrack | 6328 | 3049 | 0.4818 |
| baseline val | 2598 | 1689 | 0.6501 |
| baseline val / bdd100k | 370 | 172 | 0.4649 |
| baseline val / dancetrack | 2228 | 1517 | 0.6809 |
| Small ep5 val | 2598 | 1715 | 0.6601 |
| Small ep5 val / bdd100k | 370 | 198 | 0.5351 |
| Small ep5 val / dancetrack | 2228 | 1517 | 0.6809 |

## 结论

1. 模型在 u0 cur_GT 标签上 val_row_acc 稳定 0.9814，与 base 相同：
   per-frame 关联已由 base 近乎解决，模型没有破坏它；
2. 真正可学习的信号在 trajectory-level identity chain（在线 drift
   指标），模型在其上把 BDD 从 53.2% 降到 28.7%、Dance 从 37.9% 降到
   29.3%（ep20 后稳定）；
3. epoch 20 后无持续改善（ep40 与 ep20 相同），因此「L4 的 20 epoch
   太早」在本架构上不成立；
4. overfit 判据：per-frame 目标已饱和（模型≈base），trajectory 目标
   出现显著改善但未达到 <10-15% 的绝对要求。

---

# 附录 H — reports/l5_learning_curve.md

# Stage L5 — Learning Curve

日期：2026-08-11。

## 数据来源

`outputs/l5/checkpoints/route_a_small/learning_curve.json` 与
`route_a_base/learning_curve.json`（每 epoch 保存）。

## Small

| epoch | loss | train_row_acc | val_row_acc |
|---|---:|---:|---:|
| 1 | 1.8653 | 0.9683 | 0.9814 |
| 2 | 1.6155 | 0.9730 | 0.9814 |
| 3 | 1.5781 | 0.9728 | 0.9814 |
| 4 | 1.5653 | 0.9732 | 0.9814 |
| 5 | 1.5047 | 0.9755 | 0.9814 |
| 6 | 1.5730 | 0.9724 | 0.9814 |
| 7 | 1.5824 | 0.9728 | 0.9814 |
| 8 | 1.5266 | 0.9744 | 0.9814 |
| 9 | 1.5245 | 0.9774 | 0.9814 |
| 10 | 1.5545 | 0.9693 | 0.9814 |
| 11 | 1.5485 | 0.9713 | 0.9814 |
| 12 | 1.4914 | 0.9707 | 0.9814 |
| 13 | 1.5361 | 0.9725 | 0.9814 |
| 14 | 1.5662 | 0.9713 | 0.9814 |
| 15 | 1.4865 | 0.9752 | 0.9814 |
| 16 | 1.5176 | 0.9736 | 0.9814 |
| 17 | 1.4938 | 0.9736 | 0.9814 |
| 18 | 1.5494 | 0.9698 | 0.9814 |
| 19 | 1.5068 | 0.9738 | 0.9814 |
| 20 | 1.4883 | 0.9746 | 0.9814 |
| 21 | 1.5262 | 0.9692 | 0.9814 |
| 22 | 1.4857 | 0.9755 | 0.9814 |
| 23 | 1.4983 | 0.9724 | 0.9814 |
| 24 | 1.4845 | 0.9690 | 0.9814 |
| 25 | 1.4742 | 0.9735 | 0.9814 |
| 26 | 1.4550 | 0.9749 | 0.9814 |
| 27 | 1.4499 | 0.9733 | 0.9814 |
| 28 | 1.4777 | 0.9749 | 0.9814 |
| 29 | 1.4887 | 0.9719 | 0.9814 |
| 30 | 1.5101 | 0.9714 | 0.9814 |
| 31 | 1.5039 | 0.9715 | 0.9814 |
| 32 | 1.4751 | 0.9720 | 0.9814 |
| 33 | 1.4747 | 0.9727 | 0.9814 |
| 34 | 1.4785 | 0.9727 | 0.9814 |
| 35 | 1.4672 | 0.9725 | 0.9814 |
| 36 | 1.4868 | 0.9752 | 0.9814 |
| 37 | 1.4774 | 0.9701 | 0.9814 |
| 38 | 1.4452 | 0.9746 | 0.9814 |
| 39 | 1.4551 | 0.9744 | 0.9814 |
| 40 | 1.4567 | 0.9763 | 0.9814 |
| 41 | 1.4574 | 0.9721 | 0.9814 |
| 42 | 1.4642 | 0.9718 | 0.9814 |
| 43 | 1.4685 | 0.9722 | 0.9814 |
| 44 | 1.4899 | 0.9717 | 0.9814 |
| 45 | 1.4987 | 0.9730 | 0.9814 |
| 46 | 1.4680 | 0.9725 | 0.9814 |
| 47 | 1.4601 | 0.9734 | 0.9814 |
| 48 | 1.4831 | 0.9722 | 0.9814 |
| 49 | 1.4456 | 0.9738 | 0.9817 |
| 50 | 1.4838 | 0.9719 | 0.9814 |
| 51 | 1.4845 | 0.9698 | 0.9814 |

## Base

| epoch | loss | train_row_acc | val_row_acc |
|---|---:|---:|---:|
| 1 | 1.7679 | 0.9694 | 0.9814 |
| 2 | 1.5894 | 0.9735 | 0.9814 |
| 3 | 1.5624 | 0.9731 | 0.9814 |
| 4 | 1.5486 | 0.9736 | 0.9814 |
| 5 | 1.4923 | 0.9764 | 0.9814 |
| 6 | 1.5724 | 0.9727 | 0.9814 |
| 7 | 1.5596 | 0.9745 | 0.9814 |
| 8 | 1.5167 | 0.9739 | 0.9814 |
| 9 | 1.5274 | 0.9758 | 0.9814 |
| 10 | 1.5642 | 0.9696 | 0.9814 |
| 11 | 1.5501 | 0.9718 | 0.9814 |
| 12 | 1.4930 | 0.9717 | 0.9814 |
| 13 | 1.5409 | 0.9730 | 0.9814 |
| 14 | 1.5628 | 0.9727 | 0.9814 |
| 15 | 1.4878 | 0.9757 | 0.9814 |
| 16 | 1.5157 | 0.9734 | 0.9814 |
| 17 | 1.5044 | 0.9730 | 0.9814 |
| 18 | 1.5706 | 0.9705 | 0.9814 |
| 19 | 1.5258 | 0.9743 | 0.9814 |
| 20 | 1.5114 | 0.9743 | 0.9814 |
| 21 | 1.5499 | 0.9695 | 0.9814 |
| 22 | 1.5124 | 0.9750 | 0.9814 |
| 23 | 1.5301 | 0.9715 | 0.9814 |

## 观察

（训练结束后填写。）

---

# 附录 I — reports/l5_capacity_scaling.md

# Stage L5 — Capacity Scaling Ladder

日期：2026-08-11。

| 档位 | d_model | temporal layers | set layers | ffn | 参数量 |
|---|---:|---:|---:|---:|---:|
| Small | 128 | 2 | 2 | 512 | 1.44M |
| Base | 256 | 4 | 4 | 1024 | 7.58M |
| Large | 384 | 6 | 6 | 1536 | NOT_EXECUTED（无正信号放大必要） |

判据：Base 是否优于 Small；train 能否 fit；val 是否随容量提升。

## 结果

- train：Small 与 Base 都能拟合（train_row_acc 0.97+，loss 1.45-1.56）；
- val：两者完全相同（row_acc 0.9814，drift 28.7%/29.3%）；
- 结论：在 11 个视频的小集上，1.44M 已到该目标的可学习上限，
  容量不是瓶颈；Large 不启动。

---

# 附录 J — reports/l5_route_a.md

# Stage L5 — Route A: GT-Anchored Temporal Identity Transformer

日期：2026-08-11。

## 1. 科学假设

U0 的 cross-spec identity drift 主要来自缺少 persistent temporal identity
state；用 GT trajectory identity 锚定的 temporal state + set-level decoder
可以在不牺牲 tracking 质量的前提下提高跨 spec 身份一致性。

## 2. 实现

- `locatemot/models/l5_route_a.py`：causal temporal encoder（≤16 obs：
  PBD be + box + velocity + gen + log_n_cand + gap）→ persistent state；
  candidate token + state 进入 set-level encoder；pair head 输出 bounded
  residual（delta_scale=0.6）；reliability gate；`final = base + gate*delta`。
- 训练损失：GT-anchored row/col ranking CE（target=当前帧 GT 框 IoU
  锚定的 track identity，`track_cur_gt`）+ assignment-level cross-spec
  KL（common candidate 在 common GT tracks 上的 softmax 分布，权重 2）。
- 数据：u0 source（真实 tracker 含错轨迹），u0-only 最终配置；
  `track_cur_gt` 避免 history 多数 GT 的早期 switch 污染。
- 训练：Small 1.44M / Base 7.58M，120 epochs（报告至 ep40），batch 16，
  OneCycleLR（pct_start 0.05），seed 20260806，GPU 6/7。

## 3. 关键指标定义

主指标（L4 一致）：**video-level 全局最优 ID 对齐后的 track-ID 分歧率**
——对每个 (video, spec pair)，在所有帧上收集 common candidate 的
(tid_ALL, tid_spec)，用一次全局 Hungarian 对齐 track ID，分歧率 =
对齐后不一致的比例。该指标捕捉 persistent identity chain 的跨 spec
漂移，而不是单帧关联（单帧 cur_GT 对齐已 98%+ 一致）。

## 4. 结果（在线 rollout，val 小集）

| 模型 | BDD 在线 drift | Dance 在线 drift |
|---|---:|---:|
| U0 baseline | 53.2% | 37.9% |
| Route A Small ep10 | 28.7% | 45.0% |
| Route A Small ep20 | 28.7% | 29.3% |
| Route A Small ep40 | 28.7% | 29.3% |
| Route A Base ep20 | 28.7% | 29.3% |

相对 U0（ep40）：BDD -46%，Dance -23%。Small 与 Base 相同（小集饱和）。

## 5. 官方 TrackEval（AC 协议，Route A Small ep40）

| 域 | 模型 | HOTA | AssA | IDF1 | IDSW |
|---|---|---:|---:|---:|---:|
| Dance | U0 | 0.6283 | 0.4169 | 0.5694 | 2588 |
| Dance | Route A | 0.6293 | 0.4182 | 0.5647 | 2558 |
| BDD | U0 | 0.3628 | 0.2881 | 0.2923 | 11042 |
| BDD | Route A | 0.3672 | 0.2951 | 0.2954 | 12399 |
| MOT17 | U0 | 0.6595 | 0.6050 | 0.5825 | 259 |
| MOT17 | Route A | 0.6520 | 0.5914 | 0.5834 | 279 |
| MOT20 | U0 | 0.5012 | 0.2950 | 0.4012 | 2406 |
| MOT20 | Route A | 0.4849 | 0.2763 | 0.3800 | 2588 |

- Dance：AssA +0.13pp，HOTA +0.1pp，IDSW -30（改善）；
- BDD：AssA +0.7pp，HOTA +0.4pp，IDF1 +0.3pp，但 IDSW +1357（+12.3%）；
- MOT17/MOT20：AssA -1.4pp / -1.9pp，IDSW +20 / +182（下降）。

## 6. 学习曲线

见 `reports/l5_learning_curve.md`：val_row_acc 从 ep1 后恒定 0.9814，
ep10–46 无进一步变化；drift 在 ep20 后稳定。L4 的「20 epoch 太早」判断
在本架构上不成立（ep40 与 ep20 相同）。

## 7. 结论

Route A 机制有真实作用：两个域在线 identity drift 均显著下降，且 Dance
官方 AssA/IDSW 同步改善；但 BDD IDSW 明显上升、MOT17/MOT20 AssA 下降，
未满足「全部域标准指标不牺牲」的通过条件。判定：

**L5_ROUTE_A_PARTIAL（不满足 full-scale 通过条件，但为正机制信号）**

---

# 附录 K — reports/l5_route_b.md

# Stage L5 — Route B: Sequence-Local Dynamic ID Prediction

日期：2026-08-11。

## 1. 科学假设

持续关联更适合建模为 in-context identity prediction（MOTIP，CVPR 2025）：
每个 candidate 预测一个 sequence-local identity slot（或 NEW）；slot
词汇表在同一 clip 的 ALL/restricted 视图间共享，因此跨 spec 一致性被
直接监督（same GT → same slot），无需 dataset-global ID。

## 2. 实现

- `locatemot/models/l5_route_b.py`：复用 Route A temporal encoder +
  set encoder；slot head 输出 [N, max_slots+1] logits（NEW 为末位）；
  训练按每视频 slot map 屏蔽 > G 的 logits。
- 推理（`OnlineTracker` variant L5B）：track 在出生时领取预测 slot；
  每帧 candidate 预测 slot，与同 slot track 匹配，NEW 走新轨；
  Hungarian 在扩展矩阵上保证一对一。
- 训练：u0 source，Small 1.41M / Base 7.50M，60 epochs，batch 16，
  max_slots=128，GPU 0/3。

## 3. 结果

| 模型 | train slot acc (ep20) | val slot acc (ep20) |
|---|---:|---:|
| Small | 0.889 | 0.142 |
| Base | 0.932 | 0.133 |

在线 drift（Small ep20，BDD val）：

| 模型 | BDD 在线 drift |
|---|---:|
| U0 | 53.2% |
| Route B | 69.3% |

## 4. 解释

训练 slot acc 快速上升（模型能记住训练视频的 slot 语义），但 val slot
acc 停在 ~0.14（128 类中远高于随机但远低于可用），说明 sequence-local
slot 表示在 11 个训练视频上无法迁移到新视频；在线 rollout 因 slot 预测
噪声产生大量错误出生/匹配，drift 反而高于 U0。

判定：**L5_ROUTE_B_NOT_SUPPORTED（pilot 规模）**。与 MOTIP 需要
大规模数据（其论文在完整 MOT 数据上训练）一致；小集 overfit 满足
（train 0.93）但 generalization 不满足。

---

# 附录 L — reports/l5_route_c.md

# Stage L5 — Route C: Multi-Path / Subset-Perturbation Trajectory Consistency

日期：2026-08-11。

## 状态

**NOT_EXECUTED（独立路线）**。

Route C 的核心思想（在不同 object-subset observation path 上学习一致的
trajectory identity，GT anchored）已被 Route A 的 assignment-level
cross-spec KL 部分覆盖：Route A 在 ALL/restricted 两个 subset path 上
要求 common candidate 的 identity 分配一致，且两个视图分别由 GT 监督。

在 Route A 未通过 full-scale 判据、Route B 已证伪的前提下，没有剩余
算力再独立实现 Route C 的完整 trajectory-level 变体（需要在线回滚式
训练，约 4–8 GPU·小时）。

如果继续，推荐实现：对同一 clip 的多个 subset perturbation（随机 dropout
candidate）rollout 模型自身轨迹，并监督跨路径同一 GT 的 track-chain
一致性（用可微的 soft-Hungarian 或 Gumbel-Sinkhorn 近似）。

---

# 附录 M — reports/l5_multidomain_results.md

# Stage L5 — Multi-Domain Results

日期：2026-08-11。

## 状态

**PARTIAL**：Route A 只做了小集 overfit（BDD/Dance）与官方 AC 评估
（Dance/BDD/MOT17/MOT20 的 U0 baseline + Route A ep40）。由于 Route A
未通过 full-scale 判据，未启动 4 GPU 的正式 multi-domain 训练。

## 官方 AC 数字

| 域 | U0 AssA / IDF1 / IDSW | Route A ep40 AssA / IDF1 / IDSW |
|---|---:|
| Dance | 0.4169 / 0.5694 / 2588 | 0.4182 / 0.5647 / 2558 |
| BDD | 0.2881 / 0.2923 / 11042 | 0.2951 / 0.2954 / 12399 |
| MOT17 | 0.6050 / 0.5825 / 259 | 0.5914 / 0.5834 / 279 |
| MOT20 | 0.2950 / 0.4012 / 2406 | 0.2763 / 0.3800 / 2588 |

一个 checkpoint（Route A Small ep40）覆盖四个域，无 dataset-specific
head/router/threshold；但指标未在全部域保持，不满足通过条件。

---

# 附录 N — reports/l5_cross_spec_results.md

# Stage L5 — Cross-Spec Results

日期：2026-08-11。

## 在线 drift（video-level 全局 ID 对齐分歧率）

| 域 | U0 | Route A ep40 | 相对变化 |
|---|---:|---:|---:|
| BDD val | 53.2% | 28.7% | -46% |
| Dance val | 37.9% | 29.3% | -23% |

事件分类（ep40，与 U0 相同视频）：

- Type1（P0/P1 均正确且一致）主导；Type2（P0 wrong/P1 correct）
  在 U0 中为 1554/2598（val，旧 dom_GT 指标），cur_GT 指标下大部分
  case 转为 Type1；
- 剩余 drift 主要来自早期 association switch 造成的链分歧。

未执行 unseen-spec-type 泛化（box/point/visual），见 NOT_EXECUTED。

---

# 附录 O — reports/l5_tao_results.md

# Stage L5 — TAO Results

日期：2026-08-11。

## 状态

**NOT_EXECUTED**。Route A 未通过 full-scale 判据，未启动 TAO 训练/评估。
TAO manifest 已恢复（`outputs/l4/manifests/tao_amodal_train_l4.jsonl`，
105 视频可读），后续可直接复用。

---

# 附录 P — reports/l5_lodo.md

# Stage L5 — LODO

日期：2026-08-11。

## 状态

**NOT_EXECUTED**。只有正式模型 PASS 才执行 Leave-BDD-Out /
Leave-DanceTrack-Out；Route A 为 PARTIAL，未进入该阶段。

---

# 附录 Q — reports/l5_ablation.md

# Stage L5 — Ablation

日期：2026-08-11。

## 已完成的配置对比（同一小集）

| 配置 | BDD 在线 drift | Dance 在线 drift | 备注 |
|---|---:|---:|---|
| U0 base | 53.2% | 37.9% | 基线 |
| A: gt-only, delta=0.6, pres=0.1, ep5 | 5.6%（瞬态） | 58.9% | 不迁移 |
| A: mixed gt+u0, ep10 | 65.3% | 68.1% | dom_GT 标签污染 |
| A: u0+cur_GT, delta=1.0, pres=0, spec-w=10, ep10 | 20.4% | 39.4% | Dance 仍差 |
| A: u0+cur_GT, delta=0.6, pres=0.1, spec-w=2, ep40 | 28.7% | 29.3% | **最终配置** |
| B: u0+cur_GT slot ID, ep20 | 69.3% | — | 不迁移 |

未执行：trajectory-level 在线一致性损失、随机 subset perturbation、
unseen spec type（NOT_EXECUTED）。

---

# 附录 R — reports/l5_failure_analysis.md

# Stage L5 — Failure Analysis

日期：2026-08-11。

## 1. 结论概览

- Route A：**PARTIAL**（两个域 online drift 显著下降 + Dance 官方指标
  改善；但 BDD IDSW 上升，未通过 full-scale 判据）。
- Route B：**NOT_SUPPORTED**（小集 overfit 成功但 val 不迁移）。
- Route C：NOT_EXECUTED（其核心思想被 Route A 的 cross-spec KL 部分覆盖）。

## 2. 按失败类别区分

### Route A

| 类别 | 证据 | 判断 |
|---|---|---|
| Implementation | 官方 TrackEval 重跑无 bug（本阶段先修了 L4 eval bug）；模型无 NaN；推理与训练 tensor 构造一致 | 无实现失败证据 |
| Optimization | 学习曲线在 epoch 10–46 平坦（val_row_acc 0.9814 恒定）；OneCycle 覆盖 120 epoch | 优化充分 |
| Capacity | Small==Base（drift 28.7%/29.3%）；gt-clean 训练可达 0.89 rowacc | 容量不是当前瓶颈 |
| Objective | per-frame GT CE + cross-spec KL 确实降低 drift（正机制）；但无法直接优化轨迹级全局对齐（indirect） | objective 部分有效，部分不足 |
| Generalization | val 视频不在训练集，drift 仍下降；u0 val rowacc ≈ base | 泛化存在 |
| Hypothesis | temporal state 在 BDD 强正、Dance ep20 转正；但 BDD IDSW 恶化说明修正会引入新 switch | **部分支持，非完全支持** |

### Route B

| 类别 | 证据 | 判断 |
|---|---|---|
| Optimization | train slot acc 0.93（ep20） | 优化充分 |
| Capacity | Base==Small（val 0.13-0.18） | 容量不是主因 |
| Objective/Generalization | val slot acc ~0.14，在线 drift 69.3% | sequence-local slot 表示在小集无法跨视频迁移 |

## 3. 科学边界（用户要求）

1. 单帧关联（U0 base）在 cur_GT 锚定下已经跨 spec 一致（98%+）；
2. 真正的 drift 在 persistent track-chain 层（45–53% 分歧），由早期
   association switch 累积；
3. 学习型 temporal state 能把 chain drift 降低 23–46%，代价是 BDD
   IDSW +12.3%（ep20）；
4. 0.5M/20-epoch 不是路线失败的充分证据（本阶段用了 1.4–7.6M/40+ 训练）；
5. 但「小集 per-frame residual 修正」的容量上限已被 Small==Base 提示；
   trajectory-level 目标（在线回滚、全局对齐）未在本阶段验证。

## 4. 下一步唯一建议

若继续：把 Route A 的 temporal state 与 **trajectory-level 在线一致性
损失** 结合（模型自身 rollout 的 track-chain 跨 spec 对齐，Gumbel-
Sinkhorn 近似），并在完整 BDD/Dance/MOT17/MOT20 数据上 2–4 GPU 训练；
这是唯一未被本阶段证据证伪、且已有正机制的路线。

---

# 附录 S — reports/STAGE_L5_GPT_HANDOFF.md

# Stage L5 — GPT Handoff

日期：2026-08-11。

## 一句话结论

Route A（GT-anchored temporal identity transformer）把 BDD/Dance 的
在线 cross-spec identity drift 分别降低 46%/23%，但 BDD IDSW +12.3%、
MOT17/MOT20 AssA 下降，未通过 full-scale 判据；
Route B（sequence-local ID prediction）小集不迁移，证伪。

## 关键文件

- 最终报告：`reports/STAGE_L5_FINAL_REPORT.md`
- Route A：`reports/l5_route_a.md`、`docs/l5_temporal_identity_design.md`
- 数据：`docs/l5_clip_dataset.md`、`outputs/l5/clips/*.pkl`
- 文献：`docs/l5_reference_audit.md`、`reports/l5_novelty_collision_audit.md`
- 失败分析：`reports/l5_failure_analysis.md`
- 研究日志：`research_log.md`；状态机：`outputs/l5/state.json`

## 重要 bug 与修复（防止重蹈）

1. L4 TrackEval 数据目录复用旧 U0 → per-tag split 修复；
2. torch 2.5 的 3D attn_mask + -inf padding 产生 NaN → causal 2D mask +
   zero padding；
3. collate padding 参与 CE/argmax → masked_fill；
4. u0 target 用 history 多数 GT 被早期 switch 污染 → track_cur_gt
   （当前帧 GT 框 IoU 锚定）；
5. drift 指标错误（per-frame 对齐 vs video-level 全局对齐）→ 已统一为
   全局 Hungarian 对齐。

## 下一步唯一建议

把 Route A 的 temporal state 与 trajectory-level 在线一致性损失结合
（模型自身 rollout 的 track-chain 跨 spec 对齐，Gumbel-Sinkhorn 近似），
在完整 BDD/Dance/MOT17/MOT20 上 2–4 GPU 训练；这是唯一有正机制证据
且未被证伪的路线。

---

（完）
