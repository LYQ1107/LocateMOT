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

## 12. 报告清单

- `reports/STAGE_L5_FINAL_REPORT.md`（本文件）
- `reports/STAGE_L5_GPT_HANDOFF.md`
- `reports/l5_evaluation_integrity_audit.md`
- `reports/l5_headroom_analysis.md`
- `reports/l5_novelty_collision_audit.md`
- `reports/l5_overfit_test.md`、`reports/l5_learning_curve.md`、
  `reports/l5_capacity_scaling.md`
- `reports/l5_route_a.md`、`reports/l5_route_b.md`、
  `reports/l5_route_c.md`
- `reports/l5_multidomain_results.md`、`reports/l5_cross_spec_results.md`
- `reports/l5_tao_results.md`、`reports/l5_lodo.md`、
  `reports/l5_ablation.md`
- `reports/l5_failure_analysis.md`
- `docs/l5_reference_audit.md`、`docs/l5_clip_dataset.md`、
  `docs/l5_temporal_identity_design.md`
- `research_log.md`、`outputs/l5/state.json`
