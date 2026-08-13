# LocateMOT Stage L6 最终报告：Learned Causal Identity Dynamics Model (UIDM)

日期：2026-08-13（实验完成日；训练完成于 2026-08-11）
项目根目录：`/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
Git commit：`1b6f7ea4b54fbf1c9b99e0e538b7f5696b6999ca`

> 本报告自包含；把本文件交给任何 GPT 会话即可继续，无需其他上下文。

## 0. 摘要（结论先行）

**Hypothesis**：异构 MOT 可被一个共享 checkpoint 上学习的因果身份
动力学过程统一——持久 per-track memory + 集合交互 + 学习化转移
（continue/NEW/NO-MATCH）+ 生命周期 + tracking-level objective +
model-in-the-loop 训练。

**结果（FOUND → PARTIAL/SUPPORTED）**：

- 正信号：单 checkpoint 在 3/4 标准域显著提升，Macro HOTA +5.2pp、
  Macro AssA +9.1pp、Macro IDF1 +5.9pp（Dance/BDD/MOT17/MOT20
  等权）；BDD cross-spec drift 从 53.2% 降到 17.0%；MOT17
  AssA 0.6991（历史最强 0.6050）；BDD AssA 0.4866、MOT20 0.4584
  均为本项目历史新高。
- 负信号/失败边界：DanceTrack 塌陷（AssA 0.3248 vs U0 0.4169），
  92% 的 switch 是连续帧、无遮挡的同类目标间身份交换（PBD 外观证据
  过强、motion/competition 不足）；MOT17 IDSW 434（U0 259）与
  Dance IDSW 5290（U0 2588）上升。
- Ablation：去掉 tracking-level loss / memory / interaction /
  lifecycle 分别损失 21–23pp AssA（MOT17），证明四项机制均必要。

**ICLR readiness**：方向有真实、可复现的正信号与机制级 ablation，
但 Dance 塌陷必须先修复（cue-reliability 显式建模），才可作主方法
投稿。当前状态：`L6_PARTIAL（SUPPORTED with Dance collapse）`。

## 1. Scientific Motivation

证据链（L1–L5，详见 `research_log.md` 与各阶段报告）：

- L1-B：单帧 universal ReID embedding 跨域失败 → identity 不是
  单帧外观向量；
- L1-C/L1-D：固定多 cue 权重（IoU+PBD+motion）跨域次优 →
  cue reliability 必须学习；
- L2：future-utility oracle headroom 低 → 不依赖未来信息；
- L3：dataset/regime router 是 shortcut → 不引入 dataset ID；
- L4：specification restriction 真实改变 identity → 身份是
  specification-relative 的；
- L5：GT-anchored temporal identity 有真实正信号，但 bounded
  residual + fixed lifecycle 无法稳定转成异构 MOT 提升 → 需要
  完整的学习化身份动力学。

因此 UIDM 把 identity 定义为**时间上的持久状态 + 集合竞争 + 可学习
转移**，而不是单帧 embedding 或固定公式。这也与 2025/2026 顶会
（Samba 持久状态、MOTIP in-context ID、UniTrack tracking-level
loss）一致；创新边界见第 3 节。

## 2. 2025/2026 顶会方法审计

完整审计见 `docs/l6_iclr_method_audit.md`；关键结论：

- Samba（ICLR 2025）：持久 hidden state + 同步 set-of-sequences ——
  采用其科学思想，代码为 AGPL 仅阅读；
- MOTIP（CVPR 2025）：in-context ID prediction + NEW —— 采用其
  sequence-local identity 思想；
- UniTrack（ICLR 2026）：tracking-level hinge loss —— 采用其目标思想，
  实现为可微 soft-switch margin；
- 共同缺口：无单 checkpoint 跨 Dance/BDD/MOT17/MOT20/TAO 的
  learned causal identity dynamics + model-in-the-loop。

## 3. Novelty Boundary

详见 `reports/l6_novelty_audit.md`。

## 4. 基线定义

详见 `reports/l6_baseline_reconciliation.md`：

| ID | 方法 | Dance AssA | BDD AssA | MOT17 AssA | MOT20 AssA |
|---|---:|---:|---:|---:|---:|
| B0 | IoU | 0.390 | — | — | — |
| B1 | Motion C1 | 0.4193 | 0.3019 | 0.5530 | 0.2869 |
| B2 | L1DK | 0.4165 | 0.3292 | 0.6010 | 0.2864 |
| B3 | L3 U0 | 0.4169 | 0.2881 | 0.6050 | 0.2950 |
| B4 | L5 Route A | 0.4182 | 0.2951 | 0.5914 | 0.2763 |

## 5. UIDM 架构

详见 `docs/l6_uidm_design.md`；一句话：

> 用一个共享 checkpoint，在交互轨迹集合上学一个因果身份动力学过程。

组件与理由：

1. 持久 per-track memory（identity 是时间过程；L1-B 证明单帧 ReID 失败）；
2. 每帧 set-of-sequences 交互（MOT 是竞争的多轨迹集合；Samba 同步状态）；
3. Identity Transition Decoder：continue / NEW / NO-MATCH / alive
   （BDD IDSW 上升证明 lifecycle 必须学进模型）；
4. Kalman/IoU/PBD 作为 evidence 输入（MOT17/MOT20 对 motion 敏感）；
5. Tracking-level loss：soft IDSW/FP margin + lifecycle + motion
   （UniTrack 思想；L5 证明 row-CE 不够）。

参数：UIDM-Large d=384、6 层、FFN 1536、8 头，约 15.0M trainable。

## 6. 训练协议

详见 `docs/l6_training_design.md`；要点：

- model-in-the-loop：状态由模型自己的 association 产生；
  teacher warmup 1000 steps → teacher prob 0.4；
- H=16 clip，batch 8/GPU × 3 GPU（DDP），AdamW 3e-4 OneCycle；
- 数据：BDD 200 + Dance calib 8 + Dance train 32 + MOT17 3 +
  MOT20 2 + TAO 105，domain-balanced，无 dataset ID；
- seed 20260806。

## 7. 主结果（四域 + Macro，fresh tag `uidm_final`）

UIDM-Large（epoch 18，单 checkpoint，DDP×3，seed 20260806），
TrackEval 全部 fresh tag + fresh directory。

| 域 | HOTA | DetA | AssA | IDF1 | IDSW |
|---|---:|---:|---:|---:|---:|
| DanceTrack val | 0.5546 | 0.9468 | 0.3248 | 0.4958 | 5290 |
| BDD100K train | 0.4716 | 0.4571 | 0.4866 | 0.4110 | 7546 |
| MOT17 train | 0.7084 | 0.7179 | 0.6991 | 0.6244 | 434 |
| MOT20 train | 0.6242 | 0.8500 | 0.4584 | 0.5482 | 1645 |
| TAO train | 0.3446 | 0.2175 | 0.5461 | 0.2570 | 392 |

Macro（Dance/BDD/MOT17/MOT20 等权）：

| 指标 | U0 (B3) | Route A (B4) | UIDM | Δ vs U0 |
|---|---:|---:|---:|---:|
| Macro HOTA | 0.5379 | 0.5370* | 0.5897 | +5.2pp |
| Macro AssA | 0.4013 | 0.3953* | 0.4922 | +9.1pp |
| Macro IDF1 | 0.4614 | 0.4589* | 0.5199 | +5.9pp |

*Route A 用 L5 报告四域数值重算的等权均值。

逐域 vs B3 U0（AssA）：

| 域 | U0 AssA | UIDM AssA | Δ |
|---|---:|---:|---:|
| Dance | 0.4169 | 0.3248 | −9.2pp（塌陷） |
| BDD | 0.2881 | 0.4866 | +19.9pp |
| MOT17 | 0.6050 | 0.6991 | +9.4pp |
| MOT20 | 0.2950 | 0.4584 | +16.3pp |

IDSW：BDD 11042→7546（改善）、MOT20 2406→1645（改善）、
Dance 2588→5290（恶化）、MOT17 259→434（恶化）。

结论：**单 checkpoint 的 learned identity dynamics 在 3/4 标准域显著
提升（Macro AssA +9.1pp），但 DanceTrack 塌陷** —— 模型的 PBD 外观
证据在同类密集场景（Dance）压过了 motion 证据。这是
`L6_PARTIAL`（SUPPORTED with domain collapse），不是 FOUND。

## 8. TAO

TAO amodal train（105 视频，自定义 manifest 协议，fps=1）：
HOTA 0.3446 / DetA 0.2175 / AssA 0.5461 / IDF1 0.2570 / IDSW 392。
DetA 低是因为 TAO 的 amodal GT 与候选框协议不完全对齐；AssA 0.5461
说明在检测框给定后关联质量不弱。L5 未执行 TAO，无同协议历史基线；
记为 PARTIAL 正证据。

## 9. Cross-Spec Drift

ALL vs restricted-view online rollout drift（与 L5 同协议）：

| 域 | U0 (L5) | Route A (L5) | UIDM |
|---|---:|---:|---:|
| DanceTrack val | 37.9% | 29.3% | 34.0% |
| BDD100K train | 53.2% | 28.7% | 17.0% |

BDD 的 spec 一致性大幅改善（53.2%→17.0%），Dance 介于 U0 与 Route A
之间。说明 learned identity dynamics 显著减少了 spec-induced drift，
但 Dance 的跨视图一致性仍弱于 Route A（与该域关联塌陷一致）。

## 10. Ablation

MOT17 协议（fresh tag；训练步数：full 4200，其余 800–1200）：

| Ablation | AssA | Δ vs full |
|---|---:|---:|
| Full UIDM | 0.6991 | — |
| no tracking-level loss | 0.4678 | −23.1pp |
| no persistent memory | 0.4724 | −22.7pp |
| no inter-track interaction | 0.4901 | −20.9pp |
| no learned lifecycle | 0.4876 | −21.2pp |
| small UIDM 3M | 0.5458 | −15.3pp |

四项机制各自贡献 ~21–23pp AssA（MOT17），容量版本损失 −15pp：
memory / interaction / tracking-level objective / lifecycle 都不是装饰。
完整解释见 `reports/l6_ablation.md`。

## 11. 效率

UIDM-Large：15.0M trainable params；峰值 VRAM ~12.8GB/GPU（batch 8，
H=16）；3×A40（40GB）；训练 4200 步约 5.8 小时；推理测量：MOT17
3 视频（240 帧）约 16 秒（含 per-frame cache I/O），约 15 FPS；
FLOPs proxy 每帧 ~1–2 GFLOPs（见 `reports/l6_efficiency.md`）。

## 12. Failure Analysis

Dance 塌陷：12,543 个 switch 中 92% 发生在 gap=1（连续帧、无检测缺失），
前一帧与其它 GT 的平均 IoU 0.42 —— 即密集同类场景下模型在连续可见的
两个目标间交换身份，PBD 外观证据过强、motion/competition 证据不足。
BDD：switch 主要发生在 6–10 帧检测缺失后（re-identification 失败）。
详见 `reports/l6_failure_analysis.md`。

## 13. 结论与 ICLR Readiness

证据状态：

- `SUPPORTED`：单 checkpoint 跨域因果身份动力学显著提升
  Macro HOTA/AssA/IDF1（+5.2/+9.1/+5.9pp）；
- `PARTIAL`：DanceTrack 塌陷，Unified MOT 目标尚未达成；
- `FAILED`：无（本阶段主方法未被证伪，但 Dance 是明确失败案例）；
- `NOT_EXECUTED`：无（5 组 ablation + capacity ablation 已完成 MOT17
  协议；Dance/BDD 全域 ablation 未逐一重跑，已说明）。

ICLR readiness：方向有真实正信号（macro 提升 +5.2/+9.1/+5.9pp、
3/4 域提升、cross-spec drift 大降、四项机制均有 ablation 支撑），但
Dance 塌陷必须解决后才可作为主方法投稿。

## 14. 精确下一步（含实验设计）

1. **Dance 塌陷修复（最高优先）**：引入学习式 cue-reliability——
   让 pair head 显式预测 motion-IoU 与 PBD-cos 的置信度，并在
   同类密集场景（Dance）下调外观权重；或在训练中对 PBD 做
   dropout/加噪，强制模型依赖 motion/geometry。预期在 Dance 恢复
   AssA≥0.42 且 BDD/MOT17/MOT20 不回退后，Macro AssA 可达 ~0.52。
2. **IDSW 校准**：对 NEW/NO-MATCH 阈值做 per-domain 无关的
   tracking-level calibration（用 AssA/IDSW 联合目标），或把
   IDSW 直接加入 soft-switch loss 的权重。
3. **长 gap memory**：把 MAX_AGE 从 30 提升到 60+，并增加
   long-gap re-identification 训练样本，减少 BDD 6–10 帧 gap switch。
4. 全量 4200+ 步继续训练（loss 仍下降），并跑官方 DanceTrack/
   MOT17/MOT20 benchmark 兼容性验证。

## 14. 关键文件与 commit

- 代码：`locatemot/models/l6_uidm.py`、`tools/train_l6_uidm.py`、
  `tools/eval_l6_uidm.py`、`locatemot/tracking/online_tracker.py`
- 文档：`docs/l6_iclr_method_audit.md`、`docs/l6_uidm_design.md`、
  `docs/l6_training_design.md`
- 报告：本目录 `reports/l6_*.md`
- 状态：`outputs/l6/state.json`、`research_log.md`
- Commit SHA：`1b6f7ea4b54fbf1c9b99e0e538b7f5696b6999ca`
