# Stage L1-C Final Report

## 1. Executive Summary

Stage L1-C 完成了 Unified Association Decoder（UAF）的完整训练与评估、
传统基线语义审计与校准、LocateAnything LoRA 官方训练链路打通、Frozen vs
LoRA PBD 诊断、IoU/PBD ambiguity 与 root-cause 分析。核心结论：

1. **UAF pilot gate 不通过**：50k 步 + NEW 修复 + margin 校准后，
   DanceTrack val AssA=0.133 / IDF1=0.270 / IDSW=26,804，低于旧 B6
   （0.155/0.308/16,456）。
2. **raw PBD 的 candidate-selection 很强但 full-video ID 连续性差**：
   R@1 0.922 / mAP 0.944，但 raw-PBD AssA 仅 0.155；失败集中在
   set-level 分配与 ID 断裂，而非判别力。
3. **PBD 不互补于 IoU**：DanceTrack 上 IoU 选对时 PBD 几乎也选对，
   IoU 选错时 PBD 也错（pbd_only=25/225k）。
4. **失败模式可预测**：PBD 选择失败与 ID 连续性失败 AUROC 0.93/0.91
   （logistic probe，calibration 训练 / val 评估）→ L1-D
   evidence-driven correction 有依据。
5. **LoRA 300 步 grounding 适配有害**：PBD R@1 −48.7pp、mAP −36.4pp、
   Recall@0.5 −17.3pp；LoRA raw-PBD AC AssA 0.042 vs Frozen 0.140。
6. **L1-D 结构方向**：保留强 base affinity（IoU/PBD/motion 融合），
   学习器只做证据门控的 set-level residual correction，不做 from-scratch
   assignment；主线保持 Frozen LocateAnything。

## 2. 项目与目标

PROJECT=LocateMOT，TASK=Unified MOT（DanceTrack/MOT17/MOT20/BDD/TAO）。
本阶段验证“冻结 LocateAnything 上可训练 contextual association”假设
（H1）与“LoRA 适配是否提升 association”（H3）。

## 3. L1-A 结论（回顾）

全视频 trajectory 路线失败：T6 HOTA 0.449 / AssA 0.227 / IDSW 5,962；
association 与 detection 必须拆开评估 → 引入 association-controlled
protocol。

## 4. L1-B 结论（回顾）

Universal Identity Adapter 失败：LODO 两方向相对 raw PBD 退化
（BDD −15.9pp / TAO −7.5pp），adapter 学到 dataset shortcut。

## 5. L1-C UAF 负结果

- 训练：50k 步，seed 20260806，batch 8，lr 1e-4 cosine，7.9M 参数；
  clip=8 帧，K=8 历史。
- 修复：NEW CE 权重 0.2；每帧 unmatched 候选封顶 8；NEW-margin 在
  calibration 校准（最优 3.5）。
- 最终 DanceTrack val（AC）：HOTA 0.355 / DetA 0.947 / AssA 0.133 /
  LocA 0.850 / MOTA 0.787 / IDF1 0.270 / IDSW 26,804。
- 对比旧 B6：AssA 0.155 / IDF1 0.308 / IDSW 16,456 → UAF 三项均不达标。

## 6. Baseline Semantic Mapping Audit

见 `reports/l1_c_baseline_mapping_audit.md`。关键：AssA≈0.419 是 Motion
（C1），不是 raw PBD；raw PBD（C2）只有 0.155。

校准后的 DanceTrack val 基线：

| method | HOTA | DetA | AssA | IDF1 | IDSW |
|---|---:|---:|---:|---:|---:|
| IoU | 0.608 | 0.947 | 0.390 | 0.529 | 3,554 |
| Motion | 0.630 | 0.947 | 0.419 | 0.566 | 2,916 |
| Raw PBD | 0.384 | 0.947 | 0.155 | 0.319 | 15,616 |
| IoU+PBD (0.7/0.3/0.3) | 0.610 | 0.947 | 0.393 | 0.537 | 2,981 |
| B6 | 0.383 | 0.948 | 0.155 | 0.308 | 16,456 |
| UAF | 0.355 | 0.947 | 0.133 | 0.270 | 26,804 |

## 7. Why Raw PBD Matters

raw PBD 的 candidate-selection 很强（R@1 0.922/mAP 0.944，PBD margin≥0.01
时选择正确率 0.91–0.97），但作为独立 full-video tracker 很弱（AssA 0.155）。
这说明 PBD 是强 appearance 证据，需要正确的 assignment 机制才能变现。

## 8. LocateAnything LoRA 工程审计

根因：①PEFT unwrap 死循环（`base_model` 自引用）；②LoRA 训练 JSONL 用了
字面量 `(x,y,x,y)` 而非官方 `<box><x1><y1><x2><y2></box>` PBD token 格式。
修复后：Frozen 等价性 PASS（box IoU 0.998、hidden cosine 1.0、重复运行
一致）；LoRA 提取 8,024 帧全部完成、feature finite。
详见 `reports/l1_c_lora_pbd_extraction.md`。

## 9. Frozen vs LoRA PBD

| 指标 | Frozen | LoRA |
|---|---:|---:|
| R@1 | 0.922 | 0.435 |
| mAP | 0.944 | 0.580 |
| Recall@0.5 | 0.977 | 0.803 |
| AC AssA (calibration) | 0.140 | 0.042 |

分类：`LORA_PBD_DEGRADED` + grounding 遗忘。Frozen 主线保留。

## 10. IoU / PBD Ambiguity

96.5% 事件 IoU margin≥0.10（easy），所有方法 acc 0.86–0.96；歧义区
（margin<0.05）仅 1.7%，所有方法 acc 0.40–0.58。PBD margin≥0.01 时选择
正确率 ≥0.91。

## 11. Cue Disagreement

both_correct 89.4%、iou_only 10.6%、pbd_only 0.01%、both_wrong 0.01%。
PBD 不互补于 IoU；问题在 set-level 分配。

## 12. Learnability Probe

PBD 选择失败 AUROC 0.933 / PR-AUC 0.632；raw-PBD ID 连续性失败 AUROC
0.915 / PR-AUC 0.575（calibration 训练，val 评估）。失败模式可预测。

## 13. Stage Decision

- UAF：`L1_C_ASSOCIATION_NOT_SUPPORTED`（相对 B6 未达标）。
- LoRA：`LORA_PBD_DEGRADED`（不采用为主线）。
- L1-D：**有依据进入**，方向 = 强 base affinity + 证据门控 set-level
  residual correction（Frozen 基座）。

## 14. Next Recommended Stage

L1-D：Evidence-Driven Unified Association（保留 PBD/IoU 强先验，学习器
只修正低可靠性 assignment），先做 targeted GitHub audit，再实现/训练/
评估。

## 15. Important Paths

- 协议/审计：`docs/l1_c_reference_audit.md`、
  `docs/l1_c_locateanything_lora_audit.md`、
  `docs/official_code_modifications.md`
- 数据：`outputs/l1_c/fixed_candidate_manifest/`、
  `outputs/l1_c/cache_lora/`（LoRA PBD cache）
- 模型：`outputs/l1_c/checkpoints/uaf/final.pt`、
  `outputs/l1_c/checkpoints/lora/checkpoint-300`
- 结果：`outputs/l1_c/association_controlled_main.csv`、
  `outputs/l1_c/cue_events.csv`、`outputs/l1_c/frozen_vs_lora_pbd.json`、
  `outputs/l1_c/learnability_probe.json`
