# Stage L1-D GPT Handoff

日期：2026-08-10。项目：LocateMOT（不是 TrackOCD/OCD_OVMOT）。
状态：`L1_D_PARTIAL`。主报告：
`reports/STAGE_L1_D_FINAL_REPORT.md`（本文件是给网页版 GPT 的自包含摘要）。

## 1. 一句话结论

证据驱动的 Kalman IoU+PBD 融合基座（L1DK）是当前最强统一关联方法
（DanceTrack val AC AssA 0.4165 / IDF1 0.563 / IDSW 2,558）；在其上
训练的 set-level 残差模型（EGRA，0.49M 参数）calibration 提升但
val 不迁移，跨域方向不一致，不部署。

## 2. 关键数字

### DanceTrack val AC（TrackEval official，DetA≈0.947 全部一致）

| method | AssA | IDF1 | IDSW |
|---|---:|---:|---:|
| C0 IoU | 0.3899 | 0.5291 | 3,554 |
| C1 Motion | 0.4193 | 0.5660 | 2,916 |
| C2 raw PBD | 0.1555 | 0.3188 | 15,616 |
| C3 IoU+PBD | 0.3934 | 0.5367 | 2,981 |
| C4 B6 | 0.1546 | 0.3083 | 16,456 |
| UA（失败 UAF） | 0.1329 | 0.2704 | 26,804 |
| **L1DK base** | **0.4165** | **0.5630** | **2,558** |
| L1DK_d03 | 0.3993 | 0.5503 | 2,579 |

### 跨域（同一 checkpoint；BDD/MOT17/MOT20 为训练域方向检查）

| 域 | method | AssA | IDF1 | IDSW |
|---|---|---:|---:|---:|
| BDD100K | base / L1D | 0.3292 / 0.2841 | 0.3167 / 0.2889 | 12,149 / 11,151 |
| MOT17 | base / L1D | 0.6010 / 0.5922 | 0.5784 / 0.5775 | 276 / 274 |
| MOT20 | base / L1D | 0.2779 / 0.2864 | 0.3232 / 0.3916 | 3,736 / 2,408 |

Macro：AssA −1.6pp、IDF1 +0.7pp、IDSW relative −10.9%（方向不一致）。

### 校正审计（DanceTrack val，225,071 事件，帧间连续性定义）

helpful 27,993 / harmful 3,187（precision 0.898）、coverage 0.782、
preservation 0.983；continuity acc 0.841→0.951。但官方 AssA/IDSW
不奖励该改善。

## 3. 方法

- L1DK base：`0.4*IoU(last) + 0.2*PBD_cos + 0.4*IoU(Kalman_pred)`，
  thr=0.25（仅 calibration 校准）；匈牙利 + 共享 NEW 规则。
- EGRA：pair 特征 19 维 + 2 层 set transformer + track 级 gate +
  `A_final = base + r*0.3*tanh(delta)`；loss = row/col CE +
  reliability BCE + 保留正则；训练数据 = base 真实在线状态
  （非 GT 完美历史）+ GT 监督；13,405 帧 / 88,465 事件；
  40 epochs / 8,360 步 / 0.49M 参数 / 单卡 ~4.3 分钟。
- 训练数据：DanceTrack calibration + BDD + MOT17 + MOT20。

## 4. 为什么不部署 residual

1. calibration 上 AssA +1.9pp，DanceTrack val −1.7pp（过拟合）；
2. 跨域方向不一致（MOT20 强正，BDD/MOT17 负）；
3. base 身份正确率仅 44.5%（历史 swap），~50% 错误行需 delta>0.6，
   超出有界残差能力；
4. 帧间连续性改善与 TrackEval AssA/IDSW 语义不对齐
   （断裂大多计入 Frag）。

## 5. 已采用 / 不采用

- 采用：L1DK base（无训练参数、统一、跨域稳定）；Frozen PBD；
  AC 协议；共享 NEW/lifecycle。
- 不采用：LoRA（`LORA_PBD_DEGRADED`）；UAF/Universal Identity
  Adapter；EGRA residual 作为统一部署。

## 6. 下一步建议

Stage L2：L1DK base 上做统一 full-tracker + TAO-compatible cache +
多类扩展；residual 仅保留为 crowd-domain 消融。

## 7. 重要路径

- 模型：`locatemot/models/l1d_association.py`
- 训练/数据：`tools/train_l1d.py`、`tools/build_l1d_dataset.py`
- Checkpoint：`outputs/l1_d/checkpoints/l1d_k/final.pt`
- 报告：`reports/STAGE_L1_D_FINAL_REPORT.md`
- 审计：`docs/l1_d_reference_audit.md`、
  `docs/l1_d_architecture_evidence.md`
