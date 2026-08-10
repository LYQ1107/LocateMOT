# LocateMOT — Progress Status

更新时间：2026-08-10 21:30 (Asia/Shanghai)

## 当前任务/阶段

Stage L1-D 已完成：`L1_D_COMPLETE`（stage decision: `L1_D_PARTIAL`）。
主报告：`reports/STAGE_L1_D_FINAL_REPORT.md`。

## 已完成内容与关键结果

1. L1-D 官方代码审计：CAMELTrack / LG-Track / LLTrack
   （`docs/l1_d_reference_audit.md`，含 MOTIP/FDTA/OVTR/COVTrack/
   HATReID/HNCD-MOTR 复核）。
2. 结构决策：Evidence-Gated Set-Level Residual Association (EGRA)
   （`reports/l1_d_structure_decision.md`）。
3. 离线 base 模拟器验证：与 L1-C C3 校准结果完全一致
   （calib AssA 0.384 / IDSW 573）。
4. L1DK base 校准：0.4 IoU + 0.2 PBD + 0.4 Kalman-motion，thr 0.25；
   calibration AssA 0.4241 / IDSW 512。
5. EGRA 训练：0.49M 参数，8,360 步，~4.3 分钟（GPU 2）。
6. DanceTrack val AC：
   - L1DK base：HOTA 0.6280 / AssA 0.4165 / IDF1 0.5630 / IDSW 2,558；
   - L1DK_d03（残差）：AssA 0.3993 / IDF1 0.5503 / IDSW 2,579；
   - 残差在 calibration +1.9pp 但 val 不迁移。
7. 校正审计（val）：helpful 27,993 / harmful 3,187（precision 0.898、
   coverage 0.782、preservation 0.983；continuity 0.841→0.951），
   但官方 AssA/IDSW 不奖励。
8. 跨域（同一 checkpoint）：MOT20 强正（IDSW −35.5%）、MOT17 持平、
   BDD AssA 下降（IDSW −8.2%）；方向不一致 → 不部署 residual。
9. 结论：采用 L1DK base；EGRA residual 保留为消融。

## 仍在运行的进程

无（所有训练/评估已完成）。

## 尚未完成 / 下一步建议

- Stage L2：L1DK base 上统一 full-tracker + TAO-compatible cache +
  多类扩展。
- LODO 与 full tracker 未执行（pilot gate 未通过，按任务书不执行）。

## 需要人工确认的问题

无阻塞问题。

---

## Stage L2（2026-08-10）：Counterfactual Future-Utility Learning

状态：`L2_ORACLE_HEADROOM_LOW`（ICLR readiness NOT_READY）。

### 已完成

1. 四域 AC baseline 矩阵（DanceTrack/MOT17/MOT20/BDD）：
   L1DK base macro AssA 0.4062 最强；BEST_STRONG_BASE = L1DK base。
2. TrackEval AssA/IDF1 目标审计；windowed AssA 与官方 TrackEval
   整视频数值完全一致。
3. 2025–2026 官方代码审计（TDLP/SambaMOTR/TRACT/UniTrack/
   PathConsistency/QuoVadis/FDTA/HATReID-MOT/HNCD-MOTR）：
   NO DIRECTLY EQUIVALENT VERIFIED METHOD FOUND。
4. Counterfactual oracle（1,945 冲突事件）：
   - 单事件窗口 headroom：DanceTrack H32 +0.74pp、BDD H16 +1.01pp、
     MOT17 H32 +2.27pp、MOT20 H32 +1.76pp；
   - 端到端 greedy oracle：DanceTrack +0.02/+0.06pp、BDD −0.88pp、
     MOT17 −2.32pp；IDSW 全部变差。
5. Local-vs-future mismatch：DanceTrack H32 21.9%、BDD H16 61.7%、
   MOT17 70.8%、MOT20 75.0% 事件 future-best ≠ base。
6. 历史污染审计：EGRA 修正集中在已污染轨迹但 helpful 率低
   （DanceTrack 25.8%、BDD 9.1%）。
7. 结论：oracle headroom 不足 → 不训练大型 TUM，进入失败分析。

### 主报告

- `reports/STAGE_L2_FINAL_REPORT.md`
- `reports/STAGE_L2_GPT_HANDOFF.md`
- `reports/l2_oracle_headroom.md`、`reports/l2_failure_analysis.md`

### 下一步建议

- 若继续该方向：换效用定义（整序列 ID 映射 + IDSW 惩罚）或换协议
  （允许 ID 重映射的 full-tracker）后重新验证 oracle；
- 否则回到 L1DK base 的 full-tracker 工程路线；
- TAO cache 缺失，需补 cache 才能做 TAO 域。
