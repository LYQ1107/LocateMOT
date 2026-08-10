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
