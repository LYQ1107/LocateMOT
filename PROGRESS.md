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

---

## Stage L4（2026-08-10，进行中）：Specification-Equivariant Unified MOT

状态：`L4_COMPLETE`（Pilot Gate FAIL，`L4_NOT_SUPPORTED`，ICLR NOT_READY）。

### 已完成

1. 2025–2026 文献/官方代码审计（`docs/l4_reference_audit.md`、
   `reports/l4_novelty_collision_audit.md`）：
   `NO_DIRECTLY_EQUIVALENT_VERIFIED_METHOD_FOUND`。
2. U0 restriction audit（P0 vs P1，`reports/l4_u0_restriction_audit.md`）：
   - BDD category drift 33–67%；DanceTrack person 32% / instance 31%；
     TAO car 24% / instance 14%；
   - P1 通常显著降低 IDSW（DanceTrack instance IDSW 799→72，
     AssA 0.559→0.841）。
3. Consistency metric audit（`docs/l4_consistency_metric_audit.md`）：
   最优 ID 映射后的 co-identity agreement + per-GT drift + TrackEval。
4. TAO cache 恢复：`cache_key` 覆盖（`docs/l4_tao_cache_recovery_plan.md`），
   105 视频全部可读。
5. Paired-view 数据（15,851 pairs）与训练：
   - A2（spec 条件化，无一致性）与 A5（+ assignment/state consistency）
     与 A5p（partition co-assignment，一次最小修正）各 20 epochs，
     1 GPU，U0 初始化。

### Pilot 结果（Gate FAIL）

- Cross-spec drift（Dance inst）：U0 0.3112 → A2 0.3272 / A5 0.3168 /
  A5p 0.3314；BDD car：U0 0.3291 → A2 0.3502 / A5 0.3262 /
  A5p 0.3398；
- 官方 TrackEval ALL：A2/A5/A5p 与 U0 完全一致（macro AssA 0.4013）；
- 失败根因：身份漂移是时间/轨迹级现象，单帧 consistency 不足。

### 后台进程

无（全部完成）。

### 下一步

- 唯一建议：trajectory-level consistency（clip 级可微 track
  propagation / path consistency）；
- 最终报告：`reports/STAGE_L4_FINAL_REPORT.md`（自包含，含 19 个附录）。
