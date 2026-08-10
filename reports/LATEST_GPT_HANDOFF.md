# Latest GPT Handoff — LocateMOT

更新：2026-08-10。最近阶段：Stage L1-D，状态 `L1_D_PARTIAL`。

一句话：L1DK base（Kalman IoU+PBD 融合，无训练参数）是当前最强统一
关联方法（DanceTrack val AC AssA 0.4165 / IDF1 0.563 / IDSW 2,558）；
EGRA set-level 残差模型 calibration 提升但 val 不迁移、跨域方向不一致，
不部署。

完整自包含摘要：`reports/STAGE_L1_D_GPT_HANDOFF.md`；
主报告：`reports/STAGE_L1_D_FINAL_REPORT.md`。

补充（事实修正）：LoRA PBD 提取本身成功
（`LORA_PBD_EXTRACTION_SUPPORTED`：Frozen equivalence PASS、8,024 帧
LoRA cache 完成）；失败的是 LoRA 训练后的 PBD 表示质量
（`LORA_PBD_DEGRADED`），数字不变。

---

更新：2026-08-10。最近阶段：Stage L2，状态 `L2_ORACLE_HEADROOM_LOW`。

一句话：Stage L2 证明 local correctness 与 future trajectory utility
确实不同构（DanceTrack H32 21.9% 冲突事件 future-best ≠ base），但
privileged counterfactual oracle 的端到端整视频 AssA headroom
< 0.1pp（DanceTrack），BDD/MOT17 为负且 IDSW 变差，按任务书停止
大型 TUM 训练并进入失败分析。

完整自包含摘要：`reports/STAGE_L2_GPT_HANDOFF.md`；
主报告：`reports/STAGE_L2_FINAL_REPORT.md`；
失败分析：`reports/l2_failure_analysis.md`。
