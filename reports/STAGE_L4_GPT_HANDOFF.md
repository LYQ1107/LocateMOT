# Stage L4 — GPT Handoff

日期：2026-08-11。项目：LocateMOT。

## 结论：FAIL（pilot gate）

```text
Problem Signal  : L4_SPEC_RESTRICTION_SIGNAL_SUPPORTED（真实存在）
Pilot           : L4_PILOT_GATE_FAIL
Mechanism       : L4_NOT_SUPPORTED（逐帧 consistency 不足）
ICLR readiness  : NOT_READY
```

## 1. 本阶段做了什么（真实数字）

1. **问题成立**：frozen U0 在 P0（Track-All-Then-Filter）与 P1
   （Pre-Filter）之间的 common-object identity drift：
   - BDD category 33–67%（car 32.9%、pedestrian 48.8%、truck 39.7%、
     bus 42.6%）；
   - DanceTrack person 32.2%、top-2 instance 31.1%；
   - TAO car 24.1%、instance 14.3%。
   P1 通常显著降低 IDSW（Dance instance IDSW 799→72，
   AssA 0.559→0.841）。
2. **文献审计**：`NO_DIRECTLY_EQUIVALENT_VERIFIED_METHOD_FOUND`
   （SAM3/GLEE/OVTR/OVTrack/STORM/QTrack/EPIPTrack/NOVA/V2-SAM/
   Path Consistency 等全部复核；`docs/l4_reference_audit.md`）。
3. **方法尝试**（U0 初始化，paired views，15,851 pairs，20 epochs）：
   - A2（spec 条件化，无一致性）：drift 全面变差；
   - A5（row/col KL + state cosine）：drift 基本不变或小幅变差
     （Dance inst 0.3112→0.3168；BDD car 0.3291→0.3262）；
   - A5p（partition co-assignment MSE，一次最小修正）：drift 仍变差
     （Dance inst 0.3314；BDD car 0.3398）。
   - 官方 TrackEval ALL 模式：A2/A5/A5p 与 U0 完全一致
     （Dance AssA 0.4169/IDF1 0.5694/IDSW 2588；MOT17 0.6050/0.5825/
     259；MOT20 0.2950/0.4012/2406；BDD 0.2881/0.2923/11042；
     macro AssA 0.4013）。
4. **TAO 恢复**：105 视频 cache 全部可读（`cache_key` 覆盖，
   不改共享数据）。

## 2. 为什么失败

- 身份漂移是**时间/轨迹级**现象：单帧 assignment 或单帧
  co-assignment 一致不代表跨帧 ID 迁移一致；
- A5 的 birth-GT 轨迹对齐在身份错误时会固化错误；A5p 的 partition
  loss 训练中只有 ~1e-4，基本无梯度；
- 训练（paired softmax）与评估（每视图独立 Hungarian + 生命周期）
  协议不匹配；
- 任务书只允许一次最小修正，已执行，不再调参堆容量。

## 3. 未执行（Gate 失败后按任务书不执行）

- 正式 multi-domain multi-spec full training；
- LODO / Leave-One-Spec-Type-Out；
- 真实 referring benchmark；
- box/point/visual prompt 接口；
- A3/A4 细分消融；
- 可视化案例（无图像渲染）。

## 4. 下一步唯一建议

不要继续在逐帧 paired consistency 上调参。若继续该方向，唯一值得
尝试的是 **trajectory-level 一致性**：把配对数据从帧对改成 clip，
用可微的 track-state 传播（或 PATH 式多路径一致）直接约束
「同一对象的跨帧身份轨迹在两种 spec 视图下等价」；若做不到，
则 Stage L4 科学主张只保留诊断部分（specification restriction
确实改变 identity），方法部分降级为 observation，不宣称 ICLR 创新。

## 5. 关键路径

- 最终报告：`reports/STAGE_L4_FINAL_REPORT.md`
- 审计数据：`outputs/l4/audit_{bdd,dance,tao}_{full,a2,a5,a5p}.json`
- 模型：`outputs/l4/checkpoints/{a2,a5,a5p}/final.pt`
- 配对数据：`outputs/l4/data/*.pkl`
- TAO manifest：`outputs/l4/manifests/tao_amodal_train_l4.jsonl`
- 研究日志：`research_log.md`
