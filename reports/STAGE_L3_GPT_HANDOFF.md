# Stage L3 GPT Handoff

日期：2026-08-10。项目：LocateMOT。

## 一句话结论

Stage L3 完成审计与 pilot：**U0（naive shared learned）超 L1DK**
（macro AssA 0.4013 vs 0.3944），但 **U1（latent regime 条件化）
未过 Gate A**（macro 0.3915，仅 MOT20 微正），且 z_regime 呈
dataset shortcut（domain classifier 96.6%）。

```text
Stage Decision: L3_REGIME_NOT_SUPPORTED + REGIME_ROUTER_DATASET_SHORTCUT
ICLR readiness: NOT_READY
```

## 关键数字

### U0 / U1（四域 AC，fresh per-video 协议，官方 TrackEval）

| Domain | L1DK AssA | U0 AssA | U1 AssA | U0 IDSW | U1 IDSW |
|---|---:|---:|---:|---:|---:|
| DanceTrack val | 0.4165 | 0.4169 | 0.4050 | 2,588 | 2,528 |
| MOT17 | 0.5883 | 0.6050 | 0.5859 | 259 | 274 |
| MOT20 | 0.2778 | 0.2950 | 0.2958 | 2,406 | 2,436 |
| BDD（11 类 GT） | 0.2951 | 0.2881 | 0.2792 | 11,042 | 11,027 |

Macro AssA：L1DK 0.3944 / U0 0.4013 / U1 0.3915。

### Regime 诊断（H=16 windowed AssA，按 prediction-side 状态分桶）

- MOT17 motion=hi：C1 0.6400 vs L1DK 0.6296；
- MOT17 iou_amb=hi：C1 0.6519 vs L1DK 0.5853（+6.7pp）；
- BDD n_cand=hi：C0 0.5572 vs L1DK 0.5534；
- 信号弱到中等，样本少。

### Routing Shortcut

- domain classifier on z_regime：96.6%（随机 25%）；
- 域质心距离 0.76–3.64，域内 std ≈0.26；
- `REGIME_ROUTER_DATASET_SHORTCUT CONFIRMED`。

## 审计结论

- SAM3/SAM3.1（Meta 2026）统一 text/point/box/mask 检测-分割-跟踪
  （BURST HOTA 43.3）→ **Claim 2 强碰撞**；
- GLEE（CVPR 2024，MIT）多数据集联合训练含 BDD/TAO → Claim 1/2
  部分碰撞；
- OVTR/OVTrack/STORM/QTrack → open-vocab/referring MOT 碰撞；
- Claim 3（latent regime）未见直接等价官方实现；
- Gate 0：`NO_DIRECT_EQUIVALENT_VERIFIED_METHOD_FOUND`（完整组合），
  但 Claim 2 不能作为 novelty。

## 数据发现

- BDD manifest 已含 11 类 GT（`gt_categories`），多类可直接用；
- TAO cache 为旧布局（`train/{BDD,AVA,...}`），与 manifest 不匹配，
  延迟处理；
- STORM-Bench/RMOT26 基准已 clone，数据未下载。

## 未完成 / 下一步建议

1. 若继续：先做 **spec-conditioned U0（B 轴，category/compat 输入，
   无 latent regime）**，验证统一对象指定；
2. 若 U0-spec 有效，再用去共线 regime 特征（scene 级代理而非
   benchmark 统计）重试 U1；
3. 否则以 U0 为统一 checkpoint 收口（full tracker + LODO 基线）；
4. TAO 需 cache 修复后纳入 open-world 证据。

## 产物

- 主报告（自包含，附录 A–M）：`reports/STAGE_L3_FINAL_REPORT.md`
- 审计：`docs/l3_reference_audit.md`、`reports/l3_novelty_collision_audit.md`
- 数据：`docs/l3_dataset_audit.md`、`docs/l3_protocol_matrix.md`
- 诊断：`reports/l3_regime_signal.md`、`reports/l3_negative_transfer_audit.md`
- Pilot：`reports/l3_u0_shared_baseline.md`、`reports/l3_u1_conditional_pilot.md`
- Shortcut：`reports/l3_shortcut_audit.md`
- 失败分析：`reports/l3_failure_analysis.md`
- 代码：`locatemot/models/l3_unified.py`、`tools/train_l3.py`、
  `tools/eval_l3.py`、`tools/l3_regime_diagnostics.py`、
  `tools/analyze_l3_routing.py`
