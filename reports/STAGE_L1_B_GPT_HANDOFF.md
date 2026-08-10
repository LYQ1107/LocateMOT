# Stage L1-B GPT Handoff

## 阶段结论

`L1_B_IDENTITY_SIGNAL_NOT_SUPPORTED`（pilot gate 未通过）

## 关键事实

- Raw 基线（Same-Cat R@1）：R0 PBD-box-end 最强（dancetrack 0.919 /
  mot17 0.946 / mot20 0.869 / tao 0.870 / ytvos 0.750 / mose 0.551）。
- R4 IdentityToken v1：full 全部不优于 raw；pbd-only 仅 DanceTrack +4.9pp。
- v2（加入 BDD100K，1,064 身份）：full 在 DanceTrack +6.5pp / BDD +3.0pp /
  TAO +8.7pp；MOT17 -3.6 / MOT20 -4.7 / YT-VOS -1.3 / MOSE -1.7；
  macro +1.0pp，gate 仍未通过。
- 未跑：association-controlled、LODO、full training（gate 前置条件未满足）。

## 产物

- 报告：reports/STAGE_L1_B_FINAL_REPORT.md
- 审计：docs/l1_b_reference_audit.md、docs/l1_b_dataset_identity_audit.md、
  docs/l1_b_unified_schema.md
- 数据：outputs/l1_b/（statistics/cache/checkpoints/retrieval CSVs）
- 代码：tools/l1_b_dataset_audit.py、build_l1b_pilot_split.py、
  cache_l1b_locateanything.py、eval_l1b_retrieval.py、
  train_l1b_identity_adapter.py、repair_l1b_cache_gt.py；
  locatemot/models/identity/

## 下一步建议

先单独验证 multi-class road 方向（BDD/TAO 全量 + 严格跨数据集协议），
同时重新设计 deformable 训练目标；若 multi-class 全量成立再扩展。
