# Stage L1-D LODO Report

日期：2026-08-10。

## 结论

**未执行。** 任务书 §55 明确 LODO（leave-domain-out / zero-finetune
generalization）仅在 L1-D pilot 成功后执行。DanceTrack val AC 上
EGRA residual（AssA 0.3993）低于其 base（L1DK，AssA 0.4165），
pilot gate 未通过，因此不执行 LODO 训练与评估。

## 若后续执行

- Leave-DanceTrack-out：base 权重须在 BDD/MOT17/MOT20 上重新校准
  （避免 DanceTrack 泄漏到 base 校准），模型仅用 BDD/MOT17/MOT20
  训练，DanceTrack train（34,046 帧缓存可用）作为评估集。
- Leave-BDD-out：DanceTrack calibration + MOT17/MOT20 训练，
  BDD100K 全部 200 视频评估（稀疏 5fps，custom MOTChallenge 格式
  已由 `tools/run_l1d_trackeval.py` 支持）。

