# Stage L1-C Grounding Forgetting Audit

评估单位：DanceTrack calibration 同帧。

| 指标 | Frozen | LoRA | 变化 |
|---|---:|---:|---:|
| Recall@0.5 | 0.9765 | 0.8032 | −17.3pp |
| candidate/frame | 8.69 | 8.53 | −0.16 |
| duplicate rate/frame | 0.00 | 0.10 | +0.10 |
| small-object recall | ~0.000 | ~0.000 | ≈0（DanceTrack 极少小目标） |

## 结论

- LoRA 适配后 grounding 明显下降（Recall@0.5 −17.3pp），candidate 重复
  增加；
- 结论：`LORA_TRACKING_GAIN_WITH_FORGETTING` 中的遗忘方向成立；
  无任何“LoRA 提升 tracking”证据。
