# Stage L4 — Efficiency Notes

日期：2026-08-10。

## 1. 模型

- L4SpecEqAssociator：0.488M 可训练参数（U0 core 0.49M + type-level
  spec embedding/projection ≈ 2K）。
- 无大 VLM / 无新增 prompt encoder；spec 只作为 token 级有界条件。

## 2. 训练

- 15,851 paired samples（BDD 7,645 / Dance 8,006 / MOT17 180 / MOT20 20）；
- batch 64，20 epochs，U0 初始化，1 GPU；
- 实测 ≈ 3.4 min/epoch（2 views forward/backward），20 epochs ≈ 58 min
  （A2/A5 各 1 GPU）。

## 3. 推理审计（frozen U0，同一脚本）

| Domain | 规模 | 耗时 |
|---|---:|---:|
| BDD（200 视频 × 12 specs） | 2,400 tracker runs | ~160 s |
| DanceTrack val（25 视频 × 3 specs） | 75 runs | ~9 min |
| TAO（105 视频 × 6 specs） | 630 runs | ~35 s |

## 4. 与 post-filter 对比

- P0（Track-All-Then-Filter）：每帧必须处理全部候选，之后丢弃非 spec
  轨迹；
- P1 / L4：每帧只处理 spec 候选（候选/token 数下降），同时 L4 训练
  一致性以保持身份稳定；
- 本项目 AC 协议固定候选，token 节省不是主 claim；只在报告记录。
