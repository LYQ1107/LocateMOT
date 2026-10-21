# Phase C：双序列 G2 耦合训练审计（step 48）

## 结果

从 MOT20-03 和 MOT20-05 各取 5 帧，最后一帧时序留出，使用同一冻结 G2 step17000 主干和同一组 4,075,271 个耦合参数。step12、step24、step48 都使用相同的 6 个训练 transition；step24、step48 从前一 checkpoint 恢复 optimizer 和耦合模块。

teacher-forced causal Grounding loss 从 step1 的 `4.6592` 降到 step12 的 `4.6592`、step24 的 `1.5149`、step48 的 `1.1808`；JEV 离散读出在 step12 已达到两个序列的 MATCH/ADMIT/Yes-No/ACTIVE `1.0`。但真实 greedy C2 框在两个序列的平均 IoU 到 step48 仍为 `0.0000`，C0 也为 `0.0000`。部分 C2 输出出现合法结构但坐标仍接近冻结 G2 基线，部分出现前缀/解析失败。

结论是：JEV typed readout 的离散监督可以学习，当前单 token memory 到 G2 坐标生成的优化还没有打通。不能把 loss 下降或 JEV 选择准确率写成 Grounding/MOT 提升；下一步先做 coordinate-token 级诊断和加权目标，再决定是否改接口。

## 运行合同

- G2 adapter：step `17000`；base SHA-256 `3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2`。
- 序列：MOT20-03 帧 82–86（目标 27）和 MOT20-05 帧 13–17（目标 17）；每个序列最后一帧留出。
- 视觉计算：10 帧、10 次 G2 视觉编码；独立 VLM 数量 0；G2 backbone 更新 0。
- 总损失：`1.0 * causal Grounding teacher-forcing + 1.0 * structured JEV`；answer token 位于 prompt `answer_start` 之后，仅作为 teacher-forcing 目标。
- step48 coupled checkpoint SHA-256：`3b68aee5fce1f9489f1908271da64db0318e2e1f027de912e9b585bee8273902`。

## 轨迹和实际输出

| 总 step | 平均 teacher-forced Grounding loss | JEV（两个序列） | C2 平均 IoU MOT20-03 | C2 平均 IoU MOT20-05 |
|---:|---:|---|---:|---:|
| 12 | 4.6592 | MATCH/ADMIT/Yes-No/ACTIVE = 1.0 | 0.0000 | 0.0000 |
| 24 | 1.5149 | MATCH/ADMIT/Yes-No/ACTIVE = 1.0 | 0.0000 | 0.0000 |
| 48 | 1.1808 | MATCH/ADMIT/Yes-No/ACTIVE = 1.0 | 0.0000 | 0.0000 |

## 保留边界

- 这是双序列耦合训练失败/未完成证据，不是完整 MOT/RMOT/OVMOT 测评。
- 不报告 HOTA、IDF1、IDSW、TETA 或跨域泛化收益。
- 下一步为 coordinate-token 级 loss/概率诊断、wrong-memory 与多身份候选竞争；在根因修复前不再增加无界训练步数。
- 运行时 receipt：`/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/phase_c_real_coupled_multisequence_step17000_v3/receipt.json`。

