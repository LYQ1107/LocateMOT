# Phase C：真实 G2 单主干耦合训练（step 17000）

## 结果

在冻结 GroundAnything/G2 step17000 主干的条件下，完成了 24 步真实 GPU 耦合更新。训练参数只有 G2 输入空间的身份记忆 token 投影、JEV 身份槽投影和结构化 typed readout，共 4,075,271 个参数；没有加载 OneJev 或 YesTrack，也没有第二个视觉模型。

Grounding causal teacher-forcing loss 从 `4.670032` 降到 `0.169164`。JEV 读出在这个单目标短序列上的 Yes/No 正确数从 `0/4` 到 `4/4`；MATCH、ADMIT、ACTIVE 在该单一目标合同下前后均为 `4/4`，不能据此宣称身份泛化。

训练后，C0（无记忆）在 5 帧的恢复框平均 IoU 为 `0.0000`；C2（上一帧记忆）平均 IoU 为 `0.99636`。第 86 帧只作为同一 MOT20-03 视频的时序留出，C2 IoU 为 `0.99270`，C0 为 `0.0000`。这说明记忆 token 已经在真实 G2 生成前路径上影响了框 token；由于训练预算、目标数和视频范围很小，结果属于耦合可训练性证据，不是独立 MOT 性能提升。

## 训练合同

- G2 adapter：step `17000`；base SHA-256 `3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2`。
- 数据：`MOT20-03` TRAIN 帧 `82–86`，目标 ID `27`；更新帧 `[83, 84, 85]`，时序留出 `86`。
- 视觉计算：`5` 帧、`5` 次 G2 视觉编码；各帧 C0/C2 共享该帧嵌入；独立 VLM 数量 `0`。
- 总损失：`1.0 * causal Grounding teacher-forcing + 1.0 * structured JEV`；JEV 包含动态 MATCH/NEW/NO_MATCH/REJECT、ADMIT、Yes/No、ACTIVE/LOST/REACTIVATE/TERMINATE 和 UPDATE/HOLD 读出。
- 记忆来源：上一帧真实 G2 image-token 整图摘要；当前帧标注只作为 teacher-forcing 目标和监督，不进入 memory。
- 代码在 prompt 的 `answer_start` 处切断；answer token 只作为 causal loss 目标，未进入 prefill 输入。

## 生成对照

| 帧 | 标注框（0–999） | C0 无记忆 | C2 训练后记忆 | C2 IoU |
|---:|---|---|---|---:|
| 82 | `[214, 563, 260, 700]` | `[['person', [1, 430, 38, 558]]]` | `无` | — |
| 83 | `[214, 561, 260, 698]` | `[['person', [1, 428, 40, 556]]]` | `[['person', [214, 561, 260, 698]]]` | 1.000000 |
| 84 | `[214, 560, 260, 697]` | `[['person', [1, 427, 42, 556]]]` | `[['person', [214, 560, 260, 697]]]` | 1.000000 |
| 85 | `[214, 557, 260, 695]` | `[['person', [2, 427, 43, 556]]]` | `[['person', [214, 557, 260, 694]]]` | 0.992754 |
| 86 | `[214, 556, 260, 692]` | `[['person', [2, 426, 44, 552]]]` | `[['person', [214, 556, 260, 693]]]` | 0.992701 |

## 边界与下一步

- 这不是完整 MOT/RMOT/OVMOT 测评；没有报告 HOTA、IDF1、IDSW 或 TETA。
- 训练和留出均在同一 MOT20-03 视频，只有一个身份，尚未覆盖全局一对一竞争、错误身份 memory、遮挡重现和跨视频泛化。
- G2 正式训练未被修改；运行使用独立 GPU7，训练 checkpoint SHA-256：`61a9645886ac1f1ceb51f30c2b82a89893edaafbff4587129f932c22ef819c68`。
- 后续先用多个训练视频训练同一耦合模块，再在独立 MOT/RMOT/OVMOT 视频上做 C0/C1/C2、固定候选和端到端对照。

