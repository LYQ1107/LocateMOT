# Stage L1-D Structure Decision

日期：2026-08-10。状态：L1_D_STRUCTURE_DECISION。

## 1. 决策结论

采用 **Evidence-Gated Set-Level Residual Association (EGRA)**：

1. Base affinity = 校准后的 IoU + PBD + 常速度 motion 线性融合
   （强先验，保持 raw PBD/IoU 的判别力）；
2. 轻量 set-level Transformer（candidate/track 全集合交互，
   GAFFE 式）输出**有界残差**与 **track 级可靠性门控**；
3. 训练目标 = assignment ranking（row+col CE），**不设 NEW 类**；
4. 推理 = Hungarian + 共享阈值，NEW/出生由统一 shell 决定；
5. 主线 Frozen PBD；LoRA 只做消融（L1-C 已判 LORA_PBD_DEGRADED）。

## 2. 为什么不是其它结构

| 候选结构 | 否决/选择理由（基于证据） |
|---|---|
| from-scratch UAF（K+1 CE） | 已实测失败：AssA 0.133 < 全部传统基线；破坏 easy 区先验（0.856 vs 0.960） |
| 只调 pairwise reliability gate 选 PBD/IoU | pbd_only 只有 25 例，PBD 不互补 IoU；主失败是 set-level ID 连续性 |
| 纯规则再加权（C3 扩展） | C3 已接近 IoU 上限；learnability probe 证明失败可预测（AUC 0.93），值得可学习修正 |
| CAMELTrack from-scratch embedding | 与 UAF 同类风险：没有保留强先验的机制；仅吸收其 InfoNCE、GAFFE 集合交互、阈值 NEW |
| 大容量 Transformer（>50M）/RL | 失败机制未定位到容量不足；禁止失败后堆容量 |

## 3. 成功标准（与任务书一致）

- 主：DanceTrack val AC 上相对 best simple base（当前预期 C3 0.393 或
  motion base）AssA 提升或 IDSW 明显下降；且 PBD 保留率高、
  helpful corrections > harmful corrections。
- 强信号参考：AssA +2pp 或 IDSW −15%（不调 val）。
- 统一性：BDD held-out、MOT17/MOT20 方向一致，不出现
  “DanceTrack +5pp / BDD −10pp”。
- LODO：训练未见过该 domain 时，机制不弱于 raw PBD 或保持正增益。

## 4. 范围控制

- 不做 long-term memory / reactivation（L1-A 已证 reactivation 失败）。
- 不做 LoRA 主线优化（已判 degraded）。
- 不做多 GPU 数据并行之外的工程扩展。
- 不在 val 上调任何阈值/权重/超参。

