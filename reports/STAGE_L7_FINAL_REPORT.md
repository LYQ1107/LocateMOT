# Stage L7 Final Report：Specification-Conditioned Unified MOT

状态：`IN_PROGRESS`（本文件随实验推进持续更新，完成后自包含）。

## 1. Executive Summary

Stage L7 把 “Unified” 的定义从“一个 tracker 跑多个数据集”升级为：

> 不同 WHAT-TO-TRACK specification（ALL / open-vocabulary category /
> referring language）共享同一个 HOW-TO-TRACK 因果身份动力学核心。

本阶段已完成的科学动作：

1. 冻结 `LEARNED_CAUSAL_IDENTITY_DYNAMICS = SUPPORTED`（L6）。
2. 完成 2025/2026 OVMOT/RMOT 官方仓库深度审计与 novelty collision
   audit：`NO_DIRECT_EQUIVALENT_VERIFIED`；关键碰撞 COVTrack（ICCV25）
   已公开 association-embedding 级 adaptive multi-cue fusion，因此 cue
   reliability 不作为第一创新。
3. 在 UIDM identity-transition decoder 内实现 decision-level cue
   reliability（5 cue experts + 可靠性路由），做一次有边界的 Dance 修复。
4. 建立 TAO 官方 OVMOT 协议（官方 v1 GT + Detic public dets +
   官方 TETA Base/Novel/All），外观 token 由 PBD 换成 frozen CLIP。

最终结论与数字：见下文各节（实验进行中）。

## 2. L1–L6 evidence chain

- L1-B：universal identity embedding 失败。
- L1-C：固定 universal association decoder / UAF / grounding LoRA 失败。
- L1-D：固定 residual correction 跨域只有部分成立。
- L2：current-action future utility oracle headroom 很低，关闭 RL 主线。
- L3：latent regime router 学成 dataset shortcut，关闭 dataset MoE。
- L3-U0：shared learned one-checkpoint core 是正结果。
- L4：specification restriction 造成 identity drift；restricted evidence
  有时显著优于 ALL evidence。
- L5：0.49M/20 epoch frame-level consistency 失败；不能判定大容量
  temporal model 失败。
- L6：UIDM（persistent memory + set interaction + learned transition +
  lifecycle + NEW/NO-MATCH）四域 Macro HOTA 0.5897（+5.2pp）、
  AssA 0.4922（+9.1pp）、IDF1 0.5199（+5.9pp）；DanceTrack AssA
  0.3248 collapse；消融显示四项机制各贡献 ~21–23pp（MOT17 AssA）。

## 3. Stage L7 scientific hypothesis

不同 specification 改变 WHAT to track，同一 causal identity dynamics
掌管 HOW identities 被维护。三层验证：closed-set（跨 domain）→
OVMOT（跨 vocabulary）→ RMOT（跨 specification）。

## 4. Unified MOT definition

一个 shared checkpoint：Specification Encoder（WHAT）+ Shared UIDM
（HOW）+ Reliability-aware Identity Transition（证据异质性组件）。
ALL 本身也是一个 specification。不要求不同 spec 输出相同轨迹
（L4/L5 已证伪硬 consistency）。

## 5. 2025/26 literature audit

见 `docs/l7_reference_audit.md`（本节在最终版内联）。

## 6. official GitHub audit

OVTR(ICLR25, 500e72c1, MIT)、OVTrack(CVPR23, e188b32e, Apache-2.0)、
OVT-B(NeurIPS24, f033b314, Apache-2.0)、COVTrack(ICCV25, 9b0ced57,
Apache-2.0)、QTrack(26, bc746fe2, Apache-2.0)、TempRMOT(24, 6a65640d,
无 LICENSE)、STORM(26, 0d87c3ba, 无 LICENSE)、ReaMOT(25, 16951600,
MIT)、TETA(b498aa87, Apache-2.0)。全部已实际阅读 README 与关键模型/
评估代码，非摘要转述。

## 7. novelty collision

- 无公开方法同时满足 closed-set MOT + OVMOT + RMOT + one shared
  identity core + persistent learned identity dynamics +
  specification-conditioned selection。
- 结论：`NO_DIRECT_EQUIVALENT_VERIFIED`（不使用 FIRST）。
- COVTrack 已公开 adaptive multi-cue fusion（embedding 级门控+置信度），
  我们只把它当作身份转移解码器内的决策级组件并明确区分。
- QTrack/STORM 已覆盖 query-driven RMOT；我们的 novelty 在跨
  formulation 的 shared HOW core，而不是 language-conditioned tracking。

## 8. dataset inventory

见 `reports/l7_dataset_inventory.md`（最终版内联）。

## 9. ordinary MOT final status

REGRESSION_ONLY（Dance repair 评估后填最终数字）。

## 10. Dance repair

机制：decision-level cue experts + reliability router（见
`docs/implementation_evidence.md` 的 Reliability-aware Identity
Transition 节）。训练与四域结果见第 9/22 节（实验进行中）。

## 11. Specification Encoder

见 `docs/l7_specification_encoder_design.md`（最终版内联）。

## 12. Shared UIDM architecture

L6 UIDM-Large（d=384/6 层，~15M trainable）原样复用；L7 增加：
`app_dim` 参数化（PBD 2048 / CLIP 512）、cue experts + reliability
router（约 +0.5M）。identity dynamics 参数在所有 task 间共享。

## 13. cue reliability

decision-level mixture：`pair_logit = Σ_k softmax(rel_k)·score_k +
context_head(full evidence)`；router 上下文 = gap/age/competition/
memory 证据；辅助 soft-target CE。区别于 COVTrack embedding 门控。

## 14. OVMOT protocol

见 `docs/l7_ovmot_protocol.md`（最终版内联）。

## 15. OVMOT datasets

TAO 官方 val（v1 类别）：988 视频 / 36,375 帧 / 1,203 类（c 461 /
f 405 / r 337）。候选 = 官方 Detic public dets。C-TAO/OVT-B 按项目
约束不使用/不下载。

## 16. Base/Novel/All results

待实验（占位）。

## 17. OVMOT official metrics

TETA50（LocA / AssocA / ClsA），Base=non-r / Novel=r / All。

## 18. OVMOT baselines

OVTrack（CVPR23，同 public dets，apples-to-apples）；COVTrack（ICCV25）
与 OVTR（ICLR25）标 REFERENCE_ONLY（协议/检测器不同处如实注明）。

## 19. RMOT protocol

见 `docs/l7_rmot_protocol.md`（最终版内联）。

## 20. RMOT dataset

Refer-KITTI-V2 为候选；本地 expression/labels 存在，KITTI tracking
帧映射待核对（MFT25 目录为 SN/BT/MSK/PF 前缀序列，需要 seq 映射），
否则需官方下载（~5GB，磁盘允许）。STORM-Bench 缺 VidOR 帧，暂不执行。

## 21. RMOT results

待实验（占位）。

## 22. closed-set regression

待 Dance repair 训练后评估（占位）。

## 23. one-checkpoint verification

待 joint training（占位）。

## 24. cross-task transfer

设计：冻结 UIDM core，仅训练新语义前端（OVMOT/RMOT），比较
Frozen core vs joint fine-tune。待实验（占位）。

## 25. oracle interface diagnostic

GT target-membership selection + UIDM 分离 WHAT/HOW bottleneck；
待 OVMOT/RMOT 主结果后执行一次（占位）。

## 26. ablations

计划关键项：without persistent identity dynamics /
without spec encoder（硬过滤）/ task-specific vs shared core /
without cue reliability / without model-in-the-loop transition。
待实验（占位）。

## 27. parameter count

- UIDM-Large trainable：14.99M（L6）；+ cue reliability ≈ 15.55M；
  app_dim=512 时略降。
- frozen CLIP ViT-B/32：约 88M frozen（另行报告，不计入 trainable core）。

## 28. GPU/time

待最终统计（占位）。

## 29. efficiency

待最终统计（占位）。

## 30. failure cases

待失败分析（占位）。

## 31. what transfers

待实验结论（占位）。

## 32. what does not transfer

待实验结论（占位）。

## 33. claim boundary

- 不声称 FIRST；不用 adaptive multi-cue fusion 作第一创新。
- 不把 L6 内部 TAO AC AssA 当 OVMOT；OVMOT 一律用官方 TETA。
- 若只完成 OVMOT：`L7_OVMOT_SUPPORTED / RMOT_NOT_EXECUTED`；
  OVMOT+RMOT 均有正信号才考虑 `L7_UNIFIED_MOT_SIGNAL_SUPPORTED`。

## 34. scientific contribution

Specification-conditioned target selection + specification-shared causal
identity dynamics；跨 domain / vocabulary / specification 的共享 HOW
证据；reliability-aware identity transition 作为支撑组件。

## 35. ICLR readiness

待全部实验（占位；当前不可写 READY）。

## 36. next single recommendation

待实验（占位）。

## 37. important code/data paths

- 模型：`locatemot/models/l6_uidm.py`、`locatemot/tracking/online_tracker.py`
- 训练/评估：`tools/train_l6_uidm.py`、`tools/eval_l6_uidm.py`、
  `tools/build_l7_tao.py`、`tools/eval_l7_ovmot.py`、
  `tools/cache_l7_clip_closedset.py`
- 数据：`outputs/l7/data/tao_val`、`outputs/l6/data`、TAO 官方帧
  `/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal`
- 状态：`outputs/l7/state.json`；日志：`outputs/l7/logs/`

## 38. git commit SHA

待最终提交（占位）。
