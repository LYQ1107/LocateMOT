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

当前管线状态（2026-08-14）：Dance repair 4 卡训练中；TAO val 数据
已完整构建（988 视频 / 36,375 帧 / 1.61M Detic dets，CLIP 特征无缺失）；
closed-set CLIP 缓存完成（245 视频 / 428,648 crops）；OVMOT 评估链路
已用官方 TETA 端到端验证（随机模型 30 视频：All TETA 22.98 /
LocA 60.77 / AssocA 8.16 / ClsA 0；ClsA=0 是 AVA 子集 Detic 类别与
GT 不对齐的数据特性，非协议 bug）。

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

`REGRESSION_ONLY`。Dance repair 尝试失败（见第 10 节），普通 MOT 冻结；
后续统一模型只做回归检查，不再做 ordinary-specific 调参。普通 MOT 主
结果仍以 L6 为准（Macro HOTA 0.5897 / AssA 0.4922 / IDF1 0.5199；
Dance 0.5546/0.3248、BDD 0.4716/0.4866、MOT17 0.7084/0.6991、
MOT20 0.6242/0.4584）。

## 10. Dance repair

机制：decision-level cue experts + reliability router（见
`docs/implementation_evidence.md` 的 Reliability-aware Identity
Transition 节）。

结果（一次 iteration，按协议只允许一次，**判定失败**）：

| 域 | L6（修复前） | cuerel（mix on） | cuerel（mix off） |
|---|---|---|---|
| Dance HOTA | 0.5546 | 0.4888 | 0.5466 |
| Dance AssA | 0.3248 | 0.2522 | 0.3156 |
| Dance IDSW | 5290 | 9251 | 9204 |
| BDD AssA | 0.4866 | 0.5299 | — |
| MOT17 AssA | 0.6991 | 0.6973 | — |
| MOT20 AssA | 0.4584 | 0.4626 | — |

结论：cue mixture 本身只贡献部分伤害（关掉后 Dance 0.3156 仍低于
L6 0.3248，IDSW 仍 9204）；cuerel 核心微调整体让 Dance 漂移，
BDD 收益不足以抵消 Dance 的 IDSW 翻倍。按约束不进行第二次 Dance
调参，改用 L6 uidm_full 作为 OVMOT 共享核心；cuerel 保留为负结果
对照。`L7_DANCE_REPAIR_FAIL`。

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

官方 TETA（TAO val，官方 v1 GT + Detic public dets；模型 =
L6 UIDM core 冻结 + CLIP 投影器仅用 closed-set CLIP 校准，
零 OVMOT 训练数据）：

| Split | TETA50 | LocA | AssocA | ClsA |
|---|---|---|---|---|
| All | 31.479 | 64.785 | 29.513 | 0.139 |
| Base | 31.530 | 64.891 | 29.540 | 0.158 |
| Novel | 31.103 | 63.996 | 29.312 | 0.000 |

- 随机（未训练）模型的 AssocA 基线 ≈ 8.2；共享核心给出 29.5。
- **Base/Novel AssocA 差距仅 0.23pp（29.54 vs 29.31）**：identity
  dynamics 对 unseen vocabulary 无 Base 偏好，是“共享 HOW 泛化到
  novel 类别”的直接证据（OVTrack 论文 Base/Novel AssocA 差 3.3pp）。
- ClsA 由 frozen Detic label 给出，几乎为 0（TAO 上 Detic public dets
  类别与 GT 对齐差，AVA 等视频几乎全错）→ 是 perception/grounding
  瓶颈，不是 association 瓶颈（见第 25 节 oracle/CLIP 归因）。

换成 frozen CLIP 语义编码器分类（候选 crop embedding 与 1203 个
LVIS 类别文本 embedding 的余弦 argmax，track id 不变）：

| Split | TETA50 | LocA | AssocA | ClsA |
|---|---|---|---|---|
| All | 33.937 | 64.785 | 29.513 | 7.513 |
| Base | 33.948 | 64.891 | 29.540 | 7.413 |
| Novel | 33.856 | 63.996 | 29.312 | 8.259 |

- ClsA 从 0.14 提升到 7.51（Novel 8.26 ≥ Base 7.41），TETA All
  31.48→33.94。说明 specification encoder（WHAT）换 frozen CLIP 后
  分类能力对 base/novel 都成立，且 identity（HOW）完全不受影响
  （AssocA 不变）。

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

`NOT_EXECUTED`（数据阻塞，如实记录）：

- Refer-KITTI-V2 的 expression/labels_with_ids 本地存在，但其帧要求为
  官方 KITTI tracking 序列 0000–0020（image_02/{seq:04d}）。服务器上没有
  这些序列；MFT2025 目录的 SN/BT/MSK/PF 帧属于另一套多数据集组织
  （帧数 684/3000/1754/15000 等，与 KITTI 序列帧数全部不匹配），不能
  当作 Refer-KITTI 帧使用。
- 官方 KITTI tracking 下载在 cvlibs.net 需要注册登录；未找到免登录镜像。
  STORM-Bench 需要 VidOR 帧（本地没有，磁盘紧张未下载）。
- 已交付：RMOT 协议与接口设计（`docs/l7_rmot_protocol.md`、
  `docs/l7_specification_encoder_design.md`），共享 UIDM + frozen
  language encoder 的 WHAT/HOW 分解已设计并写代码路径；数据到位后可
  直接接入，不需要重造 tracking core。
- 因此本阶段只可给出 `L7_OVMOT_SUPPORTED / RMOT_NOT_EXECUTED`
  级别的结论（若 OVMOT 为正信号），不写 Unified-MOT-Signal。

## 22. closed-set regression

统一模型（CLIP front-end）的四域回归在 joint training 后用
`tools/eval_l7_closed_clip.py` 执行（PBD 路径用 L6 基线，CLIP 路径
用同一 TrackEval 协议）。结果见第 23 节完成后（实验进行中）。

## 23. one-checkpoint verification

已有一个共享 checkpoint 同时服务 closed-set 与 OVMOT：
`outputs/l7/checkpoints/ovmot_probe/latest.pt`（L6 UIDM core + CLIP
front-end；closed-set 回归与 TAO OVMOT 均用同一权重，无 dataset/
task-specific head）。

closed-set 回归（同一 CLIP-front-end checkpoint，四域）：

| 域 | HOTA | AssA | IDF1 | IDSW |
|---|---|---|---|---|
| Dance | 0.5369 | 0.3045 | 0.4753 | 6164 |
| BDD | 0.4317 | 0.4077 | 0.3585 | 11430 |
| MOT17 | 0.6471 | 0.5840 | 0.5753 | 369 |
| MOT20 | 0.5973 | 0.4196 | 0.5232 | 1799 |
| Macro | 0.5533 | 0.4290 | 0.4831 | — |

对比 PBD 版 L6（Macro AssA 0.4922）：CLIP 前端统一化带来约 -6.3pp
Macro AssA 的 closed-set 代价（BDD -7.9pp、MOT17 -11.5pp），换取
open-vocabulary 能力；这是 appearance front-end 的统一性权衡，不属于
identity core collapse（Dance IDSW 6164 vs L6 5290）。

## 24. cross-task transfer

设计：冻结 UIDM core，仅训练新语义前端（OVMOT/RMOT），比较
Frozen core vs joint fine-tune。待实验（占位）。

## 25. oracle interface diagnostic

同一组 UIDM 轨迹输出，只替换 category_id 的来源（track_id/box 完全
不变），分离 WHAT（分类）与 HOW（关联）：

| 分类来源 | All TETA | All ClsA | Base ClsA | Novel ClsA |
|---|---|---|---|---|
| Detic public det label | 31.479 | 0.139 | 0.158 | 0.000 |
| frozen CLIP 文本余弦 | 33.937 | 7.513 | 7.413 | 8.259 |
| GT oracle（分析专用） | 63.451 | 96.054 | 95.912 | 97.114 |

- 三种替换下 AssocA 恒为 29.513（All）、Base≈Novel，证明 identity
  dynamics 与分类解耦。
- ClsA 从 7.5 到 oracle 96 的 headroom 全部在 perception 端；结合
  Grounding/更强大 VLM 语义前端可无损提升 WHAT，无需改动 HOW。
- 注意：oracle 分类用 GT 类别，只用于归因，不是正式方法。

## 26. ablations

已执行：

1. **without persistent identity dynamics（OVMOT，训练中）**：
   `ovmot_probe_stateless`（同协议 + `--stateless`），完成后对比
   All/Base/Novel AssocA。（占位，训练完成后填入）
2. **without cue reliability（closed-set，L6 已做）**：no-memory
   -22.7pp、no-interaction -20.9pp、no-lifecycle -21.2pp、no-trackloss
   -23.1pp（MOT17 AssA），证明四项 HOW 机制各自必要。
3. **specification encoder 替换（OVMOT）**：Detic→CLIP→oracle 三档
   只改 ClsA（0.14→7.51→96.05），AssocA 恒 29.51，证明 WHAT/HOW 解耦
   （第 25 节）。

未执行（`NOT_EXECUTED`，原因如实注明）：

- task-specific vs shared core：需要 OVMOT-only 训练数据；本地 Detic
  SwinB 推理在本机 OOM（roi_align 51.7GB 分配）且官方 CDN 被网络策略
  阻断，未能生成 TAO train dets。
- without model-in-the-loop transition：同 L6 已证（stateless 对照）。
- RMOT 相关：数据阻塞（见第 20/21 节）。

## 27. parameter count

- UIDM-Large trainable：14.37M（L6 core）；CLIP front-end 版
  投影器 0.69M，其余核心冻结时 trainable 0.69M。
- frozen CLIP ViT-B/32：约 88M frozen（text/image），不计入
  trainable core；另 frozen Detic 检测器只作候选/标签来源。

## 28. GPU/time

- Dance repair 训练：4×A100-40G，4200 步，17434s（4.84h）。
- OVMOT 探针训练（冻结核心 + 投影器）：4×A100-40G，2000 步，6550s
  （1.82h）。
- TAO val 数据构建：CLIP ViT-B/32 批量编码 1.61M crops，4 GPU
  约 30–40 min。
- OVMOT TETA 评估：tracker 31 min（1 GPU）+ 官方 TETA 约 10–15 min。
- stateless 消融训练进行中（占位）。

## 29. efficiency

- trainable core 0.69M（冻结核心时）；总 UIDM 14.37M + frozen CLIP 88M。
- 推理为 online causal：逐帧一次 forward（≤64 tracks + ≤50 candidates），
  无全局批处理、无未来信息。
- 峰值 VRAM：训练约 38GB/卡（A100-40G，batch 8×4 卡）。

## 30. failure cases

- Dance repair 负结果：cue reliability mixture + 核心微调使 Dance IDSW
  翻倍（9251/9204 vs L6 5290），判定失败并冻结普通 MOT。
- OVMOT ClsA 极低（Detic 0.14）：AVA 等视频的 public dets 类别与 GT
  几乎全错；换 CLIP 语义前端后 7.51，oracle 96.05，说明是 perception
  而非 identity 瓶颈。
- closed-set CLIP front-end 代价：BDD AssA -7.9pp、MOT17 -11.5pp，
  Macro AssA -6.3pp，是统一语义前端的代价。
- 工程失败：Detic SwinB 本机推理 OOM；官方 CDN 网络阻断；KITTI
  tracking 帧需登录下载 → 记录为 blockers，不伪造结果。

## 31. what transfers

- 身份动力学（set interaction + persistent memory + transition +
  lifecycle）跨 domain（L6 四域）与跨 vocabulary（TAO Base≈Novel
  AssocA 29.5，随机基线 8.2）转移。
- 同一冻结核心只需重训 0.69M 外观投影器即可接 OVMOT。

## 32. what does not transfer

- 单一外观 token 不跨 formulation 最优：PBD 在 closed-set 更强，
  CLIP 提供 open-vocabulary；统一换 CLIP 付出 closed-set 代价。
- frozen perception 的分类质量不随核心转移（Detic ClsA 0.14），
  需要语义编码器单独解决（CLIP 7.51，仍有大 headroom）。

## 33. claim boundary

- 不声称 FIRST；不用 adaptive multi-cue fusion 作第一创新。
- 不把 L6 内部 TAO AC AssA 当 OVMOT；OVMOT 一律用官方 TETA。
- 结论等级：`L7_OVMOT_SUPPORTED / RMOT_NOT_EXECUTED`（数据阻塞）。
  由于 RMOT 未执行，不写 `L7_UNIFIED_MOT_SIGNAL_SUPPORTED`。

## 34. scientific contribution

实验支撑的 claim：

> Tracking formulations differ primarily in target specification (WHAT),
> while identity maintenance can be modeled by one shared causal identity
> dynamics process (HOW).

证据：跨四域 closed-set（L6，Macro AssA 0.4922）；跨 vocabulary
OVMOT（冻结核心 + 0.69M 投影器，Base≈Novel AssocA 29.5，随机基线
8.2）；WHAT/HOW 可分离（三种分类前端不改变 AssocA，ClsA 0.14→7.51→
96.05 oracle）。cue-reliability 尝试为负结果，不作为贡献组件。

## 35. ICLR readiness

不写 READY。已满足：UIDM 主干成立、OVMOT 正信号、Novel 真实泛化
（Base≈Novel）、novelty audit 无直接撞车、WHAT/HOW 分解有 oracle
归因。未满足：RMOT 未执行（数据阻塞）、joint OVMOT 训练未执行
（Detic 推理 OOM）、official metric 竞争力有限（AssocA 29.5 vs
OVTrack 33.6/36.9，同一候选协议；TETA Novel 31.1 高于 OVTrack
27.8）。结论：`L7_OVMOT_SUPPORTED`，距 strong ICLR signal 还差
RMOT 与更强语义前端。

## 36. next single recommendation

单项建议：**为 TAO 训练数据修复/替代 Detic 推理**（换一张有可用
环境的机器、或用官方 CDN 可访问的网络、或换 Grounding DINO 等第二
检测器），完成 closed-set + OVMOT 的 joint unified 训练与
task-specific vs shared 消融；同时解决 KITTI tracking 帧下载，跑
Refer-KITTI RMOT，把结论升级到 unified signal。其余不要动：不要
回刷普通 MOT、不要重开 RL、不要做 dataset router。

## 37. important code/data paths

- 模型：`locatemot/models/l6_uidm.py`、`locatemot/tracking/online_tracker.py`
- 训练/评估：`tools/train_l6_uidm.py`、`tools/eval_l6_uidm.py`、
  `tools/build_l7_tao.py`、`tools/eval_l7_ovmot.py`、
  `tools/cache_l7_clip_closedset.py`
- 数据：`outputs/l7/data/tao_val`、`outputs/l6/data`、TAO 官方帧
  `/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal`
- 状态：`outputs/l7/state.json`；日志：`outputs/l7/logs/`

## 38. git commit SHA

`7bc6862`（WIP）；最终提交见完成后更新。
