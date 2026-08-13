# Stage L7 Unified MOT Design（工作版）

## 科学主张

> Tracking formulations differ primarily in target specification (WHAT),
> while identity maintenance can be modeled by one shared causal identity
> dynamics process (HOW).

三层验证：

- Closed-set MOT：跨 domain（BDD/Dance/MOT17/MOT20）共享 UIDM；
- OVMOT：跨 object vocabulary（TAO Base/Novel）共享同一 UIDM；
- RMOT：跨 target specification（referring language）共享同一 UIDM。

## 主方法三件套（避免模块拼盘）

1. **Specification Encoder（WHAT）**：`spec = ALL | category text |
   referring text` → spec embedding → candidate relevance / selection。
   首版 OVMOT 用 frozen CLIP ViT-B/32 text/image embedding：
   relevance = cosine(candidate crop embedding, spec text embedding)。
   ALL 是显式 specification（track all candidates）。
2. **Shared Causal Identity Dynamics（HOW）**：L6 UIDM 原样复用：
   持久 per-track memory + anchor、set-of-tracks interaction、
   identity transition（continue/NEW/NO-MATCH）、learned lifecycle、
   model-in-the-loop。参数跨 formulation 共享。
3. **Reliability-aware Identity Transition**：decision-level cue experts
   （motion/geometry/appearance/competition/memory）+ 局部可靠性路由，
   位于 transition decoder 内，不是 dataset router；与 COVTrack 的
   association-embedding 门控融合明确区分（见 novelty audit）。

## 关键接口

- 外观 token 抽象：closed-set 用 LocateAnything PBD（2048-d），
  OVMOT/RMOT 用 frozen CLIP crop embedding（512-d）；
  `UIDM(app_dim=2048|512)` 只换前端投影器，其余核心参数完全共享。
- 候选格式：boxes + gen + app token + semantic label；
  association 输出（track id）与 perception 输出（Detic/CLIP label）分离，
  便于按官方 TETA 分别计 LocA/AssocA/ClsA。
- 推理因果在线：逐帧 model-in-the-loop，无未来信息、无全局批处理。

## OVMOT 首版实验设计（先回答转移问题，不追 SOTA）

1. 在 closed-set 数据上只训练新的 CLIP 外观投影器（冻结 L6/repair 后
   的 UIDM core）→ 校准 front-end 与核心的分布兼容。
2. TAO val 官方协议评估：官方 v1 GT、官方 Detic public dets、
   官方 TETA（Base=LVIS f/c、Novel=LVIS r、All）。
3. 正信号判据：Novel AssocA 非随机（明显 > 0）、Base/Novel 均工作、
   普通四域 regression 不崩。
4. 正信号后：joint closed-set + OVMOT 训练得到 one unified checkpoint。

## 与已冻结负结果的边界

- 不重开：universal ReID、dataset router、future utility、
  retrospective revision、tiny residual correction、fixed cue fusion、
  specification hard consistency（L4/L5）。
- 不把 C-TAO / `.MOTSynth.partial` 数据用于训练评估。

