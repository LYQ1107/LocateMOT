# Stage L3 — Novelty Collision Audit（Gate 0）

日期：2026-08-10。

## 1. 三个 Claim 候选

- Claim-Candidate 1：一个 checkpoint 统一 heterogeneous box-MOT
  domains（DanceTrack/MOT17/MOT20/BDD multi-class/TAO）。
- Claim-Candidate 2：同一个 tracking core 统一 class-agnostic /
  category / open-vocab / referring / visual prompt object specification。
- Claim-Candidate 3：模型用 latent tracking regime 条件化 tracking
  computation，而不是 dataset-specific specialization。

## 2. 逐项碰撞结论

### Claim 1：统一 box-MOT domains

已核实：

- GLEE（CVPR 2024）在**检测/分割 foundation** 层面联合训练多数据集
  （含 BDD multi-class、TAO），是部分碰撞；
- SAM 3.1 在 promptable video 域（BURST/YTVIS/OVIS/SA-V）统一多目标
  跟踪，但不是标准 box-MOT 的检测集固定 AC 协议；
- MOTIP/TDLP/CAMELTrack 等均为单域或两域 trained association，无
  四域统一 checkpoint 的主结果。

结论：**在标准 box-MOT Association-Controlled 协议下，未发现
“一个 checkpoint 统一 DanceTrack/MOT17/MOT20/BDD multi-class 且
无 dataset-specific 参数”的已验证官方实现**。但必须把“统一”定义为
标准 MOT 协议，不能与 GLEE 的检测 foundation 混为一谈。

### Claim 2：统一 object specification 接口

已核实：

- SAM 3 / SAM 3.1：text/point/box/mask/exemplar 统一到图像+视频的
  检测-分割-跟踪，open-vocabulary；BURST HOTA 43.3（SAM3.1）；
- GLEE：text（CLIP）+ image/video + 多任务头，含 referring/grounding；
- OVTR/OVTrack：open-vocab MOT；
- STORM/QTrack：referring/query-driven MOT（2026）。

结论：**Claim 2 单独不成立（强碰撞）**。本项目不能把
“统一 prompt 接口”当核心 novelty；只能把 spec conditioning 当作
与 MOT identity/regime 正交的第二个轴，并在与 SAM3/GLEE 的
差异（标准 MOT 身份关联、AC 协议、regime 条件化）上建立边界。

### Claim 3：latent tracking regime 条件化

已核实：

- 无已验证官方实现把“从 prediction-side 状态估计 latent regime，
  并条件化共享 association core”用于标准 box-MOT；
- 最近邻：3D MOT 场景自适应阈值（2026）、AHAT 自适应混合关联、
  COVTrack 自适应多 cue 融合——都不是共享模型内部 latent regime
  因子分解，且无官方代码；
- MoE/conditional routing 在 SOT 多模态（ICML 2026 Dual MoE）存在，
  但无 MOT association 版本。

结论：**Claim 3 是三个 claim 中新颖性最强、唯一未见直接等价实现的
候选**。但必须由实验证明：(a) naive shared 有负迁移；(b) regime
条件化减轻负迁移；(c) 不是 dataset router。

## 3. Gate 0 判定

```text
NO_DIRECT_EQUIVALENT_VERIFIED_METHOD_FOUND
（对 Specification × Regime Factorization 在标准 box-MOT 的完整组合）
```

边界与风险：

1. Claim 2 有明确 collision（SAM3/GLEE/OVTR/QTrack/STORM），
   最终 claim 必须把“统一接口”降为第二个轴，不得作为独立 novelty；
2. 不得使用 first / foundation tracker / universal tracker 表述；
3. 若 pilot 无法证明 regime 条件化减轻负迁移，则回到
   `L3_REGIME_NOT_SUPPORTED`，不硬写 novelty。
