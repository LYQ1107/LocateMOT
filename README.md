# LocateMOT

LocateMOT — LocateAnything-based Persistent Multi-Object Tracking.

本仓库是与原 GLEE-PMOT 完全隔离的新研究项目。原项目
`/data1/LWR/vranlee/SERVER_ONLY/avis/GLEE-PMOT` 仅作为只读基线，禁止导入其代码
或在其目录内开发。

## 当前阶段

Stage L0 — LocateAnything Tracking Prototype：

1. 独立复现官方 LocateAnything-3B 并验证图片定位/密集定位；
2. 找到 PBD box 与 hidden states 的对应关系，提取 Object Token；
3. 构建两帧配对数据（reference box / reference crop visual prompt）；
4. 训练 Persistent Track Decoder，完成两帧一对一身份关联与 NO_MATCH；
5. 官方 visual-prompt LoRA 适配一次；
6. 比较 IoU / region cosine / PBD token / Pairwise MLP / Track Decoder / LoRA，
   给出是否进入完整统一 MOT 阶段的报告。

本阶段不执行：长视频 rollout、automatic birth、完整生命周期、mask decoder、
point tracking、正式文本/RMOT/OV-MOT 实验、DanceTrack/MOT17 完整训练、
GRPO/RL、多 Seed、SOTA benchmark。

## 目录

- `third_party/Eagle`：官方 NVlabs/Eagle（LocateAnything 所在仓库），固定 commit
- `references/`：检索并固定的官方参考实现，用于审计与设计依据
- `locatemot/`：新项目核心代码
- `docs/`：架构、许可证、参考仓库清单、实现证据
- `configs/`：阶段配置与数据 manifest
- `outputs/`、`reports/`：实验输出与报告（不提交权重和大缓存）

## 环境与输出

- 独立 conda 环境：`/home/lwr/anaconda3/envs/locatemot`
- 输出根目录：`/data3/testdata/vranlee/LocateMOT`
- 模型/缓存：`/data3/testdata/vranlee/LocateMOT/{models,huggingface,cache}`

详细规格见 `docs/` 与 `configs/stage_l0.yaml`。
