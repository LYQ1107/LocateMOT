# Stage L3 — 2025/2026 官方实现审计

日期：2026-08-10。
原则：只记录实际 clone + 阅读的官方代码；commit 以仓库 HEAD 为准；
不把 README 摘要当实现依据。

## 0. 审计方法

对每个仓库：

- `git remote get-url origin` 验证官方 URL；
- `git rev-parse HEAD` 记录 commit；
- 阅读 README / LICENSE / 模型定义 / 训练 loss / 数据加载 / 推理与
  tracker 状态 / prompt 编码 / association / query update。

## 1. SAM 3 / SAM 3.1（Meta，2026）

- 官方仓库：`github.com/facebookresearch/sam3` ✅
- Commit：`96914d24`；License：SAM License（2025-11-19，需单独确认条款）
- 已读文件：
  - `README.md`、`RELEASE_SAM3p1.md`
  - `sam3/model/video_tracking_multiplex.py`（Object Multiplex 联合多目标跟踪）
  - `sam3/model/video_tracking_multiplex_demo.py`（推理状态管理）
  - `sam3/model/sam3_multiplex_tracking.py`、`sam3_multiplex_video_predictor.py`
  - `sam3/model/memory.py`（mask memory）、`multiplex_utils.py`（MultiplexState）
- Input interface：text phrase / exemplar / point / box / mask；
  open-vocabulary 概念（SA-CO 270K concepts）。
- Output interface：masks + boxes + object pointers；video 下检测-分割-跟踪。
- Supports MOT identity？是（多目标、ID 持久、BURST HOTA 43.3/44.5）。
- Supports multi-object？是（SAM 3.1 Object Multiplex，~7x 加速）。
- 如何关联：SAM2 式 memory propagation + 检测器-跟踪器关联启发式
  （“detection-tracker association and other heuristics”）；对象 pointer
  cross-attention；非重叠 mask 约束。
- Conditional computation：无 latent regime；固定 tracking core。
- 多域训练：SA-CO/VEval/SA-V/YT-Temporal/BURST/YTVIS/OVIS/MOSE 等
  promptable 视频域，非标准 box-MOT 域。
- 复用性：prompt encoder / spec 表示可作为 B 的 foundation 参考；
  MOT 身份关联在标准 AC 协议上不覆盖本项目四域。
- 碰撞：**Claim 2（统一 prompt 接口）强烈碰撞**；Claim 1/3 不直接覆盖。

## 2. GLEE（CVPR 2024 Highlight，ByteDance）

- 官方仓库：`github.com/FoundationVision/GLEE` ✅
- Commit：`f36a49e8`；License：MIT
- 已读文件：
  - `projects/GLEE/glee/GLEE.py`（多数据集 category 表：COCO/LVIS/Obj365/
    OpenImage/BDD/TAO/YTVIS/OVIS/LVVIS/BURST/UVO/SA1B/grounding/RVOS）
  - `projects/GLEE/glee/models/glee_model.py`（tracking contrastive loss
    IDOL-style、匈牙利匹配、query 结构）
  - `glee/data/datasets/bdd100k.py`、`tao.py`、`uni_video_image_mapper.py`
- Input interface：image/video + text（CLIP variants）+ 可扩展任务头
  （det/inst-seg/video-inst-seg/RVOS/grounding）。
- Output interface：boxes/masks + track embeddings。
- Supports MOT identity？是（track embedding + contrastive loss +
  query 匹配；TAO MOT 榜）。
- 如何关联：每帧 query 检测 + `pred_track_embed` 跨帧对比损失 +
  Hungarian；无 latent regime。
- 多域训练：真实多数据集联合训练（含 BDD multi-class 与 TAO）。
- 复用性：联合训练配方、text 接口、BDD/TAO 数据接入。
- 碰撞：**Claim 1（多域统一）与 Claim 2（spec 接口）均部分碰撞**；
  但 GLEE 是检测/分割 foundation，不是标准 box-MOT 的 association
  条件化核心，且无 regime 条件化。

## 3. OVTR（ICLR 2025）

- 官方仓库：`github.com/jinyanglii/OVTR` ✅
- Commit：`500e72c1`；License：MIT
- 已读文件：`ovtr/models/ovtr.py`、`utils.py`（`protect_track_preds`、
  `text_query` 处理、miss_tolerance/obj_idxes）、`transformer.py`、
  `updater.py`、`matcher.py`
- Input：text category embeddings + detection/track queries。
- Output：open-vocab 类别 + 跟踪框；TAO TETA。
- 关联：MOTR 式 query propagation（detection queries + track queries，
  obj_idxes/disappear_time，track update）。
- 训练目标：DETR set loss + 文本类别匹配（CIP 类别传播）。
- Conditional computation：无 regime。
- 碰撞：Claim 2 open-vocab MOT 方向。

## 4. OVTrack（CVPR 2023）

- 官方仓库：`github.com/SysCV/ovtrack` ✅
- Commit：`e188b32e`；License：Apache-2.0
- 已读文件：README（VLM distillation + 数据幻觉；TETA 评估协议）
- Input：检测 + VLM 类别查询；Output：open-vocab boxes + IDs。
- 关联：两阶段（检测→关联），无学习型 regime。
- 碰撞：Claim 2 open-vocab 历史方法。

## 5. Grounded SAM 2（IDEA-Research）

- 官方仓库：`github.com/IDEA-Research/grounded-sam-2` ✅
- Commit：`b7a9c29f`；License：Apache-2.0
- 已读文件：README（Grounding DINO/DINO-X/Florence-2 + SAM2 pipeline）
- 本质：**pipeline，不是单一模型**；无联合训练；MOT 身份仅靠
  SAM2 propagation。

## 6. SAM2MOT（AAAI 2026，Huawei Cloud）

- 官方仓库：`github.com/TripleJoy/SAM2MOT` ✅（仅 README）
- Commit：`7bdae12c`；License：Apache-2.0
- **代码未发布**（README：Incoming）。只能 paper-guided，
  不得声称复用官方实现。

## 7. STORM-Bench / STORM（2026，Amazon）

- 官方仓库：`github.com/amazon-science/storm-referring-multi-object-grounding` ✅
- Commit：`0d87c3ba`
- 内容：STORM-Bench（VidOR 派生，90,617 帧、29,933 tracks、
  30,700 expressions，1fps）；STORM 模型（end-to-end MLLM，
  Task-Composition Learning；HOTA 66.7 / IDF1 78.3）**模型代码不在该仓库**。
- Input：referring expression；Output：multi-object trajectories。
- 碰撞：Claim 2 referring MOT 方向（基准已发布，模型代码未见）。

## 8. QTrack / RMOT26（2026，MIT License）

- 官方仓库：`github.com/gaash-lab/QTrack` ✅
- Commit：`bc746fe2`；License：MIT
- 已读文件：README（query-driven MOT；RMOT26 基准；TPA-PO 策略优化；
  RMOT26 0.30 MCP / 0.75 MOTP）
- Input：reference frame + 文本 query；Output：只跟踪 query 指定的目标。
- 关联：VLM 端到端推理 + 时间一致性；无 latent regime 条件化。
- 碰撞：Claim 2 referring/query-driven MOT 方向。

## 9. AnyTrack（2026）

- 官方仓库：`github.com/IdolLab/AnyTrack` ✅
- Commit：`7d5ca454`
- 内容：SOT 的多模态（RGB-T/D/E）统一；非 MOT；作 B 的模态参考。

## 10. 无已验证官方代码的 2026 方法（记录，不当作依据）

- GOVTrack（CVPR 2026，generative OVMOT）：未找到官方 repo。
- COVTrack（ICCV 2025）/ COVTrack++（2026）：adaptive multi-cue
  fusion OVMOT；官方 repo 未见发布（“code will be available”）。
- STORM（ICML 2026，6D tracking）：单目标 6D，与 MOT 无关。

## 11. Conditional computation / MoE（结构参考）

- Unified Multimodal Visual Tracking with Dual MoE（ICML 2026）：
  SOT 多模态 T-MoE/M-MoE；无 MOT association。
- AHAT（2026）：adaptive hybrid association（复杂运动场景）——
  动态特征融合，非 latent regime。
- 3D MOT scene-adaptive learned thresholds（2026）：按 density/motion
  自动调阈值——与 regime 思想接近，但仅 3D 阈值，非共享条件化计算。
- 结论：condition-aware routing 在视觉任务有结构先例，但
  **没有在标准 box-MOT association 上做 latent regime 条件化的
  已验证官方实现**。

## 12. 总表

| Method | Year/Venue | Official repo verified | Commit | License | MOT identity | 多目标 | Text prompt | Visual prompt | Open-vocab | Dataset-specific params | Multi-domain train | Conditional computation | Regime adaptation | Novelty collision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SAM 3/3.1 | 2026 Meta | ✅ | 96914d24 | SAM License | ✅ | ✅ | ✅ | ✅ | ✅ | 否 | 是（promptable 视频） | 否 | 否 | Claim 2 强 |
| GLEE | CVPR 2024 | ✅ | f36a49e8 | MIT | ✅ | ✅ | ✅ | 部分 | ✅ | 否 | 是（含 BDD/TAO） | 否 | 否 | Claim 1/2 部分 |
| OVTR | ICLR 2025 | ✅ | 500e72c1 | MIT | ✅ | ✅ | ✅ | 否 | ✅ | 否 | 否（TAO 训练） | 否 | 否 | Claim 2 |
| OVTrack | CVPR 2023 | ✅ | e188b32e | Apache-2.0 | ✅ | ✅ | ✅ | 否 | ✅ | 否 | 否 | 否 | 否 | Claim 2 |
| Grounded SAM 2 | 2024/25 | ✅ | b7a9c29f | Apache-2.0 | ✅ | ✅ | ✅ | ✅ | ✅ | 否 | 否（pipeline） | 否 | 否 | Claim 2 |
| SAM2MOT | AAAI 2026 | ✅（代码未发布） | 7bdae12c | Apache-2.0 | ✅ | ✅ | 否 | ✅ | 否 | 否 | 否 | 否 | 否 | 无 |
| STORM | 2026 | ✅（bench 仓库） | 0d87c3ba | — | ✅ | ✅ | ✅ | 否 | 部分 | 否 | 是 | 否 | 否 | Claim 2 referring |
| QTrack | 2026 | ✅ | bc746fe2 | MIT | ✅ | ✅ | ✅ | 否 | 部分 | 否 | 否 | 否 | 否 | Claim 2 referring |
| AnyTrack | 2026 | ✅ | 7d5ca454 | — | SOT | 否 | 部分 | 部分 | 否 | 否 | 否 | MoE | 否 | B 模态参考 |
| COVTrack/++ | ICCV25/26 | ❌ 无代码 | — | — | ✅ | ✅ | ✅ | 否 | ✅ | 否 | 否 | adaptive fusion | 部分 | Claim 2/3 部分 |
| GOVTrack | CVPR 2026 | ❌ 无代码 | — | — | ✅ | ✅ | ✅ | 否 | ✅ | 否 | 否 | 否 | 否 | Claim 2 |

## 13. 审计结论

1. **Claim 2（统一 prompt 接口）已被 SAM3/GLEE 强碰撞**：text/point/box/
   mask prompt 统一本身不是 novelty；
2. **Claim 1（多域统一）被 GLEE 部分碰撞**：GLEE 联合训练 BDD/TAO 等，
   但不是在标准 box-MOT AC 协议下的 association 核心统一；
3. **Claim 3（latent regime 条件化 tracking computation）未发现
   已验证官方等价实现**：最接近的是 3D 场景自适应阈值 / 自适应融合，
   但都不是共享模型内部的 latent regime 条件化。
