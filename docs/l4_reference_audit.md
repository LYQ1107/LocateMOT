# Stage L4 — 2025/2026 Specification/Identity-Consistency 官方实现审计

日期：2026-08-10（Asia/Shanghai）。
原则：只记录实际 clone + 阅读的官方代码，或明确标注「paper-only / 未公开
代码 / 未 clone」。commit 以 `git rev-parse HEAD` 为准；不根据摘要或
博客转述实现细节。

## 0. 审计问题

Stage L4 要回答的核心问题：

1. 是否存在已公开方法明确研究「same video + different specification →
   same persistent identity」；
2. 是否存在 MOT 方法要求 `Track(Restrict(X)) ≈ Restrict(Track(X))`
   （候选子集限制下的身份等价）；
3. 是否只是 SAM3/GLEE 式「多种 prompt 输入」的重述；
4. 是否只是 Track-All-Then-Filter；
5. 是否只是 Path Consistency 的 prompt 版本。

## 1. 2026 新检索与 clone

### 1.1 NOVA（IROS 2026，3D open-vocabulary MOT）

- 论文：NOVA: Next-step Open-Vocabulary Autoregression for 3D Multi-Object
  Tracking in Autonomous Driving（arXiv 2603.06254）。
- 官方仓库：`github.com/xifen523/NOVA`
- Commit：main `1bd3ff18`；**release/v0.1.0 `4358a627`（含完整代码）**
- License：Apache-2.0（LICENSE + NOTICE）。
- 已读文件：
  - `README.md`（release 分支：hybrid prompting、base/novel、
    `p_yes`、Hungarian、lifecycle）；
  - `nova/models/association_model.py`（geometry 注入 + causal LM +
    yes/no logprob → association cost）；
  - `nova/data/class_split.py`（canonical Base/Novel、alias、
    `AS_UNKNOWN` / `REJECT` 策略、authoritative class map 门）；
  - `nova/data/sample_builder.py`、`nova/preprocessing/common.py`
    （`build_semantic_prompt`、prompt policy）。
- 与 L4 关系：open-vocab class 用 base 名 / “Unknown” 的 hybrid prompt
  处理 Novel；但**没有**「同一视频、不同候选子集、身份应一致」的
  限制等价目标；3D 任务、无标准 2D box-MOT AC 协议。可借鉴
  class-split 的严谨性（禁止自造 taxonomy），不采用其 LM 关联核心。

### 1.2 V²-SAM（CVPR 2026，cross-view object correspondence）

- 论文：Marrying SAM2 with Multi-Prompt Experts for Cross-View Object
  Correspondence（arXiv 2511.20886）。
- 官方仓库：`github.com/jaychempan/V2-SAM`
- Commit：`31c3babf`；License：README 标注 MIT，**仓库根目录未见 LICENSE
  文件**（复制代码需作者确认）。
- 已读文件：
  - `projects/v2sam_visual/models/vp_matcher.py`（VPFeatureMatcher：
    prompt 视觉 embedding + mask 几何编码 + QKV cross-attention +
    spatial gate + FiLM 条件注入 + mask→mask decoder）；
  - `projects/v2sam_visual/models/v2sam.py`（RegionPooling、
    `get_contr_loss` 双向对比损失、`constr_prompt_fcs` 构造 SAM2 条件）；
  - `projects/v2sam_visual/models/sam2.py`（`inject_language_embd`）。
- 与 L4 关系：多 prompt expert + contrastive 表示对齐是「视觉 prompt
  一致性」参考；PCCS（post-hoc cyclic consistency selector）在 release
  代码中未检出明确实现。任务为 **cross-view correspondence / SOT**，
  不是同一视频上候选子集限制下的 MOT 身份等价。

### 1.3 DOVTrack（NeurIPS 2025，data-efficient OVMOT）

- 官方仓库：`github.com/zekunqian/DOVTrack`
- Commit：`5748236a`；内容：仅 README（Coming Soon），无代码/license。
- 与 L4 关系：open-vocab MOT 数据效率方向；未提供可审计实现。

### 1.4 TempRMOT（arXiv 2406.05039，referring MOT）

- 官方仓库：`github.com/zyn213/TempRMOT`
- Commit：`6a65640d`；License：**仓库无 LICENSE 文件**。
- 已读文件：`models/transrmot_pro.py`（ClipMatcher、`loss_refers`
  focal/CE）、`models/qim_pro.py`（QueryInteractionModule 轨迹 query
  更新）、`models/memory_bank.py`。
- 与 L4 关系：referring query 驱动 MOT，身份一致性只针对同一 query
  集合；不研究「不同 query/spec 下同一真实对象身份是否保持」。

### 1.5 TellTrack（“Tell Me What to Track”，2024/2026）

- 官方仓库：`github.com/Durablion/telltrack`
- Commit：HEAD（shallow clone）；内容：只有 index.html / privacy policy，
  **无模型代码**。
- 与 L4 关系：referring MOT 论文；无官方实现可审计。

### 1.6 未找到官方实现的 2026 方法（paper-only，不当作依据）

- **EPIPTrack**（arXiv 2510.13235）：explicit + implicit prompts 的
  multimodal MOT；GitHub API 搜索 0 结果，arXiv 页无 code 链接。
- **GOVTrack**（CVPR 2026）：generative OVMOT；未找到官方 repo。
- **ViewSAM**（arXiv 2605.02638）：weakly supervised cross-view
  referring MOT；未找到官方 repo。
- **Robust Exemplar Prompt Learning for MOT**（ACM MM 2026）：未找到
  official repo。
- **COVTrack / COVTrack++**（ICCV 2025 / 2026）：未发布代码。

## 2. 既有 L3/L2/L1 已 clone 审计复核（L4 视角）

以下仓库已在 `docs/l3_reference_audit.md`、`docs/l2_reference_audit.md`、
`docs/l1_d_reference_audit.md` 逐文件阅读，此处只做 L4 相关结论复核。

| Method | Year/Venue | Official repo | Commit | License | 多域 | 多 spec 输入 | Persistent ID | 跨 spec ID 一致性 | Restriction equivariance | 与 L4 关系 |
|---|---|---|---|---|---|---|---|---|---|---|
| SAM3/3.1 | 2026 Meta | sam3 | 96914d24 | SAM License | promptable 视频 | text/box/point/mask/exemplar | 是 | 否 | 否 | prompt 接口碰撞，非核心问题 |
| GLEE | CVPR 2024 | GLEE | f36a49e8 | MIT | 是（BDD/TAO） | text/query | 是 | 否 | 否 | 多域统一部分碰撞 |
| OVTR | ICLR 2025 | OVTR | 500e72c1 | MIT | 否 | text category | 是 | 否 | 否 | open-vocab prompt 历史 |
| OVTrack | CVPR 2023 | ovtrack | e188b32e | Apache-2.0 | 否 | text category | 是 | 否 | 否 | open-vocab prompt 历史 |
| Grounded SAM2 | 2024/25 | grounded-sam-2 | b7a9c29f | Apache-2.0 | 否（pipeline） | text/box/point | 部分 | 否 | 否 | pipeline 无学习型一致性 |
| SAM2MOT | AAAI 2026 | SAM2MOT | 7bdae12c | Apache-2.0 | 否 | 否 | 是 | 否 | 否 | 代码未发布 |
| STORM | CVPR 2026 | STORM-Bench | 0d87c3ba | — | 否 | referring | 是 | 否 | 否 | referring 基准可用 |
| QTrack/RMOT26 | 2026 | QTrack | bc746fe2 | MIT | 否 | query | 是 | 否 | 否 | query-driven MOT |
| AnyTrack | 2026 | AnyTrack | 7d5ca454 | — | SOT | 模态 prompt | SOT | 否 | 否 | 模态统一，非 subset 等价 |
| TDLP | 2025/26 | TDLP | 50344b92 | MIT | 否 | 无 | 是 | 否 | 否 | 下一帧 link prediction |
| Path Consistency | CVPR 2024 | path-consistency | f4b7d26d | Apache-2.0 | 否 | 无 | 是 | 多路径 | 路径级非 spec 级 | 一致性数学框架参考 |
| UniTrack | ICLR 2026 | UniTrack | afdd9869 | README MIT（无 LICENSE 文件） | 是 | 无 | 是 | 轨迹平滑 | 否 | 轨迹正则参考 |
| MOTIP/MOTIP-2 | CVPR 2025 | MOTIP / MOTIP-2 | ffc0e905 / 012856c1 | Apache-2.0 | 否 | 无 | ID 预测 | 否 | 否 | ID decoder 历史 |
| CAMELTrack | 2025 | CAMELTrack | 46a74bb | 见仓库 | 否 | 无 | 是 | 否 | 否 | set-level 竞争 + InfoNCE |
| FDTA | CVPR 2026 | FDTA | b3b3b778 | 有 | 否 | 无 | 是 | 否 | 否 | 判别 embedding |
| TRACT | ICCV 2025 | TRACT | 19f01d72 | 无 | 否 | text | 是 | 轨迹内 | 否 | 轨迹一致性表示 |
| LG-Track / LLTrack | 2023/25 | LG-Track / LLTrack | 432a467 / 2ab7994 | MIT/有 | 否 | 类别互斥 | 是 | 否 | 否 | 语义互斥 cue |
| MeMOTR | ECCV 2022 | MeMOTR | HEAD | 有 | 否 | 无 | 是 | 否 | 否 | memory 参考 |
| NOVA | IROS 2026 | NOVA | 4358a627 | Apache-2.0 | 3D | hybrid base/novel | 是 | 否 | 否 | open-vocab class 严谨 split |
| V²-SAM | CVPR 2026 | V2-SAM | 31c3babf | README MIT（无 LICENSE） | 否 | visual prompt | SOT | cross-view 表示 | 否 | 表示对齐/对比损失参考 |
| DOVTrack | NeurIPS 2025 | DOVTrack | 5748236a | 无 | 否 | text | 是 | 否 | 否 | 代码未发布 |
| TempRMOT | 2024/26 | TempRMOT | 6a65640d | 无 | 否 | referring | 是 | 否 | 否 | RMOT query 更新 |
| TellTrack | 2024/26 | telltrack | HEAD | 无 | 否 | referring | 是 | 否 | 否 | 无代码 |
| EPIPTrack | 2025 | — | — | — | 否 | explicit/implicit prompt | 是 | 否 | 否 | paper-only |
| GOVTrack | CVPR 2026 | — | — | — | 否 | text | 是 | 否 | 否 | paper-only |
| ViewSAM | 2026 | — | — | — | 否 | referring | 跨视角 | 跨视角 | 否 | paper-only |
| COVTrack/++ | 2025/26 | — | — | — | 否 | text | 是 | 否 | 否 | paper-only |

## 3. 非 MOT 的结构参考（只作理论，不冒充 MOT baseline）

- Multiset/Set-equivariant set prediction（ICLR 2022）：集合预测的
  permutation 结构先例；不涉及 persistent identity。
- V²-SAM PCCS 思路（论文，代码未检出）：multi-expert cyclic
  consistency 选择；跨视角 correspondence，不是候选子集限制。
- Path Consistency：多观测路径的关联一致性监督，是 L4 assignment
  consistency loss 的数学近亲；但路径来自时间子采样，不是 spec 候选子集。

## 4. 审计结论

1. **未发现**任何已公开 MOT 方法明确要求
   `Track(Restrict_s(X)) ≈ Restrict_s(Track(X))` 或研究
   「same video + different specification → same persistent identity」；
2. 已公开的 prompt/query 类 MOT（SAM3、GLEE、QTrack、STORM、
   EPIPTrack、TempRMOT、TellTrack）都把 spec 作为输入接口，身份一致性
   只在**同一 spec 内部**维持；
3. Path Consistency 有「一致性」之名，但路径来自观测子采样，不是
   specification 诱导的候选集限制；
4. 因此 L4 的候选子集限制等价性在本次审计范围内无直接等价方法。
