# Stage L1-B Reference Audit（Universal Identity Representation）

审计时间：2026-08-08。范围：2025–2026 的 universal / cross-category /
generalizable object identity representation 官方实现。原则：不根据论文结构图
或记忆写复杂模块；每个设计点必须有真实代码证据。

## 1. OG-ReID — Object-Generalized Re-Identification（CVPR 2026）

- Paper: Object-Generalized Re-Identification: A Step Towards Universal
  Instance Perception（CVPR 2026，Chen et al.）
- Official repo: https://github.com/whucsy/OG-ReID
- 审计结果：`git ls-remote` 无任何 refs（仓库为空/未公开/私有）。
- 结论：`NO VERIFIED OFFICIAL IMPLEMENTATION FOUND`。仅记录论文公开思想：
  统一身份空间需同时满足 category-agnostic（跨类同一实例）与
  intra-category discriminative（同类不同实例可区分）；其 MGOR 用
  semantic distribution regularization 处理类别迁移。
- 采用：作为科学目标与评估口径参考（same-category discrimination 是核心），
  不采用任何代码。

## 2. VICP — Generalizable Object Re-Identification via Visual In-Context
Prompting（ICCV 2025）

- Paper: arXiv 2508.21222
- Official repo: https://github.com/Hzzone/VICP
- Local: `references/universal_identity/VICP`
- Commit: `5ae97924d2880a941c5b38daa2e8b05a4767b628`
- License: 无 LICENSE 文件（仅 README；不允许复制代码）
- 实际实现：few-shot positive/negative pairs 作为 in-context prompts；
  LLM 推断语义身份规则，指导冻结 VFM（DINO）提取 identity-discriminative
  features（`train_vpt_lora.py`，dynamic visual prompts）。
- 采用：reference/prompt conditioning 概念（对后续 prompt 阶段有参考价值）；
  不采用 LLM 语义规则模块（训练/推理开销大且不是本阶段核心）。
- 不采用原因：本阶段 Identity Adapter 输入是 LocateAnything ObjectToken，
  不是原始像素；LLM 规则与 unified checkpoint 目标冲突。

## 3. UPCL — Unbiased Prototype Consistency Learning for Multi-Modal and
Multi-Task Object Re-Identification（NeurIPS 2025）

- Official repo: https://github.com/ZhouZhongao/UPCL
- Local: `references/universal_identity/UPCL`
- Commit: `c2c01c2b4ecbe79b39de555da872647d10a55ff8`
- License: 无 LICENSE 文件
- 实际实现：一个统一模型跨 modality / category 做 ReID；loss 以
  ID softmax（+label smoothing）+ triplet + center 为主
  （`layers/make_loss.py`），配合 prototype consistency learning。
- 采用：共享 embedding + 主 identity loss（ID softmax/triplet 之一）+
  prototype consistency 作为候选目标；跨类别统一训练思路。
- 不采用：多模态专属 head、center loss 堆叠。

## 4. UniTrack — Differentiable Graph Representation Learning for MOT
（2026）

- Official repo: https://github.com/ostadabbas/UniTrack
- Local: `references/universal_identity/UniTrack`
- Commit: `afdd9869d31ff115d2fe03b14dd36e0b4f366557`
- License: 无 LICENSE 文件
- 实际实现：把 detection / identity / temporal consistency 统一为图
  表示学习（`unitrack_criterion.py`：tracking score + spatial +
  temporal consistency loss，几何/轨迹级约束为主）。
- 采用：几乎不采用（它解决 association-level 一致性，不是 embedding 学习）；
  记录为关联方法对照。

## 5. 复用已有本地审计（L1-A 已做）

- FDTA（CVPR 2026，MIT，commit b3b3b77）：Identity Contrastive 学习
  discriminative object embeddings；本阶段 identity loss 设计参考其
  contrastive 目标。
- MOTIP / MOTIP-2（CVPR 2025，MIT）：把 MOT 视为 ID prediction；
  ID 词表 + 阈值，是 identity-as-class 路线的官方证据。
- MeMOTR / CO-MOT / GTR / OC-SORT：association/memory 方法，本阶段不采用
  （L1-B 明确先不加 trajectory/memory）。

## 6. 本阶段 Identity Adapter 设计来源（结论）

- 输入：LocateAnything ObjectToken（PBD box-end / coordinate / MoonViT
  region），全冻结。
- 结构：轻量 projection（维度待定，256–512 参考 VICP/UPCL 特征维度），
  共享参数，无 dataset head。
- 主目标候选：Supervised Contrastive / InfoNCE 或 ID softmax+triplet
  （以官方代码审计为准，第一版最多 1 主 + 1 辅助）。
- 评估核心：same-category retrieval（R@1/mAP）必须单独报告。
