# Stage L3 — Object Specification Backbone 审计

日期：2026-08-10。

## 1. 候选

### S0 — LocateAnything / PBD（现有）

- 优点：现有 PBD cache/manifest 工程成熟（L0–L2 全部基于它）；
- 缺点：text 提示非原生；普通 grounding LoRA 已失败
  （LORA_PBD_DEGRADED）；视觉 prompt（box/point/mask）无原生编码；
- 定位：继续作为 object/track token 表示，spec 用额外轻量编码。

### S1 — SAM3（Meta 2026）

- 能力：text/point/box/mask/exemplar 统一编码，open-vocab；
- 限制：SAM License（需确认条款）、HF 登录、CUDA 12.6 + torch 2.10
  + flash-attn-3；本环境 torch 2.x/CUDA 12.1 不匹配；
- 定位：B 的强 comparison；不作为默认（工程成本高）。

### S2 — GLEE（CVPR 2024，MIT）

- 能力：text（CLIP 类）+ image/video 多任务；
- 限制：完整模型大；env 无 CLIP；且 GLEE 是检测 foundation，
  与 AC association 核心叠加成本高；
- 定位：comparison/消融。

## 2. Pilot 采用

本阶段 B 未进入训练（A Gate 未过）。若继续，最小实现：

- spec = learned embedding（ALL + BDD 11 类 + person + OPEN，
  见 `locatemot/models/l3_unified.py::SPECS`）；
- 候选类别兼容输入：`cand_spec_compat`（0/1）附加到 candidate
  features（类别来自检测侧；评估时可用候选 GT 类别过滤计算指标）；
- open-vocab：transformers 提供 frozen text encoder 作为扩展，
  不阻塞。

## 3. 结论

- S0 作为主 object token；spec 编码器从轻量 learned embedding 起步；
- S1/S2 仅在需要“open-vocab text / visual prompt”强证据时评估，
  且需处理 License/环境限制；
- 当前不因 SAM3/GLEE 存在就放弃 B，但 B 的 novelty 边界已收缩
  （见 `reports/l3_novelty_collision_audit.md`）。
