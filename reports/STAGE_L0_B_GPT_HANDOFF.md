# LocateMOT Stage L0-B GPT Handoff Report

Last updated: 2026-08-07
Current Git commit: 8523db352765bddeb29d28a4f35d485ac5e1b1e4（L0-B 最终提交；本报告与 final_status 由 91af008 之后补充）
Current state: L0_B_COMPLETE（`outputs/stage_l0/state.json`）
Overall status: 官方 LocateAnything 复现成功；PBD→ObjectToken 映射验证 100%；小样本特征 sanity 完成
Completed stages: L0 初始化、官方基线审计、环境搭建、L0-A 复现、L0-B 映射验证
Pending stages: L0-C 两帧数据与 cache、P2 Track Decoder 训练、P3 Visual Prompt LoRA、后续评估
Blocking issue: `/data3/testdata/vranlee/LocateMOT` 无写权限（不影响 L0-B；影响后续全量 cache 落盘位置）
Recommended next action: 进入 Stage L0-C，构建固定两帧 pair 数据与小规模 ObjectToken cache，然后训练 B0–B4 关联基线

## 1. Executive Summary

LocateMOT 是一个基于 NVIDIA LocateAnything（3B VLM）做持久多目标跟踪的研究项目。Stage L0-B 只回答一个问题：LocateAnything 最终接受的每个预测框，能否提取严格一一对应、可重复、并具有初步跨帧身份判别能力的 Object Token。

目前完成度：官方模型已在 A100 上成功复现；PBD 生成流程已做事件级插桩；9 张图、36 个查询中 accepted box blocks = 最终解析框 = ObjectToken = 149，映射完整性 100%；同一 seed 重复运行完全一致；YouTube-VOS 小样本特征 sanity 显示 PBD 特征正负 AUC 约 0.77，region 特征约 0.63。

最重要的发现：
1. PBD 真实 6-token 顺序是 `[box_start, x1, y1, x2, y2, box_end]`，官方代码注释中写 “x1,x2,y1,y2” 是误导。
2. Hybrid fallback 中被拒 MTP block 会与后续 AR token 合并成最终框，ObjectToken 必须按“最终被接受路径”提取。
3. 特征 sanity 显示 PBD hidden state 有初步身份信号，但正负分布重叠较大，不能直接做全局匹配。
4. Candidate recall@0.5=75%，约 25% 的 GT 对象没有候选，是后续关联上限的瓶颈信号。
5. 当前唯一最重要的问题：sample 规模小（24 个正对），AUC 结论需要更大样本复核。

建议：进入 Stage L0-C。不建议现在做 RL、长视频或完整 MOT。

## 2. Project Background

- 旧项目 GLEE-PMOT：基于 GLEE 的持久多目标跟踪，已冻结为只读基线（`/data1/LWR/vranlee/SERVER_ONLY/avis/GLEE-PMOT`）。
- 为什么换路线：新项目需要更可控的“定位 + 关联”两阶段：先由 LocateAnything 生成高质量候选与对象特征，再由独立 Track Decoder 做一对一身份关联。
- LocateAnything 负责：图像/文本定位、密集检测、输出 PBD 结构化的框。
- Persistent Track Decoder 未来负责：reference tracks 与 current candidates 的多目标一对一关联（reference 作为 query 是有意反转 MOTIP 方向）。
- 当前 Stage L0-B 还不是完整 MOT：没有 long-video rollout、automatic birth、生命周期、mask decoder、RL。

## 3. Current Research Question

“LocateAnything 最终接受的预测框，能否提取严格一一对应、可复现并具有初步跨帧身份判别能力的 Object Token？”

这是两帧关联、长期 MOT、visual prompt 和 RL 的前提：如果框与特征无法一一对应，后续所有关联与训练都不可靠。

## 4. Environment and Reproducibility

| 项 | 值 |
|---|---|
| 项目路径 | /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT |
| Git commit（L0-B 代码） | 91af008；L0-B 最终 HEAD 8523db3 |
| Eagle 官方 commit | 783f656d127ee498137b5ff52603ce36c292d317 |
| 模型 | nvidia/LocateAnything-3B（本地 /data1/.../models/LocateAnything-3B） |
| checkpoint SHA256 | 923cfc10…0d2d58 / 3459ba10…8fc7d47（两个 safetensors） |
| Python | 3.12.2（venv locatemot） |
| PyTorch | 2.5.1+cu124 |
| CUDA | 12.4（A100-SXM4-40GB） |
| transformers | 4.57.1 |
| attention backend | LLM=sdpa；MoonViT=eager（官方 sdpa 路径在大 token 数下 A100 不稳定） |
| max sequence length | 不超过 4096（A100 官方限制） |
| random seed | 20260806 |
| 运行日期 | 2026-08-06/07 |

## 5. Official Baseline Reproduction

- 测试图片：9 张（COCO val + 合成图）
- 查询：ground_single / detect / negative / point / detect_text / synthetic
- 推理次数：36 + 若干单测
- 成功：默认采样参数下全部完成；失败/异常：detect_text 在少文本图上循环到 max_new_tokens（记录为异常）
- 推理速度：36 次共 29.0s，中位 0.34s/次，最长 12.86s（detect_text 截断）
- 峰值显存：8.5GB（COCO 640px 级）；720p + 4096 token 预算约 10.9GB
- 默认采样：temperature=0.7、top_p=0.9、repetition_penalty=1.1、max_new_tokens 1024–2048
- 已确认问题：greedy 会重复生成同一框直到截断；`<box>None</box>` 实际为大写 `None`；point 偶发幻觉；A100 上 MoonViT sdpa 路径在大图下 OOM/invalid argument（改用官方 eager）。

## 6. LocateAnything Architecture Findings

- MoonViT：patch 14×14，27 层，2×2 patch merger，输出每 token 4608 维。
- MLP projector：`LayerNorm(4608) → Linear(2048) → GELU → Linear(2048)`。
- Qwen：Qwen2.5-3B-Instruct（hidden 2048，36 层）。
- PBD：6-token block；`[box_start, x1, y1, x2, y2, box_end]`；fast=MTP only、slow=NTP only、hybrid=MTP 优先 + AR fallback。
- visual prompt：官方已发布 LoRA 微调脚本与 `visual_prompt=true` 数据转换；当前公开权重不支持 visual prompt 推理。
- 区分：论文声称 12.7 BPS 与高精度；代码实际实现 PBD 解码与 block mask；本项目实际验证了加载、推理、box 解析、hidden-state 提取。

## 7. PBD Generation Trace

最终确认：

- 6-token 真实顺序：`[box_start, x1, y1, x2, y2, box_end]`
- block 位置：生成 token 序列中的绝对位置
- hidden-state 位置：MTP 中与输出位置相同；AR 中为预测该 token 的前一输入位置
- accepted block：`handle_pattern` 返回 coord_box 且格式合法
- rejected block：error_box（MTP 拒绝）
- Hybrid fallback：error_box → AR 逐 token 生成 → box_end → 切回 MTP

代表案例（COCO 000000000139，detect）：

```text
Query: Locate all the instances that matches the following description: person</c>car</c>bicycle.
Decode mode: hybrid
Attempted blocks: ref object → MTP error_box(1) → AR coords(2) → box_end(1) → ref car → empty None → ref bicycle → empty None → end
Rejected blocks: 1 (error_box)
Accepted blocks: 1
Final parsed boxes: 1 (person)
ObjectToken count: 1
Mapping result: 100% 一致
```

## 8. Object Token Definition

ObjectToken 包含：

| 字段 | 来源 | 维度 |
|---|---|---:|
| pbd_box_end / coordinate_mean / full_block_mean | 官方 LLM hidden states（last 层 + penultimate 层） | 2048 |
| region_feature | MoonViT 原始特征（框内 mean pooling） | 4608 |
| geometry_feature | 归一化 xyxy + 面积 | 5 |
| generation_score | 官方解码置信度（明确标注为 generation score，不是检测置信度） | 1 |
| fused_feature | 本项目随机初始化 projection | 256 |

说明：confidence_feature=null（官方没有可解释的 box 置信度）；projection 未训练，不能当作性能证据；原始特征保留供后续训练。

## 9. Mapping Integrity Results

| Item | Count / Result |
|---|---:|
| Evaluated images | 9 |
| Evaluated queries | 36 |
| Accepted box blocks | 149 |
| Parsed final boxes | 149 |
| Generated ObjectTokens | 149 |
| Rejected blocks excluded | 15（error_box） |
| Hybrid fallback cases | 15 |
| Point cases | 8 |
| None cases | 16 |
| Batch-isolation failures | 0 |
| Mapping mismatches | 0 |
| Mapping integrity | 100% |

结论：达到 100% 映射；失败路径（detect_text 截断、point 幻觉）不影响 box ObjectToken 的对应关系。

## 10. Feature Sanity Results

数据集：YouTube-VOS 2019 train（8 视频、2 帧/视频、36 个 GT 对象、24 正对、120 负对）。

| Feature | Positive mean | Positive median | Negative mean | Negative median | AUC | Valid pairs |
|---|---:|---:|---:|---:|---:|---:|
| PBD box-end | 0.925 | 0.993 | 0.882 | 0.935 | 0.768 | 24/120 |
| PBD coordinate mean | 0.792 | 0.881 | 0.650 | 0.695 | 0.777 | 24/120 |
| PBD full-block mean | 0.873 | 0.938 | 0.768 | 0.805 | 0.769 | 24/120 |
| MoonViT region | 0.907 | 0.888 | 0.812 | 0.801 | 0.633 | 24/120 |
| Simple fused (untrained) | 0.906 | 0.944 | 0.826 | 0.852 | 0.788 | 24/120 |

- Candidate recall@0.3/0.5/0.7：75.0% / 75.0% / 66.7%
- 同目标正对 24、同视频负对 + 跨视频负对共 120、时间间隔 0–5 帧
- GT 只用于 prompt/pair 标注与匹配判断；未用于筛选当前帧候选

这是 sanity check，不是正式 MOT benchmark。

## 11. Key Findings

### Confirmed findings

- 官方模型可复现；PBD block 顺序为 xyxy；映射 100%；同 seed 完全确定。
- Hybrid fallback 的最终框由 MTP 被拒 block + AR token 合并，ObjectToken 只取被接受路径。
- PBD hidden feature 有初步跨帧身份信号（AUC≈0.77，24 正对）。

### Likely findings

- coordinate-mean 可能略优于 box-end/full-block（0.777 vs 0.768/0.769），但样本太小，差异不显著。
- region feature 在本小样本上较弱（AUC 0.633），可能受限于 mean pooling 与类别多样性。

### Unknowns

- 更大样本下 AUC 是否稳定；
- fast / slow 模式下的特征是否与 hybrid 一致；
- region 特征是否真的更差（需要更多类别/目标）；
- visual prompt LoRA 是否改善候选 recall。

## 12. Failures and Anomalies

### Greedy 重复框
- Evidence: temperature=0 时模型反复输出同一框直到 max_new_tokens。
- Root cause: MTP box 解码路径的 top-k 平均不受 repetition penalty 有效约束。
- Fix: 使用官方默认采样参数；保留一次记录，不反复执行。

### detect_text 循环截断
- Evidence: 少文本图 247 步、114 框、12.9s、truncated=True。
- Root cause: 模型在无文本图上持续生成微小 `.</box>` 框。
- Fix: 记录为异常；不用 detect_text 作为 ObjectToken 主协议。

### point 幻觉
- Evidence: 无红绿灯图上输出 3 个 point。
- Root cause: point 模式没有严格 none 约束。
- Fix: 记录；point 不产生 box ObjectToken。

### MoonViT sdpa A100 不稳定
- Evidence: 720p 帧 sdpa vision 报 invalid argument / OOM。
- Root cause: 官方 sdpa 路径的 3D bool mask 在 A100/大 token 数下不稳定。
- Fix: 使用官方 eager attention；正确包 no_grad；必要时 `in_token_limit=4096`。

### 本阶段自身 bug（已修复）
- Evidence: region 提取重复跑视觉编码 + 未包 no_grad → 37GB OOM。
- Root cause: 我们的 trace 在 `torch.no_grad()` 外调用 `extract_feature`，autograd 保留全部中间激活。
- Fix: 整个循环包 `torch.no_grad()`；缓存原始视觉特征供 region 复用。
- Validation: 720p 帧峰值降到 10.9GB，映射/特征不受影响。

### 权限问题
- Evidence: `/data3/testdata/vranlee` 属主 testuser，无法创建 `/data3/testdata/vranlee/LocateMOT`。
- Fix: L0-B 全部写入项目目录；全量 cache 前需解决。

## 13. Code and File Changes

| File | Purpose | Status |
|---|---|---|
| locatemot/models/object_tokens/types.py | 事件与 ObjectToken 数据结构 | complete |
| locatemot/models/object_tokens/generation_trace.py | 官方循环插桩 + hidden hook | complete |
| locatemot/models/object_tokens/pbd_extractor.py | PBD 特征提取 | complete |
| locatemot/models/object_tokens/region_extractor.py | MoonViT 区域特征 | complete |
| locatemot/models/object_tokens/projection.py | 256 维 fused 接口 | complete |
| locatemot/models/object_tokens/extractor.py | 顶层提取器 | complete |
| locatemot/evaluation/assignment.py | 多 track NO_MATCH 一对一分配 | complete |
| locatemot/evaluation/token_sanity.py | sanity pair 构建与指标 | complete |
| tools/debug_pbd_generation.py | 事件/token 采集 | complete |
| tools/validate_object_tokens.py | 映射完整性校验 | complete |
| tools/run_token_sanity.py | sanity 运行 | complete |
| tests/（5 个测试文件） | 单元测试 | complete（10 passed） |

未修改 third_party；未产生 patch 文件（以 wrapper/hook 接入）；未复制外部代码到核心包（调用官方 generate_utils 函数并注明来源）；许可证处理见 docs/license_audit.md。L0-B 代码提交 91af008，报告与 final_status 补充提交 8523db3（实际 HEAD）。

## 14. Resource Usage

- GPU：A100-SXM4-40GB × 1（测试时 GPU 8/1）
- 峰值显存：8.5GB（COCO 640px）；10.9GB（720p + in_token_limit=4096）
- 峰值 RAM：未单独记录（进程级 < 20GB）
- 总运行时间：模型加载约 1–2 分钟/进程；36 次推理 29.0s；sanity 约 2–3 分钟
- 每图推理：中位 0.34s，密集 1–2s，detect_text 异常 12.9s
- 输出占用：37MB（JSONL + CSV）
- 预计全量 cache：JSON 约 28–60GB，二进制约 7–10GB（详见 reports/storage_plan.md）
- 建议并发：单进程 1 worker；多进程缓存前先测单 worker 峰值再定并发

## 15. Stage Decision

决策：L0_B_PASS_WITH_LIMITATIONS

- 判断依据：映射完整性 100%、确定性验证通过、接口完整、sanity 显示 PBD 特征有初步身份信号。
- 是否进入 Stage L0-C：是。
- 进入条件：正式 pair 数据构建时使用固定划分、candidate 不读 GT、输出落盘位置确定（/data3 权限或 /data1 空间）。
- 不进入时的最小修复：无（当前无需修复）。

## 16. Recommended Next Stage

主推荐：进入 Stage L0-C，构建固定两帧 pair 数据与小规模 ObjectToken cache，然后训练 B0–B4 关联基线。

- 下一阶段不应做：全量 cache、长视频、automatic birth、RL/GRPO、visual prompt LoRA、DanceTrack/MOT17 正式训练。
- 为什么现在不做 RL：先证明两帧关联与特征有效性，RL 是 Stage L1 之后的事。

## 17. Claim Boundary

可以声称：

- 官方 LocateAnything 在 A100 上可加载、可推理、可解析；
- PBD accepted box 与 ObjectToken 一一对应（9 图/149 token，100%）；
- ObjectToken 接口（PBD 2048、region 4608、fused 256）可用且可复现；
- PBD 特征在小样本上有初步身份判别信号（AUC≈0.77）。

不能声称：

- 已完成 MOT 或达到任何 benchmark 指标；
- visual prompt 已有效；
- 长视频 ID 稳定；
- birth/lifecycle 有效；
- RL 有效；
- 多数据集统一训练成功；
- fused projection 有效（未训练）。

## 18. Important Absolute Paths

- /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/reports/STAGE_L0_B_GPT_HANDOFF.md —— 本报告
- /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/reports/LATEST_GPT_HANDOFF.md —— 最新报告入口
- /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/reports/object_token_validation.md —— 映射验证
- /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/reports/object_token_sanity.md —— 特征 sanity
- /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/docs/pbd_token_mapping.md —— 映射文档
- /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/outputs/l0_b_token_debug/final_status.json —— 最终状态
- /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/outputs/l0_b_token_debug/generation_events.jsonl —— 事件日志
- /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/outputs/l0_b_token_debug/mapping_integrity.json —— 完整性
- /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/outputs/l0_b_token_debug/sanity_metrics.json —— sanity 指标
- /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT —— Git 仓库

## 19. Questions for External GPT Review

1. 当前 PBD ObjectToken 定义（box-end/coord-mean/full-block + region + geometry）是否合理？是否应加类别语义或位置编码？
2. PBD hidden feature 与 MoonViT region feature 应如何融合？region AUC 0.633 是否说明 mean pooling 不适合？
3. AUC≈0.77（24 正对）是否足以支持进入 Track Decoder 训练？需要什么规模的复核？
4. Stage L0-C 应优先 reference-query 还是 current-query 方向？接口如何设计才能低成本对照？
5. Candidate recall@0.5=75% 是否已是关联上限瓶颈？应优先修 candidate 协议还是关联模型？
6. Visual Prompt LoRA 应在哪个时点开始？是否需要先看 B0–B4 结果？
7. 哪些结论证据不足（例如 coordinate-mean 优于 box-end）？
8. 下一阶段最小且有信息量的实验是什么？
