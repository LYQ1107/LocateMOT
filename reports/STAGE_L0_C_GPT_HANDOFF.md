# LocateMOT Stage L0-C GPT Handoff Report

Last updated: 2026-08-07
Current Git commit: 见 outputs/l0_c/final_status.json（L0-C 主提交）；最终 HEAD 见 Git log
Current state: L0_C_COMPLETE
Overall status: 两帧多目标关联全链路已跑通；B4 优于简单 PBD cosine 但幅度有限（+1.3pp）；候选覆盖是主要瓶颈

## 1. 项目背景（自包含）

LocateMOT 是基于 NVIDIA LocateAnything-3B 的持久多目标跟踪研究项目。LocateAnything 负责把图像+文本变成结构化预测框；本项目在其之上设计 ObjectToken 与关联模型，目标是两帧/多帧的一对一身份关联与 NO_MATCH 判断。

Stage L0-B 已证明：LocateAnything 最终接受的每个预测框可以严格一一对应地提取 ObjectToken（PBD hidden state + MoonViT region），映射完整性 100%。

## 2. Stage L0-C 的科学问题

“LocateAnything 产生的 ObjectToken，经过可学习的 pairwise 模型或 Persistent Track Decoder 后，能否在两帧、多 reference、多 candidate 和目标缺失场景中完成可靠的一对一身份关联与 NO_MATCH 判断？”

## 3. 数据与划分（实际）

- 数据集：YouTube-VOS 2019 train、MOSE v2 train（只读）
- 冻结划分来源：旧项目已冻结 manifest（6066 train / 300 calibration / 1071 held-out）
- 本项目子采样：train 400（200 YT + 200 MOSE）、calibration 80（40+40）、held-out 150（75+75）
- 三者视频级互斥（train∩calib∩heldout=0），无文件缺失
- pair：train 6858、calibration 1383、held-out 2556；unique videos 400/80/150
- 划分文件：configs/data/l0_c_{train,calibration,heldout}_videos.json

## 4. Candidate cache（实际）

- 3780 shards，429MB，float16 safetensors + meta json
- 路径：/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L0C/cache
- 协议：
  - category_guided（YouTube，按 GT 类别模板；GT 仅用于 prompt/标签，不用于 current 候选筛选）
  - generic（冻结模板 "Locate all instances of objects."，在 calibration 8 帧上从 3 个模板中选出）
- recall（cache 全量统计）：
  - category_guided：0.898 / 0.862 / 0.801 @0.3/0.5/0.7
  - generic：0.589 / 0.528 / 0.469

## 5. Pair 与标签语义

- candidate_missing 与 true_no_match 严格区分：目标在 current 存在但无候选 → candidate_missing，不计入 association loss，在 e2e 指标中计为失败；目标不存在/完全遮挡 → true_no_match。
- held-out 中 candidate_missing reference 693 个；true no-match 占比约 10–15%。

## 6. 模型与训练

- 特征统一投影 256 维：PBD coordinate-mean（last 层）+ MoonViT region + geometry + generation score + 协议 embedding。
- B3 Pairwise MLP：calibration 最佳 0.547（全量 6858 pairs，600+ steps）。
- B4 Persistent Track Decoder：d=256、4 层、8 heads、FFN 1024；reference-query；calibration 最佳 0.609（全量 600 steps）。
- 训练：AdamW lr=2e-4、wd=1e-4、cosine、warmup 5%、bf16、grad clip 1.0、早停 patience 8。
- checkpoint 选择只使用 calibration；held-out 未参与任何选择。

## 7. 方向检查

- 同一 1500-pairs 子集、同一训练步数：
  - reference_query calibration 0.7206
  - current_query calibration 0.7181
- 差异 0.25pp < 1pp → 保留 reference_query。

## 8. Held-out 正式评测（B0–B4）

总 reference 4746，candidate-conditional 2878：

| 模型 | e2e | conditional | NO_MATCH F1 | ID F1 |
|---|---:|---:|---:|---:|
| B0 IoU | 0.496 | 0.743 | 0.747 | 0.796 |
| B1 Region cosine | 0.356 | 0.544 | 0.431 | 0.540 |
| B2 PBD cosine | 0.415 | 0.614 | 0.701 | 0.666 |
| B2 box-end | 0.434 | 0.643 | 0.719 | 0.699 |
| B3 Pairwise MLP | 0.413 | 0.618 | 0.627 | 0.633 |
| B4 TrackDecoder | 0.425 | 0.627 | 0.735 | 0.649 |

定位：loc@0.5≈0.999、loc@0.7≈0.93（已匹配候选上）。

## 9. 分层结果（B4）

| 维度 | 子组 | conditional | NO_MATCH F1 |
|---|---|---:|---:|
| dataset | youtube_vos | 0.727 | 0.455 |
| dataset | mosev2 | 0.293 | 0.819 |
| protocol | category_guided | 0.723 | 0.455 |
| protocol | generic | 0.542 | 0.772 |
| gap | 1–4 | 0.405 | 0.849 |
| gap | 5–16 | 0.599 | 0.833 |
| gap | 17–64 | 0.612 | 0.717 |
| gap | >64 | 0.791 | 0.438 |
| target | 1 | 0.796 | - |
| target | 2–4 | 0.589 | - |
| target | 5–8 | 0.229 | - |
| candidate | present | 0.627 | 0.536 |
| candidate | missing | - | 0.798 |

## 10. 消融结论

- PBD box-end（0.643）略优于 coordinate-mean（0.614），但差距小。
- learned fused（B4 0.627）> PBD cosine（0.614）> Region cosine（0.544）。
- B4 比 B3 在 NO_MATCH F1 上强 10.8pp，assignment 仅 +0.9pp。
- 多目标 5–8 竞争是最大关联短板（0.229）。

## 11. 主要失败类型与瓶颈

- 主要瓶颈是 candidate 覆盖：generic recall@0.5 仅 0.528，MOSE conditional 仅 0.293。
- B4 vs B2 只有 +1.3pp，未达到预设 ≥2pp 的“明显优于简单余弦”标准（vs B1 超过 2pp）。
- 长间隔 >64 的 NO_MATCH F1 低（0.438）。
- 训练稳定性曾出现 NaN，根因是 attention mask 语义与空样本全 mask，已修复。

## 12. 当前可以/不能声称

可以声称：

- 数据划分无泄漏；candidate 生成未读取 current GT；多 track 可同时 NO_MATCH；一对一无重复分配。
- B4 在 NO_MATCH 判断上显著优于 B3；B4 优于 B1/B2 的 conditional accuracy（vs B2 幅度有限）。
- 全链路（LocateAnything→cache→B3/B4→Hungarian）可运行、可复现。

不能声称：

- 已满足“B3/B4 明显优于所有简单基线 ≥2pp”的预设成功标准（vs B2 未达）。
- 已实现 MOT 或 benchmark SOTA；visual prompt 尚未训练；RL 未做。

## 13. 建议

唯一主推荐：进入 Visual Prompt LoRA 或先做 generic candidate 协议改进，优先提升 candidate recall，再做一轮 B3/B4 训练与评测。

理由：conditional 关联已有初步能力（B4 0.627），但 e2e 被 candidate_missing 和 generic recall 限制；先解决候选覆盖比继续堆 decoder 更有效。

## 14. 重要路径

- /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/reports/STAGE_L0_C_GPT_HANDOFF.md
- /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/reports/l0_c_evaluation.md
- /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/reports/l0_c_training_report.md
- /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/reports/l0_c_failure_analysis.md
- /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/outputs/l0_c/final_status.json
- /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/outputs/l0_c/baseline_results.csv
- /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/outputs/l0_c/pair_manifest.jsonl
- Git：/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT

## 15. 给外部 GPT 的问题

1. B4 相对 B2 只高 1.3pp，是否说明 PBD cosine 已接近该特征空间上限？
2. generic recall 0.528 应优先修 prompt、加 visual prompt，还是换候选生成？
3. MOSE 困难是否主要来自 generic-only 与类别缺失？
4. 多目标 5–8 竞争该加 reference 间显式竞争还是更多数据？
5. 长间隔 NO_MATCH 低，是否应引入时间上下文 embedding？
6. B3/B4 是否值得继续训练（当前 600 steps），还是应先解决数据瓶颈？
7. 下一阶段最小有信息量的实验是什么？
