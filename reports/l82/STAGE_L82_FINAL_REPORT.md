# Stage L82 — Exact Bipartite Query–Candidate Alignment and Grounding Transfer

## 1. Executive result

L82-A 完成了数据矩阵、GroundingDINO candidate-reference 接口和冻结表示公平
rank probe。权威目录为
`/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/outputs/l82/train/frozen_rank_probe_retry3/`。
结果为 `rank_representation_gate_fail`：L82 表示相对 L81 有真实的 hard-negative
改善，但没有形成足够的跨视频 target-bag/multi-positive correspondence，而且
不优于 L59 control。

## 2. Final status

```text
status=rank_representation_gate_fail
phase_d_authoritative_retry=retry3
phase_e_task_composition=not_run_after_gate_failure
historical_16cal24val=not_run
screening=not_run
official_test=not_run
trackeval_hota=not_run
```

这不是最终 RMOT 成绩，也不是 HOTA/TrackEval 结果。依据预注册停机规则，不启用
L82-B LoRA/task-composition，不继续延长 L82-A。

## 3. Project root / Luna thread / date

- root：`/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
- Luna thread：`01a02014-fce8-7f51-8414-e7ed6ab44745`
- 日期：2026-09-02

## 4. L81 frozen control

L81 step-100 `candidate_evidence` 作为冻结表示控制，未续训或修改。L81 历史固定
semantic validation（来自 L81 immutable report）为 recall `.7096774`、precision
`.0298913`、FP/frame `29.75`、hard violation `.9230769`、multi-positive
`.5138889`；L29 历史 control 为 recall `.7333333`、precision `.0830189`、
FP/frame `10.125`、pred/positive `8.8333`、hard `.9166667`、multi `.8194444`。
这些是历史 control，不是本次 dev rank probe 的逐行同候选集合指标。

## 5. Source-of-truth hashes

固定 manifest 当前 SHA256 仍为
`06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`。
其它关键输入如下：

| 输入 | SHA256/记录 |
|---|---|
| UIDM step11000 | `f6529743630e335b947118bf860cc2215a95315a4c551351e6866ea9e1076343` |
| L81 step100 checkpoint | `2b6131584f4fe0fe018ee4494d61f481ac8eacb5f7ed7abe1125bc4a37c46915` |
| L49 train units | `4546081ef0e2d42e1578f0b9e398987e9dcd7fbafaf5c85888a4b903a88f43f3` |
| L82 video split | `63dd55ee6ed9b678219b0ca485907f1417913bd169c50d50cb0e494886a8a353` |
| L69 feature-bank manifest | `121e2a05362b17566adde8f53affbd0418c4af08e4e9ce389b6fbf7df9318788` |
| GroundingDINO config | `0f120eda9ff3ea03e6447b6ba211ddb5e38c5f947842c04f62e43afd19f621c1` |
| GroundingDINO weight | `b448804bb1af6fa688887f0f2454625edbeeae4e868bc95620e3e6413581051a` |

`outputs/l82/audit/final_integrity.json` 对 manifest、UIDM、L81 checkpoint、17 个
L69 feature files、JSON finite、row retention 和禁止路径逐项检查均为 true。

## 6. 2025–2026 literature and code audit

`reports/l82/L82_2025_2026_LITERATURE_AND_CODE_AUDIT.md` 和
`L82_NOVELTY_COLLISION_MATRIX.md` 记录了真实来源与访问/版本信息，包括 COAL、
STORM、FlexHook、DKGTrack、ReferDINO、VMRMOT、TempRMOT、HFF-Tracker、GroundingDINO
及相关 RMOT 工作。FlexHook 的 parallel expressions/PCD、COAL 的 counterfactual
training、STORM 的 task composition、DKGTrack 的 static/motion heuristic 和
ReferDINO 的 grounding foundation 均被视为已有结构启发或碰撞项，未被包装为
L82 首创。没有外部 checkpoint、私有合成标签或未核验权重进入本阶段；本地
GroundingDINO checkout 也没有被修改。token/span→region 与 static/motion
supervision 仍为 `UNALIGNED`。

## 7. Novelty collision decision

L82 的内部名称仅为 `Exact Bipartite Query–Candidate Alignment and Grounding
Transfer`，未作论文创新声明。真实同帧 cross-query label matrix、candidate×query
interaction 和 factorized nuisance 分离可作为待验证研究区别，但本阶段 rank gate
失败，不能据此宣称新颖性或性能贡献。

## 8. Expression matrix statistics

L82 Route-A 使用 5,314 个 V1/V2 fit units，构成 2,931 个 frame groups，其中
eligible groups 为 662（V1=328、V2=334）。候选平均数量为 45.6636，范围 25–78。
真实监督审计得到 6,074 个 candidate-query flip triplets、1,998 个 target-bag
query flips 和 9,259 个同帧 query pairs。该规模满足预注册的 200/40/100/5000/1000
最低条件；没有伪标签。

## 9. V1/V2 exact query-flip statistics

视频隔离 split 的 dev 为 138 groups；L82 candidate-reference query-swap accuracy
为 V1 `.7150838`、V2 `.7484812`。V1 与 V2 均有可见的 query-wise 变化，但这不等于
target-level 对应关系已经可靠，尤其 V2 hard violation 为 `.9057592`。

## 10. Duplicate-positive and canonical-loss pathology

此前 fit-only pathology audit 已量化 L69 main/reserve duplicate、低质量正行和
L80/L81 minimum-positive 压力，并被作为 L82 target-bag 设计依据。L82 rank probe
报告 target-bag 与 row-level 分数，但没有把 duplicate ID 送入网络，也没有用
best-row oracle 替代真实输出。

## 11. Registered hypothesis

本阶段只检验：Grounding foundation candidate-reference state，加上真实同帧
candidate 轴与 query 轴排序监督及 factorized interaction，能否在跨视频 dev 上
改善 hard-negative ordering，并避免 L81 broad acceptance。L82 不同时改变 bank、
candidate acquisition、threshold、NULL、tracker 或 ordinary MOT/OVMOT。

## 12. Candidate-reference GroundingDINO architecture

L82 复用本地已验证 GroundingDINO 配置/权重，通过 LocateMOT 内部 wrapper 将完整
L69 candidate boxes 映射为 candidate-reference 输入；所有 current rows 同时保留。
三条表示通过同一 probe：L81 `candidate_evidence`、重新构造的 L59 fused ROI、
L82 candidate-reference final hidden。主输出是 `R_iq=interaction`；candidate-only
与 query-only 仅作 nuisance diagnostics。

## 13. Native decoder equivalence

`outputs/l82/audit/grounding_interface_attempt12/contract.json` 的 native
pre-decoder equivalence 为通过，native query/reference 最大绝对差为 0。32 个
fit-only pair 有 candidate-specific expression sensitivity；没有使用 native
top-k boxes、native grounding scores 或 native predicted boxes。candidate permutation
的输入 reference/seed 最大差为 0；decoder output 的 `.0029068` 差异被保留为
CUDA reduction 的 diagnostic，不被误写为 native equivalence 失败。

## 14. Box/reference contract

L69 原始 row order、frame pointer、duplicate candidate rows 和完整 candidate count
进入 wrapper；reference 为固定 candidate box 初始化，不做 bbox refinement。所有
references finite 且 row keys 完整。L82 不把 `candidate_index`、raw rank、source、
pool 或 track ID 作为 feature。

## 15. Forbidden-input audit

ID 仅用于 row/key 连接、causal 数据构造和 provenance；未进入 `R_iq`。旧 L29/L75
分数、threshold、top-k、NMS、GT-derived identity 和 native class score 未进入模型。
interface contract 在 labels=false 期间完成；fit labels 只在完整 representation
构造后用于 matrix/loss。无 future frame。没有 token/span 或 motion-language 的已验证
标注。

## 16. Frozen L81/L59/L82 fair probe

所有 representation 都用相同的
`LayerNorm(256) → Linear(256,256) → GELU → Dropout(.05) → Linear(256,1)`，共
66,561 trainable parameters；schedule、loss、seed、video-disjoint split 和 row
protocol 相同。每个当前 frame 的全部 L69 rows 都保留。

## 17. Rank-only train/dev results

| representation | hard violation | query-swap acc/AUC | target-bag R@1/R@5 | multi-positive coverage | inactive false acceptance | score std |
|---|---:|---:|---:|---:|---:|---:|
| L81 control | .9391026 | .480040/.480040 | .137820/.378205 | .171975 | 1.000000 | .208992 |
| L59 control | .8205128 | .737525/.737525 | .320513/.685897 | .324841 | .979592 | 1.588507 |
| L82 candidate-reference | .8653846 | .742515/.742515 | .288462/.772436 | .426752 | 1.000000 | 1.900830 |

L82 分域 hard violation 为 V1 `.8016529`、V2 `.9057592`；query-swap accuracy
为 V1 `.7150838`、V2 `.7484812`；target-bag R@1 为 V1 `.3140496`、V2 `.2722513`。

## 18. Rank gate checks

通过项：hard violation ≤`.8666667`、相对 L81 改善 `.073718`、query-swap accuracy
≥`.70`、V1/V2 accuracy ≥`.65`、interaction variance 高于 repeat noise、paired
bootstrap positive、complete finite rows、无删除/截断。

失败项：

- query-swap AUC `.742515 < .75`；
- target-bag Recall@1 `.288462 < .7894444`；
- multi-positive coverage `.426752 < .7894444`；
- L82 hard `.865385` 不如 L59 control hard `.820513`。

因此最终 gate 为 `rank_representation_gate_fail`。

## 19. Four-GPU task-composition setup

G/T/R task-composition、GroundingDINO LoRA、causal temporal adapter 和三 seed 只在
rank gate 通过后才可运行。本次没有运行这些内容；不存在可以报告的 L82-B 训练
曲线或 semantic checkpoint。

## 20. Training finite/gradient/reload/DDP evidence

L82-A 使用四卡 DDP、seed `20260829`、10 epochs。三个表示各有 5,240 条聚合 loss
trace（每个 rank 1,310 条），全部 finite/nonzero-gradient。三份 epoch-10 package
在 `reload_audit.json` 中 strict reload 通过，输出 `[2,7,256]→interaction[2,7]`
finite。retry3 总 wall time 约 100.17 秒，world size=4。

初始失败证据已保留：缺少 `tee` 父目录、rank-0 gather list 为空、rank-0 barrier
生命周期和 retention bool 语义反转。修复均为实现/审计层最小修复；retry3 是干净
退出且 flags 正确的权威结果。

## 21. Interaction R, nuisance A+B, final S decomposition

本阶段 primary 只评估 `R_iq`，candidate-only/query-only 是同一 probe 的 nuisance
诊断，没有把 `A+B` 或最终 `S` 伪装成已通过的 emission。L82 的 `R` 方差 `.53519`
和 score std `1.90083` 表明不是完全无变化；但 target-bag 与 multi-positive
指标仍失败，不能授权把它接回 final energy 或 temporal composition。

## 22. Historical fixed 16/24 results

本阶段根据 rank gate 停止，没有读取或重新运行历史 fixed 16-calibration/24-validation
labels。L29/L81 历史数值仅作为 source-derived controls 引用，未与 L82 dev rows
伪配对。

## 23. L29/L80/L81 controls

L29 与 L81 control 均保持 immutable。L80/L81 的历史 semantic failures 和 pathology
均在前置报告中记录；L82 的 fair probe 没有重写旧 score records，也没有把旧分数
接入输入。

## 24. V1/V2 breakdown

L82 在 V1 hard `.8016529`、V2 hard `.9057592`，V2 明显未达到 rank hard gate；虽然
V2 query-swap accuracy 较高，但 target-bag R@1 只有 `.2722513`。不能用 V1 的局部
改善掩盖 V2。

## 25. Multi-positive/duplicate/inactive/present-uncovered breakdown

L82 dev multi-positive coverage `.4267516`，低于 `.7894444`；inactive false
acceptance 为 1.0，且本 rank-only probe没有独立 NULL head。present-uncovered 在
matrix/loss 中保持 coverage mask，没有被写成全负；candidate rows 的 duplicate
candidate index 保留。该结果支持“目标级对应泛化不足”，而不是“通过 NULL 或删行
解决”。

## 26. Ablations

按停机规则未运行 L82-B ablations，也未在 fixed validation 上挑选 ablation。L59
和 L81 是预注册的 representation controls，不是事后 ablation 选择。

## 27. Statistical uncertainty

相对 L81 的 hard-violation improvement point estimate 为 `.0937198`，paired bootstrap
95% CI `[.033303,.156685]`，138 paired groups、1,000 resamples、seed `20260829`。
该 CI 只支持相对 L81 的 rank 诊断，不能替代 target-bag gate 或 HOTA。

## 28. Compute and storage

使用不超过四张 GPU、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`；运行结束无 L82 worker
残留。检查时 `/data1` 约 36 GB 可用；没有复制 1.1 GB detector 权重、没有写 raw/dense
feature cache。只保存 compact JSONL、trace、small probe checkpoints 和报告。

## 29. Evidence classification

| 证据 | 分类 |
|---|---|
| matrix/interface contract | implementation/contract evidence |
| 524/138 split rank probe | fit-only + development validation evidence |
| hard/target-bag/query-swap tables | development rank diagnostics |
| L29/L81 historical numbers | immutable historical control |
| oracle/AUC/HOTA | no oracle ceiling or HOTA claim in this stage |
| screening/official/TrackEval | not run |

## 30. What was not run

未运行 L82-B G/T/R、GroundingDINO LoRA、三 seed、历史 16/24 semantic gate、64/96
screening、official test、TrackEval/HOTA、完整 V1/V2/Dance 正式测试、ordinary MOT、
OVMOT、TAO 或 UIDM 修改。

## 31. Changed files

L82 新代码/审计包括 `locatemot/models/l82_grounding_reference.py`、
`locatemot/models/l82_rank_probe.py`、`locatemot/rmot/l82_grounding_runtime.py`、
`locatemot/rmot/l82_losses.py`、`tools/l82_train_frozen_rank_probe.py`、
`tools/l82_audit_final_integrity.py` 和 `tools/l82_audit_checkpoint_reload.py`，以及
L82 专用 data/audit helpers。旧 L11–L81 源码、bank、checkpoint、GT、TrackEval、
ordinary MOT/OVMOT 入口未修改。

## 32. Commands

核心权威命令为：

```text
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/data1/LWR/vranlee/LLM/mmdetection-3.3.0:/data1/LWR/vranlee/SERVER_ONLY/avis \
/home/lwr/anaconda3/envs/masaenv_debug/bin/python -m torch.distributed.run \
--standalone --nproc_per_node=4 tools/l82_train_frozen_rank_probe.py \
--out outputs/l82/train/frozen_rank_probe_retry3
```

完整命令和运行时参数写入 `outputs/l82/train/frozen_rank_probe_retry3/provenance.json`。

## 33. Limitations

当前 dev split 仍是 fit-video 内部开发集，不是全新外部 benchmark；candidate-reference
state 的 expression sensitivity 不等于可靠 grounding；rank-only 没有 NULL head；
没有 verified token/span 或 motion-language supervision。L69 V2 的 proposal coverage
也不能被本阶段 rank probe重新定义。

## 34. Failure root cause

第一 actionable root cause 为
`query_candidate_correspondence_ceiling_insufficient_for_target_bag_generalization`：
L82 有候选特异交互，但同帧 query-swap 的 aggregate 仍未达到 AUC 门槛，target-bag
R@1 和 multi-positive coverage 很低，并且 L59 control 更强。不是 finite、DDP、
candidate deletion 或监督规模导致的实现停机。

## 35. 唯一下一行动

`STOPPED_PENDING_SUPERVISOR_REVIEW`：由监督者批准一个新的单因素结构/数据解阻塞
假设；本轮不运行 L82-B，不增加容量，不调 threshold/top-k/NULL，不读取 screening/test
labels，也不触碰 ordinary MOT/OVMOT。

## 36. Integrity flags

```text
screening_gt_used=false
official_test_labels_read=false
ordinary_mot_ovmot_touched=false
hota_trackeval_run=false
candidate_deletion=false
candidate_truncation=false
token_span_region_alignment=UNALIGNED
static_motion_alignment=UNALIGNED
gpu_world_size=4
l81_modified=false
uidm_shared_checkpoint_modified=false
```
