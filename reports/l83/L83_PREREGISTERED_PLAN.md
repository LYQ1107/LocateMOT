# L83 预注册执行计划

日期：2026-09-02
项目根目录：`/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
Luna thread：`01a02014-fce8-7f51-8414-e7ed6ab44745`
起点提交：`75e9f9cd0482645c07a9f71ad4419b0c5f57132b`（L82 分支冻结）

## 科学问题与不变边界

L82 的 dev rank probe 证明 GroundingDINO candidate-reference 表示比 L81 有
query-wise 变化，但 target-level sharpening 仍不足：hard violation `.8653846`、
target-bag R@1 `.288462`、multi-positive coverage `.426752`，且 L59 同容量
control 的 hard `.820513`/R@1 `.320513` 更好。L83 只验证一个纠正：真正的
duplicate-aware target-bag supervision、独立的 query-swap ROC-AUC 和逐 decoder
layer sharpness 是否改变结论；不修改 L82，不改变 bank、候选采集、旧权重、旧
metrics、threshold 或 tracker。

L83 的唯一主假设是：同一真实 target 的 main/reserve/duplicate rows 应先合并为
target bag，positive bag 只需一个可靠 observation，但 multi-target query 仍须
覆盖每个 bag；这种 faithful loss 可能把已有 grounding signal 转成更尖锐的
target correspondence。如果纠正后仍不能在 video-disjoint dev 泛化，则不授权
LoRA/task-composition。

## 固定输入与来源

- L69 budget-40 feature view：`outputs/l69/attempt9/budget40_features/kitti/`，
  17 个 V1/V2 fit video，按 bank 自己的 frame pointers 重建完整 rows。
- fit units：`outputs/l49/data/train_units.jsonl`，仅 V1/V2 `split=fit`，5,314
  rows；video-disjoint split `outputs/l82/protocol/fit_video_train_dev_split.json`，
  524 train groups / 138 dev groups，seed `20260829`。
- frozen controls：L81 step-100、L59 fused ROI、L82 candidate-reference；
  统一 66,561 参数 probe，只重新计算 corrected target-bag metrics。
- fixed manifest：`outputs/l19/protocol/kitti_fast_eval_manifest.json`，预注册
  SHA256 `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`。
- GroundingDINO、L48 text cache、UIDM 与历史 checkpoint 只读；不复制权重或
  写 dense/raw feature cache。

## 预注册顺序

1. 记录 source-of-truth hashes；任一冻结资产漂移即 `source_of_truth_mismatch`。
2. 用 AST/source inspection 审计 L82 五个协议偏差；源码与计划不一致即停止。
3. 构造 `TargetBagLayout`，断言同帧 row order/box/candidate_gt 一致；测试
   duplicate invariance、multi-target、present-uncovered、inactive 和真实 ROC-AUC。
4. 对 L81/L59/L82 现有 checkpoint 只读重算 corrected target-bag baseline。
5. 统一容量、统一 train/dev、统一 optimizer/seed，使用 faithful target-bag loss
   做 10-epoch frozen representation probe。primary bag score 固定为 per-target
   `max`，背景 row 是 singleton negative bag。
6. 无论 faithful gate 是否通过，都必须做一次预注册的 decoder layer-wise
   sharpness audit；该 audit 只是诊断，不会把失败的 faithful gate 变成通过。
   若 faithful gate 失败，后续 factorized energy、LoRA、task composition 和历史
   16/24 semantic 均不运行。
7. 只有 faithful gate 与 factorized gate 都通过，才允许后续监督另行授权大训练；
   本阶段不自动读 screening/official-test labels，不跑 TrackEval/HOTA/MOT/OVMOT。

## Faithful gate（固定，不看结果后改）

对每个 representation，新的 target-bag dev 指标必须满足：G1 corrected bag hard
violation 相对该 representation 的旧 L82-trained probe 改善至少 `.05`，或若旧值
已 `<=.75` 则不得恶化超过 `.01`；G2 bag hit@1 提升至少 `.08`；G3
multi-target exact top-T 提升至少 `.08`；G4 query-swap pair accuracy 不下降
超过 `.02`；G5 V2 bag hard 至少改善 `.03`；G6 完整 finite rows、no deletion、no
truncation。至少一个 GroundingDINO 表示（L59 或 L82）必须通过全部 G1–G6。
representation selection 固定 tuple：lower corrected bag hard、higher bag hit@1、
higher multi-target exact、higher swap margin、higher V2 bag hit@1、simpler representation。

## 资源、监督和停止

首轮 contract/test 使用 CPU；probe 最多 `CUDA_VISIBLE_DEVICES=0,1,2,3`，固定
`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、BF16 仅在 finite sanity 后使用。labels
只在完整 L69 rows/representations 构造后进入 fit loss；dev 是 video-disjoint
fit-derived development evidence，不是 fixed historical validation。present-uncovered
mask correspondence loss，不伪造成 negative；candidate_gt/target IDs 只用于
loss/metrics，不进入 model tensor。token/span→region 与 static/motion alignment
保持 `UNALIGNED`。

失败时只写允许的唯一状态，并保留 attempt/INCOMPLETE：先修首个 actionable
implementation error并做 targeted regression；科学 gate 失败则停止，不增加容量、
不换 threshold/top-k/NULL、不跑 historical 16/24、screening、official test、
TrackEval/HOTA、ordinary MOT 或 OVMOT。最终唯一下一步由 failure decomposition
给出并等待监督。

## 固定 flags

`screening_gt_used=false`；`official_test_labels_read=false`；
`ordinary_mot_ovmot_touched=false`；`hota_trackeval_run=false`；
`candidate_deletion=false`；`candidate_truncation=false`；
`token_span_region_alignment=UNALIGNED`；`static_motion_alignment=UNALIGNED`；
`l81_modified=false`；`l82_modified=false`；`uidm_shared_checkpoint_modified=false`。
