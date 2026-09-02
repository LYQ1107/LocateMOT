# L82-A 冻结表示 rank probe 报告

## 结论

权威运行是
`/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/outputs/l82/train/frozen_rank_probe_retry3/`。
其工程合同完整，但预注册的 `rank_representation_gate` 失败，状态为
`rank_representation_gate_fail`。L82-B 的四卡 task-composition/LoRA 训练没有
被授权，也没有读取历史 16/24 validation labels、screening/test labels 或运行
TrackEval/HOTA。

## 范围与数据

- 项目根目录：`/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`。
- Luna thread：`01a02014-fce8-7f51-8414-e7ed6ab44745`。
- 使用 L82 Route-A 的真实 fit-only matrix：662 个 eligible frame groups
  （V1=328，V2=334），来自 5,314 个 V1/V2 fit units；固定视频隔离 split 为
  train=524、dev=138，未把同一视频拆到两边。
- matrix 审计记录 6,074 个 candidate-query flip triplets、1,998 个
  target-bag query flips。`present_uncovered` 在 loss 中保持 coverage mask，
  没有伪造成 inactive negative。
- 每个表示使用完整 L69 budget-40 当前帧候选 rows，未使用 source/pool/group/
  query/track/candidate ID 作为 feature，未做 candidate deletion/truncation。

## 表示与公平 probe

三条输入表示使用完全相同的 66,561 参数 probe：

```text
LayerNorm(256) -> Linear(256,256) -> GELU -> Dropout(0.05) -> Linear(256,1)
```

主分数是交互项 `R_iq=interaction`；candidate-only/query-only 只是同一 probe
的 nuisance diagnostics。比较项为：

1. frozen L81 step-100 `candidate_evidence`；
2. 在相同 L69 rows 上重新构造的 L59-style fused ROI；
3. L82 GroundingDINO candidate-reference final hidden。

GroundingDINO interface contract 已在
`outputs/l82/audit/grounding_interface_attempt12/` 通过：native query/reference
最大差为 0，模型参数冻结，32 个 fit-only expression pairs 有 candidate-specific
变化，完整 rows 保留。L82 不使用 native top-k boxes/scores，也没有修改第三方
MMDetection checkout。没有 token/span→region 或 static/motion 标注，状态仍为
`UNALIGNED`。

## 训练与重载

retry3 使用 seed `20260829`、AdamW `2e-4`、weight decay `1e-4`、5% warmup、
cosine schedule、gradient clip=1.0、四进程 DDP、10 epochs。524 个 train groups
按视频隔离切分并由四个 rank 分片；每个表示的 trace 有 5,240 条 rank-local
update 记录（每 rank 1,310 条），全部 finite 且 `nonzero_gradient=true`。

| 表示 | loss mean | first | last | grad norm min–max | checkpoint SHA256 |
|---|---:|---:|---:|---:|---|
| L59 fused ROI | 1.609606 | 2.133206 | 0.835648 | 0.839528–10.254317 | `bc957ed71af26716a060395f9bbf2e23dddb41c1e555e02d19f77ece64f2eb2b` |
| L81 candidate evidence | 2.191645 | 2.171594 | 1.719815 | 0.429719–2.060553 | `9fca222c37694ac760aa6244408a58648b369e305a45413acd3ec2c7092bb5e9` |
| L82 candidate reference | 1.573012 | 2.436326 | 1.400790 | 0.886400–6.652157 | `371edd45f5715a46ce839052be67d98705f1021bc9737d8431974fde03a6fcc6` |

独立 CPU strict reload 审计
`outputs/l82/train/frozen_rank_probe_retry3/reload_audit.json` 对三个 package
均得到空 missing/unexpected keys、finite 输出，输入 `[2,7,256]` 的 interaction
输出为 `[2,7]`。`outputs/l82/audit/final_integrity.json` 的所有冻结资产、
finite、行完整性和禁止路径检查均为 true。

## Dev rank 结果

指标来自 138 个 video-disjoint dev groups，主分数为 `R_iq`，不是最终 emission。

| 表示 | hard violation | query-swap acc/AUC | target-bag R@1/R@5 | multi-positive coverage | inactive false acceptance | score std |
|---|---:|---:|---:|---:|---:|---:|
| L81 control | .9391026 | .480040/.480040 | .137820/.378205 | .171975 | 1.000000 | .208992 |
| L59 control | .8205128 | .737525/.737525 | .320513/.685897 | .324841 | .979592 | 1.588507 |
| L82 candidate-reference | .8653846 | .742515/.742515 | .288462/.772436 | .426752 | 1.000000 | 1.900830 |

L82 按域分解：

| 域 | hard violation | query-swap accuracy | target-bag R@1 | multi-positive coverage |
|---|---:|---:|---:|---:|
| V1 | .801653 | .715084 | .314050 | .400000 |
| V2 | .905759 | .748481 | .272251 | .445652 |

L82 相对 L81 的 hard violation 改善为 `.073718`，paired bootstrap 95% CI 为
`[.033303, .156685]`（138 groups，1,000 resamples，seed `20260829`），因此
它确实产生了比 L81 更有用的某些交互变化。但它没有形成足够的 target-level
对应关系：query-swap AUC `.742515 < .75`，target-bag R@1 `.288462`，
multi-positive coverage `.426752`，均低于预注册门槛；同时 L59 control 的
hard violation `.820513` 和 target-bag R@1 `.320513` 更好，触发
`l82_not_worse_than_l59=false`。

## Gate 判定

`outputs/l82/train/frozen_rank_probe_retry3/rank_gate.json` 的结果：

- 通过：dev hard violation ≤ `.8666667`；相对 L81 改善 ≥ `.05`；query-swap
  accuracy；V1/V2 query-swap accuracy；interaction variance 高于 repeat noise；
  paired bootstrap positive；完整 rows/finite/no deletion/no truncation。
- 失败：query-swap AUC、target-bag Recall@1、multi-positive target coverage、
  `l82_not_worse_than_l59`。
- inactive false acceptance 对 L82 仍为 1.0；本 rank-only probe 没有 NULL head，
  因而不能把该问题解释为已解决。

这不是 HOTA、TrackEval、screening 或正式 RMOT 成绩。它也不是 validation gate：
开发集全部来自 fit videos 的视频隔离 dev，历史 16/24 labels 仍未读取。

## DDP 启动故障记录

期间出现的两类“每卡两个进程”现象是执行故障而非模型结构：最初 `tee` 的父目录
不存在，导致一份 torchrun 仍被启动；随后又启动了正确的 launcher，造成两组 worker
重叠。确认后只终止了已核实的 L82 进程组，没有触碰外部 GPU0 上的 `intermot`。
随后依次修复了 rank-0 `gather_object` 目标列表、rank-0 提前 return 前的 barrier，
以及仅报告层的 candidate-retention bool 反转。原始目录、traceback 和 retry1/2
均保留；retry3 是第一次干净退出且 flag 语义正确的权威结果。

## L82-B 边界

由于 rank gate 失败，未运行四 GPU task-composition（G/T/R）、GroundingDINO
LoRA、三 seed、历史 16/24 semantic gate、screening、official test、TrackEval、
HOTA、ordinary MOT 或 OVMOT。不能用 fit loss、bootstrap、AUC 或 L59 control
包装成 L82 成功。

## 完整性 flags

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
