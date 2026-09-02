# L82-B 四 GPU task-composition 训练状态

## 状态

`not_authorized_after_rank_gate_fail`。

L82 的预注册顺序要求先让 candidate-reference representation 在
video-disjoint development split 上通过 rank gate，之后才能运行 Stage G/T/R
的四 GPU LoRA/task-composition 训练。权威 L82-A retry3 没有通过该门，因此本文件
只记录停止原因；没有创建或运行 `task_composition_primary`，也没有把失败的 rank
probe 延长成大训练。

## 已运行的四卡内容

唯一完成的四卡任务是冻结表示公平 rank probe：

- `CUDA_VISIBLE_DEVICES=0,1,2,3`，`world_size=4`，seed `20260829`；
- 524 个 train groups / 138 个 video-disjoint dev groups；
- L59、L81、L82 三个表示各自使用相同的 66,561-parameter probe 和 10 epochs；
- 每个表示 5,240 条聚合 trace（每 rank 1,310 条），finite/nonzero-gradient；
- 三个 checkpoint 均经独立 CPU strict reload 审计。

## 未授权且未运行的内容

以下全部未运行：Stage G exact-matrix grounding、Stage T temporal composition、
Stage R joint RMOT、GroundingDINO LoRA、三 seed、旧 16/24 historical semantic
gate、64/96 screening、official test、TrackEval/HOTA、ordinary MOT、OVMOT、TAO
以及任何 UIDM 修改。

## 原因

L82 candidate-reference 的 hard violation `.8653846` 达到单项上限，但
query-swap AUC `.742515`、target-bag Recall@1 `.2884615` 和 multi-positive
coverage `.4267516` 未达到预注册门槛；同时 L59 control 的 hard violation
`.8205128` 更好。继续 G/T/R 会把未证明的 rank 信号与 temporal/LoRA 变化混在
一起，不能作为合格的科学延伸。

## 资源与隔离

L82 worker 已全部退出。运行期间观察到的外部 `intermot` 进程不属于本项目，未被
终止或修改。没有复制 detector/CLIP 权重，没有写 raw/dense feature cache，
ordinary MOT/OVMOT 入口、共享 UIDM 和旧资产保持冻结。
