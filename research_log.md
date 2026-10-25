Warning: truncated output (original token count: 172051)
Total output lines: 5288

## 2026-10-24 — real native 三任务联合训练 step3500 checkpoint 完整

联合 GroundJEV/OneJev native MOT/RMOT/OVMOT 训练已到 step3500/8227（examples=56,000），最近事件 loss=1.365135、CE=0.988602、Brier=0.376534、slot_acc=0.6775、sec/step=23.649。`adapter-step3500` 已完整落盘；adapter safetensors 为 43,346,432 bytes，SHA-256 `91014b14a2aad8b06112fbdf911ad45ec80c4c141738777303c5310f7eaf6461`，372 个 tensor 的 CPU `safe_open` 校验通过。owner 仍 RUNNING，尚无 terminal receipt 或 JEV held-out 指标。

## 2026-10-24 — real native 三任务联合训练 step3000 checkpoint 完整

联合 GroundJEV/OneJev native MOT/RMOT/OVMOT 训练已到 step3000/8227（examples=48,000），最近事件 loss=1.356744、CE=0.983385、Brier=0.373359、slot_acc=0.6875、sec/step=23.481。`adapter-step3000` 已完整落盘；`adapter_model.safetensors` 为 43,346,432 bytes，SHA-256 `f83220d5e43e3e6b3791290004aa17fd91bebc31a6e85d0f29cf04588084a166`。owner 仍为 `REAL_NATIVE_THREE_TASK_TRAINING_RUNNING`，尚无 terminal receipt，JEV held-out 评测尚未启动。

## 2026-10-24 — real native 三任务联合训练 step2500 checkpoint 完整

现有 compact native GroundJEV/OneJev 联合训练推进到 step2500/8227（examples=40,000），最近事件 loss=1.410957、CE=1.020909、Brier=0.390047、slot_acc=0.67、sec/step=23.615。`adapter-step2500` 已完整落盘并通过文件级 SHA-256/PEFT 配置审计（adapter safetensors 43,346,432 bytes，SHA-256 `2a5301825abe51d59f21808b826843b4870cb2a386c57809c76d976d94107e99`）；基础模型路径仍为 `onejev_unified600_native_v2/ckpt-step600`。训练 owner 仍 RUNNING，无终止 receipt，尚未进行 JEV held-out 指标评测。

## 2026-10-24 — real native 三任务联合训练 step2400 进度与恢复点

现有 compact native GroundJEV/OneJev 联合训练仍在 GPU3、GPU7 运行，未重启、未迁移 world size。权威 `log.jsonl` 已到 step2400/8227（examples=38,400），最近一步 loss=1.384645、CE=1.023810、Brier=0.360835、slot_acc=0.6975、sec/step=24.218、峰值显存约 7.80 GiB；最近完整 adapter checkpoint 为 `adapter-step2000`，并保留 `adapter-step500/1000/1500`。owner 状态仍为 `REAL_NATIVE_THREE_TASK_TRAINING_RUNNING`，尚无终止 receipt 或 JEV held-out 指标。训练健康但不能据此宣称性能提升；终止后先 CPU merge 到独立模型目录，再用相同 held-out scopes 进行 TrackJEV 官方评测。

## 2026-10-24 — post-joint OneJev merge gate prepared

The active native three-task trainer saves a PEFT LoRA adapter, while the
TrackJEV runtime requires a standalone multimodal model directory. Added
`tools/groundjev/merge_onejev_adapter.py`, which is gated for post-terminal use:
it refuses a non-empty output, preserves the base and adapter, merges on CPU,
copies the processor/tokenizer and writes a load-contract receipt. It has only
been compiled; no model was loaded and no GPU or active training process was
touched.

## 2026-10-24 — bounded audit confirms immutable source/runtime box mismatch

The full 131,214-row recomputation was intentionally stopped after about 20 minutes because it was auxiliary CPU work and risked slowing the protected GPU training. A deterministic 512-row prefix audit completed: all 512 stored matrices match the correct native `xyxy` calculation and the converted GT `xywh` calculation, while 0/512 match the historical raw-`xyxy`-as-`xywh` runtime interpretation; all 512 lack an explicit box-format tag. The first row is stored `1000,44,...`, correct native `xyxy` is `44`, and raw-as-`xywh` is `571`. This confirms the active immutable source was generated consistently in converted geometry but is unsafe to consume through an implicit runtime convention. No source row or active checkpoint was modified. Evidence: `docs/GROUNDJEV/TRACKJEV_BOX_CONTRACT_AUDIT_20261024.{json,md}`.

## 2026-10-24 — TrackJEV xyxy/xywh 契约审计与运行时修复

审计发现 GroundAnything/OnlineTracker 的候选框是像素 `xyxy`，而历史 TrackJEV 竞争模块默认按 MOT `xywh` 解读。旧导出器曾用转换后的 `xywh` 计算竞争矩阵，却把原始 `xyxy` 写回状态；其历史行的记录 IoU=44‰，若运行时误按 `xywh` 读取同一候选则为571‰。修复保留历史 `xywh` 默认接口，新增显式 `box_format` 边界：Native 导出、TrackJEV adapter/model/tracker 和真实权重 smoke 使用 `xyxy`，GT 匹配继续使用 `xywh`，裁剪改为 `(x1,y1,x2,y2)`。当前联合训练的 immutable manifest 未改写、未重启。`trackjev_selftest.py`、compileall 和 xyxy/xywh 等价数值回归均通过；该修复尚未产生新的 JEV 性能指标。

## 2026-10-24 — G2 Clean vs G0 独立 held-out 比较

在相同输入、解析器、dense decoder、Simple tracker 和官方 TrackEval scorer 下，G2 Clean step18227 相比冻结 G0：MOT20-05 validation1657 的 HOTA 0.342615→0.444739（+0.102123，+29.80%），IDF1 0.290533→0.400865，IDSW 13,712→8,464；RMOT multi0007 的 HOTA 0.314869→0.507575（+0.192706，+61.20%），IDF1 0.374223→0.565583，IDSW 123→107。RMOT single0007 的 HOTA 0.053121→0.061876，但 IDF1 0.014388→0.012878 且 IDSW 2→3，属于混合变化。TAO OVMOT video105 两者均 40/40 帧拒绝、TETA=0。四个 held-out scope 与 G2 TRAIN manifest 的 exact image overlap 为空。该结果是 G2 感知前端+固定 Simple tracker 证据，不是 JEV/OneJev 联合模型指标。证据：`docs/GROUNDJEV/G2_CLEAN_VS_G0_HELDOUT_COMPARISON_20261024.{json,md}`。

## 2026-10-24 — real native 三任务联合训练 step1225 进度审计

当前 compact native GroundJEV/OneJev 联合训练仍在 GPU3、GPU7 运行，使用 131,618 行原生 MOT/RMOT/OVMOT manifest（`compact_competition_v1`，SHA-256 `7c92778fb4417ea1e49bf286cab7758ceef1443d683ab059e5e758e5bd823aa3`），计划 8,227 steps。最新权威事件为 step1225/8227、examples=19,600，loss=1.627591、CE=1.209416、Brier=0.418175、slot_acc=0.615，梯度与显存均有限。step1000 的完整恢复状态和 adapter 已在 CPU 重载核验，SHA 分别为 `f3b37033f57963dce452b384503009fc587531ac05f6cc16742dd511103e5213` 和 `10a54906af4f001ba466b58ee3cc9c0655721594cc7de8f70f3938256fd505b8`。这是训练健康/可恢复性证据，尚不构成 MOT/RMOT/OVMOT 性能提升结论。证据：`docs/GROUNDJEV/REAL_NATIVE_COUPLING_PROGRESS_STEP1225_20261024.{json,md}`。

## 2026-10-21 — 双序列耦合训练 step48 审计：JEV 学会但 C2 坐标未学会

MOT20-03 与 MOT20-05 各 5 帧、6 个 transition，共恢复到 step48；冻结 G2 step17000，10 帧各一次视觉编码。teacher-forced Grounding loss 从 4.6592 降至 1.1808，两个序列的 typed JEV MATCH/ADMIT/Yes-No/ACTIVE 达到 1.0，但实际 greedy C2 框在两个序列平均 IoU 仍为 0.0000，C0 也为 0.0。部分 C2 输出合法但坐标接近冻结基线，部分出现前缀/解析失败。结论：离散 JEV 监督已可学习，单 token memory 到 G2 坐标生成的优化尚未打通；停止无界加步，转向 coordinate-token 级诊断/目标、wrong-memory 与多身份竞争。证据：`docs/GROUNDJEV_COUPLED/PHASE_C_REAL_G2_COUPLED_MULTISEQUENCE_STEP17000_20261021.{json,md}`。

## 2026-10-21 — G2 耦合模块跨序列诊断失败（MOT20-05）

将 MOT20-03 帧 82–85 上训练的耦合 checkpoint 原样用于 MOT20-05 帧 13–17，5 帧各一次 G2 视觉编码，评测期间无参数更新。C0/C2 平均 IoU 都是 0.0000；但 JEV 仍给出近乎满置信的 MATCH/Yes（P(ID) 约 1.0，P(Yes) 约 0.9995）。这定位为单视频记忆/几何过拟合和概率过度自信，否定跨序列泛化主张。保留为负证据，下一步扩大多视频训练并加入错误 memory、多身份竞争和校准。证据：`docs/GROUNDJEV_COUPLED/PHASE_C_REAL_G2_COUPLED_CROSS_SEQUENCE_MOT20_05_STEP17000_20261021.{json,md}`。

## 2026-10-21 — 真实 G2 单主干耦合 memory/JEV pilot（step 17000）

在冻结 G2 step17000 主干的条件下完成 24 步真实 GPU 更新，只训练 G2 输入空间记忆 token 投影、JEV 身份槽投影和结构化 typed readout（4,075,271 参数）；5 个 MOT20-03 TRAIN 帧各只做一次 G2 视觉编码。causal Grounding teacher-forcing loss 从 4.6700318654 降至 0.1691639870，JEV Yes/No 从 0/4 到 4/4。训练后 C0 无记忆平均框 IoU 为 0，C2 上一帧记忆平均 IoU 为 0.99636，同视频时序留出帧 86 的 C2 IoU 为 0.99270。该结果证明记忆 token 已影响真实 G2 生成前路径，但只覆盖单视频、单身份、24 步，不能作为独立 MOT/RMOT/OVMOT 泛化或 HOTA 提升；后续必须做多视频、错误 memory 和正式视频级评测。证据：`docs/GROUNDJEV_COUPLED/PHASE_C_REAL_COUPLED_MEMORY_JEV_TRAINING_STEP17000_20261021.{json,md}`。

## 2026-10-21 — G2 真实四帧 causal memory replay（step 17000）

假设：单主干 GroundJEV 可以在不加载第二个视觉模型的情况下，将上一帧状态插入下一帧 G2 causal prefill。使用 MOT20-03 TRAIN 帧 82–85 和目标 ID 27，每帧只做一次 G2 视觉编码；第 83–85 帧对比 C0 无记忆与 C2 上一帧记忆。结果：4 次视觉编码、3 个跨帧记忆 prefill 均完成，官方单框结构终止和 parser 均成功；未训练的 `IdentityMemoryTokenProjector` 没有改变任何恢复框，C0/C2 的平均标注 IoU 均为 0.0000。该结果保留为真实耦合边界的负证据，不能表述为 tracking/grounding 提升；下一步需要训练 memory/projector 与结构化 JEV 决策。证据：`docs/GROUNDJEV_COUPLED/PHASE_C_REAL_MULTIFRAME_REPLAY_STEP17000_20261021.{json,md}`。

## 2026-10-21 — real G2 causal memory prefill and structured-output repair

A real GPU7 Phase-C comparison loaded the protected G2 Clean adapter at step
17000 and encoded MOT20-03 TRAIN frames 82 and 83 once each. The current-frame
visual embedding was reused for memory-disabled, correct-previous-memory and
shuffled-memory language prefills; each condition used one causal prefill and no
independent VLM. The first attempt exposed a real dtype boundary failure: BF16
embeddings reached a restored FP32 LoRA projection outside the formal G2
autocast context. The smallest fix wrapped only the real G2 prefill in the same
CUDA BF16 autocast boundary. The next attempt showed unconstrained greedy
decoding could run past four coordinates without `<|box_end|>`; the runner now
uses the official single-box structural termination rule, constraining only
coordinate slots and emitting the boundary without synthesizing or reordering
coordinates. The repaired C0/C2 outputs all yielded one complete legal pair;
C0 and both C2 boxes were identical in this untrained-memory control. This is
real prefill/parser evidence and a neutral result, not identity-conditioned
grounding or tracking improvement. Evidence:
`docs/GROUNDJEV_COUPLED/PHASE_C_REAL_MEMORY_PREFILL_STEP17000_20261021.{json,md}`.

## 2026-10-21 — single-backbone native G2 context pilot at adapter step 17000

Following the new GroundJEV single-backbone direction, I ran a bounded native
G2 context readout pilot on physical GPU7 without touching the protected G2
Clean two-rank run. The official GroundAnything DLM was restored with the
step-17000 G2 adapter and encoded 16 MOT17-02-FRCNN TRAIN frames exactly once
per frame (16 native DLM/bridge forwards). Eight frames were used for training
and eight for a same-video temporal development split; the frozen G2 outputs
and teacher-forced previous-frame GT-matched region tokens fed the typed
GroundJEV readout. Eighty optimizer updates reduced identity accuracy from
0.052419 to 0.750000 on the train prefix and from 0.032258 to 0.536290 on the
temporal development split; admit accuracy changed from 0.653225 to 0.927419
and 0.596774 to 0.915322. These are bounded learnability controls only: no
HOTA/IDF1/IDSW, independent-video, or learned identity-conditioned box claim
is made, and teacher-forced causal memory remains a limitation. Evidence:
`docs/GROUNDJEV_COUPLED/PHASE_C_NATIVE_CONTEXT_PILOT_STEP17000_20261021.{json,md}`.

## 2026-10-21 — G2 Clean checkpoint step 15000 audited on CPU

The protected exact-once two-rank G2 run wrote and successfully reloaded `adapter-step015000.pt` and `latest_training_state.pt` on CPU with explicit `weights_only=False`. Both report step 15,000, world size 2, the registered Clean TRAIN manifest SHA, optimizer/scheduler/RNG state and the strict G2 format. The latest event is step 15002 with finite loss, gradient norm and memory; `final_receipt.json` is absent, so TAO/G1/held-out/three-task OneJev remain gated. This is checkpoint/recovery evidence only, not a metric claim. Evidence: `docs/GROUNDJEV/G2_CHECKPOINT_PROGRESS_20261021_STEP15000.{json,md}`.

## 2026-10-21 — G2 Clean checkpoint step 14500 audited on CPU

The protected exact-once two-rank G2 run wrote and successfully reloaded `adapter-step014500.pt` and `latest_training_state.pt` on CPU with explicit `weights_only=False`. Both report step 14,500, world size 2, the registered Clean TRAIN manifest SHA, optimizer/scheduler/RNG state and the strict G2 format. The latest event is step 14512 with finite loss, gradient norm and memory; `final_receipt.json` is absent, so TAO/G1/held-out/three-task OneJev remain gated. This is checkpoint/recovery evidence only, not a metric claim. Evidence: `docs/GROUNDJEV/G2_CHECKPOINT_PROGRESS_20261021_STEP14500.{json,md}`.

## 2026-10-21 — post-G2 downstream preflight

A read-only launch audit verified the registered TAO TRAIN eight-shard scope, runtime GroundAnything exporter/streamer, native OVMOT joiner, native three-task OneJev owner, G2 bundle owner, and G1 GPU5 owner. All required paths were present, the registered entrypoints compiled, and the wait owners were live. The formal G2 terminal receipt is still absent, so no TAO shard, optimizer, model service, or GPU was started. Evidence: `docs/GROUNDJEV/POST_G2_DOWNSTREAM_PREFLIGHT_20261021.{json,md}`.

## 2026-10-21 — G2 Clean live progress at step 14331

The protected two-rank G2 Clean run is still active on physical GPUs 3/5 and has a contiguous event stream through `14331/18,227`. The latest event at `2026-10-21 10:39:32 +0800` has finite loss `0.165262759198`, gradient norm `9.687554359436`, and peak event memory `12.828872 GiB`. The durable recovery pair remains `adapter-step014000.pt` and `latest_training_state.pt`; `final_receipt.json` and `adapter_projector_final.pt` are absent. Native TAO/OVMOT joining, three-task OneJev optimization, G1, and YesTrack still report zero optimizer updates under their dependency/resource gates. This is training-health evidence only and makes no coupled or tracking metric claim. Evidence: `docs/GROUNDJEV/G2_CHECKPOINT_PROGRESS_20261021_STEP14331.{json,md}`.

## 2026-10-21 — YesTrack permission failure repaired; coupled optimizer still gated

The first real YesTrack GPU7 owner reached its resource gate but failed before
creating the training process with `PermissionError(13, 'Permission denied')`.
The actionable cause was direct execution of the mode-0666 source wrapper
`run_with_resources.py`; the queue now invokes that wrapper through the pinned
OneJev Python environment. Compilation and the queue entrypoint check pass. The
terminal failed queue is preserved, and a new retry owner is waiting under a
separate queue root for GPU7 exclusivity because an unrelated `szgx` process
currently occupies GPU7. No process was stopped and no YesTrack metric is
claimed. The real G2/OneJev three-task coupled optimizer remains at zero updates
until the protected G2 Clean terminal receipt exists. Evidence:
`docs/GROUNDJEV/YESTRACK_PERMISSION_ROOT_CAUSE_20261021.{md,json}`.

## 2026-10-21 — G2 step 16977 snapshot and YesTrack owner recovery

The protected G2 Clean trainer remains live at step 16,977/18,227 on GPUs 3/5
with finite loss and gradient norm; its terminal receipt and terminal adapter are
still absent. The native TAO eight-shard owner, CPU OVMOT joiner, and native
three-task OneJev owner remain live under their explicit gates, with zero joint
optimizer updates. The GPU5 G1 owner is also waiting for the G2 terminal receipt.
The repaired YesTrack retry owner had exited without a failure receipt, so it was
resumed in a detached session and is now live at the GPU7 exclusivity gate. The
unrelated GPU7 compute process remains untouched. Evidence:
`docs/GROUNDJEV/STATUS_SNAPSHOT_20261021_STEP16977.{md,json}`.

## 2026-10-21 — YesTrack entered real GPU training after permission repair

The detached retry owner passed the GPU7 exclusivity gate and entered real
native YesTrack LoRA training using the OneJev step-600 checkpoint. The 726-row
train and 390-row validation manifests match their registered SHA-256 values;
the run is scheduled for 273 optimizer steps and had reached step 10 with finite
loss, slot accuracy and memory. This is training-start evidence only. The
terminal adapter, frozen-base/fine-tuned validation and YesTrack metrics are
still pending, and the formal three-task G2/OneJev optimizer remains at zero
updates. Evidence:
`docs/GROUNDJEV/YESTRACK_TRAINING_STARTED_20261021_STEP10.{md,json}`.

## 2026-10-21 — YesTrack paused to restore coupling priority

The latest direction is the GroundJEV single-backbone coupling experiment, so
the auxiliary YesTrack verifier was stopped cleanly after real GPU training
step 20/273 (160/2178 examples). Its partial log and queue state are preserved;
no adapter, validation receipt or metric was promoted. GPU7 is kept available
for the independent held-out gate after G2 terminal completion. G2, G0/G200/G1
artifacts and the native three-task queue were not modified. Evidence:
`docs/GROUNDJEV/YESTRACK_INTERRUPTED_FOR_COUPLING_PRIORITY_20261021.{md,json}`.

## 2026-10-21 — G2 Clean step 17000 checkpoint audited on CPU

The protected exact-once G2 run wrote adapter-step017000 and the full latest
recovery state. Both load on CPU with explicit `weights_only=False`, report
world size 2, the registered Clean TRAIN manifest SHA, strict training contract,
optimizer/scheduler/sampler state and all Python/NumPy/Torch/CUDA RNG state.
The live event stream is at step 17003/18227; the terminal receipt and terminal
adapter remain absent, so G1, TAO, native OVMOT, joint OneJev and held-out gates
remain closed. This is recovery evidence only. Evidence:
`docs/GROUNDJEV/G2_CHECKPOINT_PROGRESS_20261021_STEP17000.{md,json}`.

# LocateMOT Research Log

Concise experimental log (latest first).

## 2026-10-21 — dense-answer G2/JEV coupling completed and official decoder checked

在冻结 G2 step17000 上新增合法 dense-answer 耦合训练：DanceTrack-0001/0016 各 3 帧，每个 row 保留该帧全部 person boxes（7/6 个），训练 2 个 transition、2 optimizer steps；只训练 `IdentityBoxStateTokenProjector`、JEV memory projector 和 typed readout，共 4,077,327 参数，G2 视觉每帧一次。Grounding teacher-forcing loss 3.83649→3.83393，JEV structured loss 4.11511→1.14514。随后接入官方 `DLMInferenceEngine`：0001 的训练转移 recall@0.5 从 C0=0.1429 到 C2=0.2857（1→2 对比），但 0001 development 2→3 及 0016 两个转移 C2 输出均 malformed/0；因此这是共享主干接口和小样本可学习性证据，不是跟踪增益或泛化结论。MOT20 高密度 dense teacher forcing 曾在 GPU7 约 63.8 GiB 达到 OOM，已保留失败边界并改用低密度合法 clip 验证，未改 target-only 合同。证据：`docs/GROUNDJEV_COUPLED/PHASE_C_REAL_G2_DENSE_COUPLING_DANCE_STEP17000_20261021.md`，两个 receipt 位于 `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/phase_c_real_g2_dense_coupling_dance_step17000_v1/receipt.json` 与 `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/phase_c_real_g2_official_dlm_dense_coupled_dance_step17000_v1/receipt.json`。

## 2026-10-21 — official G2 DLM dense contract confirms one-box baseline before coupling

用冻结 G2 step17000 在 GPU4 通过 GroundAnything 官方 `DLMInferenceEngine` 的 `hierarchy_dynamic` 路径重跑真实 dense 解码；结构化终止开启且 `structured_max_coordinate_groups=0`，没有强制 single-box。MOT20-03/82（96 个 TRAIN person boxes）和 MOT20-05/13（247 个）各编码一次，官方 parser 均恢复 1 个合法框，但 IoU0.5 召回均为 0。模型在第一个 `<|box_end|>` 结束，说明先前 causal 单框现象也存在于官方混合 DLM 路径；这不是耦合收益，也不是 tracking 指标。下一步把持久身份状态插入同一官方 DLM prompt/clean stream，并在完整 GroundAnything answer 合同下训练和评测。证据：`docs/GROUNDJEV_COUPLED/PHASE_C_REAL_G2_OFFICIAL_DLM_DENSE_STEP17000_20261021.md`，receipt 位于 `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/phase_c_real_g2_official_dlm_dense_step17000_v1/receipt.json`。




## 2026-10-21 — G2 Clean checkpoint step 14000 audited on CPU

The protected exact-once two-rank G2 run wrote `adapter-step014000.pt` (SHA-256 `0cbf160b57a4feccb11a77f602cd8ebd39e79d93d79453cb78316a1d3ef6371e`) and `latest_training_state.pt` (SHA-256 `d7bbdfdecfeba789ff022d31800e1cd7d1fedb465e34e325ed738f33f66bae78`). Both load on CPU with explicit `weights_only=False`, report step 14,000, world size 2, exact-once contract, registered manifest SHA, optimizer/scheduler/RNG state and seed 20261019. The event stream is at step 14,021 with finite loss/gradient/memory; final receipt and terminal adapter remain absent, so dependent queues stay gated. This is checkpoint/recovery and training-health evidence only. Evidence: `docs/GROUNDJEV/G2_CHECKPOINT_PROGRESS_20261021_STEP14000.{json,md}`.

## 2026-10-21 — G2 Clean checkpoint step 13500 audited on CPU

The protected exact-once two-rank G2 run reached step `13,500/18,227` and wrote `adapter-step013500.pt` (SHA-256 `84a7e84a391d375cfeb3c74dea59bdb36f9a608fdf523132f0c6c9f6e1273c70`) plus `latest_training_state.pt` (SHA-256 `3a3a9a51d6f1e6b8f3e4a7d1ad399bd4864805c50dd0eab2eeb19c6def85626c`). Both were loaded successfully on CPU with explicit trusted-checkpoint `weights_only=False`; the initial PyTorch 2.6 default `weights_only=True` refusal was a validation-tool issue, not a checkpoint failure. Step, world size, exact-once contract and Clean TRAIN manifest SHA match. The final receipt and terminal adapter remain absent, so downstream gates stay closed. This is checkpoint/recovery evidence only, not a metric claim. Evidence: `docs/GROUNDJEV/G2_CHECKPOINT_PROGRESS_20261021_STEP13500.{json,md}`.

## 2026-10-21 — G2 Clean live progress at step 13361

The protected two-rank G2 Clean event stream advanced to `13,361/18,227` at `2026-10-21 07:35:07 +0800`. The latest loss (`0.2484929720`), gradient norm (`11.54479408`) and peak event memory (`12.828843 GiB`) are finite; the known parent PID remains live on GPUs 3/5. `final_receipt.json` and `adapter_projector_final.pt` are still absent, so G1, native TAO/OVMOT joining, three-task OneJev optimization and independent held-out evaluation remain gated. This is training-health evidence only and makes no G2 metric claim. Evidence: `docs/GROUNDJEV/G2_PROGRESS_LIVE_20261021_STEP13361.{json,md}`.

## 2026-10-21 — G2 event-integrity audit through step 12650

- Read the formal event log without altering the trainer. The completed audit
  cutoff contained 12,650 unique, contiguous steps from 1 through 12,650,
  with zero missing IDs and zero non-finite values across the recorded loss,
  gradient, memory and learning-rate fields.
- Twenty-six explicit checkpoint events were present through the durable
  step-12500 checkpoint; `final_receipt.json` remained absent. The live log
  then advanced to step 12651 with finite loss/gradient values.
- Decision: preserve the current two-rank run and its recovery state. This is
  numeric/resumability evidence, not a performance claim.
- Evidence: `docs/GROUNDJEV/G2_EVENT_INTEGRITY_AUDIT_20261021_STEP12650.md`.

## 2026-10-21 — downstream launch-gate audit

- Read the five G2-dependent entrypoints and confirmed their exact terminal
  contract: status `G2_EXACT_COVERAGE_TRAINING_COMPLETE`, step `18_227`, and
  clean manifest SHA `f682519a…06e71`; the bundle exporter also requires the
  terminal adapter file.
- Confirmed YesTrack is intentionally independent of G2 training and remains
  gated by real GPU7 exclusivity and its native split, so it cannot silently
  fall back to a CPU mock.
- No gate defect or premature launch was found. The static audit changed no
  runtime state and no benchmark claim. Evidence:
  `docs/GROUNDJEV/DOWNSTREAM_GATE_AUDIT_20261021.md`.

## 2026-10-21 — GroundJEV G2 step-12642 progress snapshot

- Evidence: the formal event log reached step `12642/18227` at
  `2026-10-21 05:17:02 +08:00`, with finite loss `0.17345873918384314` and
  finite gradient norm `10.642035484313965`.
- Gate state: `final_receipt.json` is absent. G1, deployment bundle,
  independent held-out, native TAO and YesTrack owners remain alive in their
  registered waiting states; no downstream metric or optimizer update is
  claimed.
- Decision: continue protecting the two-rank G2 run on physical GPUs 3/5 and
  preserve all checkpoints and historical G2-PL outputs.
- Boundary: this is runtime evidence, not a MOT/RMOT/OVMOT performance result.
  See `docs/GROUNDJEV/STATUS_SNAPSHOT_20261021_G2_STEP12642.md`.

## 2026-10-21 — GroundJEV G2 step-12633 progress snapshot

- Evidence: the formal event log advanced to step `12633/18227` at
  `2026-10-21 05:15:19 +08:00`, with finite loss `0.28367656865157187`, finite
  gradient norm `10.239775657653809`, and peak memory `12.8289 GiB`.
- Gate state: `final_receipt.json` is still absent. G1, deployment bundle,
  independent held-out, native TAO and YesTrack owners remain alive but in
  their registered waiting states; no downstream optimizer update or metric is
  claimed.
- Decision: continue protecting the two-rank G2 run on physical GPUs 3/5 and
  preserve all checkpoints and historical G2-PL results.
- Boundary: this is runtime evidence, not a MOT/RMOT/OVMOT performance result.
  See `docs/GROUNDJEV/STATUS_SNAPSHOT_20261021_G2_STEP12633.md`.

## 2026-10-21 — GroundJEV G2 step-12613 progress snapshot

- Hypothesis/contract: preserve the formal two-rank G2 Clean run and open all
  dependent G1, deployment-bundle, held-out, native-TAO and YesTrack gates only
  after the exact terminal receipt.
- Evidence: the authoritative event log reached step `12613/18227` with
  finite loss `0.2061017351`; `adapter-step012500.pt` and
  `latest_training_state.pt` are present, while `final_receipt.json` is absent.
- Decision: retain the live run and all waiting owners. No downstream optimizer,
  evaluation, label promotion, or historical G2-PL overwrite was performed.
- Boundary: this is a runtime progress record, not a MOT/RMOT/OVMOT metric or
  a claim that the final G2 detector exists. See
  `docs/GROUNDJEV/STATUS_SNAPSHOT_20261021_G2_STEP12613.md`.

## 2026-10-01 — latest R1 restoration completed

- Hypothesis: restore only the newest R1 execution boundary and its legal
  public-data/runtime inputs on `/data2/user/LocateMOT`; do not pretend that
  unavailable L69/L89E/R1 payloads were reproduced.
- Restored: official KITTI training images, Refer-KITTI V1, Refer-KITTI V2,
  MMDetection 3.3.0/GroundingDINO, BERT, and the direct R1 source dependency
  modules. V1 audit is complete. V2 has 157 legal expression-frame target
  references (11 repeated physical video/frame/ID pairs) pointing to IDs
  absent from corrected labels; missing frames are zero, and the anomaly is
  retained for the next adapter instead of silently filtered. The new public
  query manifest records every affected row.
- Runtime evidence: GroundingDINO loads the SHA-verified official checkpoint
  with 172,977,693 frozen parameters; one real KITTI frame/expression GPU
  inference returned 300 finite candidate boxes and finite scores. R1 imports
  and compile checks pass after path migration. The safe source loader also
  reads all 8,397 legal V1/V2 expression records from the restored
  per-expression layout without opening official-evaluation videos.
- Failure/limit: historical R1 checkpoints, L69 bank, dense indexes,
  language/visual/aligned caches and fixed manifest are unavailable. No old
  HOTA or score is claimed as reproduced; the old R1 sidecar is not extended
  with threshold/layer/learning-rate sweeps.
- Next action: construct a new legal public-data manifest/index and then a
  freshly generated candidate bank before any new correspondence training.
  The manifest is now complete. A 40-row midpoint-frame GroundingDINO probe
  found mean best IoU `.3188`, IoU@.25 `.6585`, IoU@.50 `.1463`, and mean best
  score `.0107` over 123 target boxes; the frame-zero attempt is preserved as
  a separate artifact. This supports a new bank audit but is not an RMOT
  score. Official-evaluation videos remain unread.

## 2026-08-19 Stage L11 — OVMOT temporal pseudo-track + KITTI front-end

- Hypothesis: unmatched DLA detections on expanded TAO-train OVMOT
  stream must receive high-precision temporal pseudo-track supervision
  instead of "every detection is NEW"; KITTI RMOT DetPr ~1% is a
  candidate front-end problem, fixable with category whitelist + NMS +
  query-conditioned CLIP top-k.
- Implementation:
  - Pseudo-track generator (forward + backward cycle consistency,
    appearance/motion/category filters, confidence gating) for all 500
    TAO train videos; sidecars in `outputs/l11/data/pseudo_tracks`.
  - Class-A GT upgraded from C-TAO base @0.5 IoU to base_and_novel
    @0.3 IoU (~10% -> ~28% coverage).
  - Trainer: no_unmatched_new, confidence-weighted identity/relevance
    losses, pseudo ids in the same slot/lifecycle machinery.
  - KITTI: whitelist + cross-NMS + top-30 -> ~15.3 cands/frame (was 50);
    CLIP top-12 calibration: query precision 10.2%, target recall 65%.
- Quality: pseudo same-ID precision 99.26% (8-video audit, target 90%);
  cycle pass 0.997; NEW rate 0.110 vs L10 collapse ~1.0.
- Running: 4-GPU repair training (from L9-ovmot, ~6-8 s/step),
  repaired-KITTI eval (L9-ovmot ckpt), 10-video OVMOT over-birth
  baseline: unique/rows 0.211 (L9), reused IDs 467.
- Pending: step-7000 OVMOT quick eval, full TAO TETA, KITTI TrackEval,
  L12 prompt seeding.

## 2026-08-20 continued

- Low-LR continuation (10000->11000, LR~6e-6): RMOT-Dance 32.49 ->
  34.05 HOTA; ordinary Macro 0.4855 -> 0.4924.  OVMOT full TETA on
  s11k running.
- Fresh-optimizer balance run (p_rmot 0.45/p_ovmot 0.20) REJECTED at
  step 1000 (RMOT 32.06 / ordinary 0.4663).
- L12 frozen prompt-seeded (DAVIS 2017, 10 multi-object videos):
  mask/point > box for identity robustness; joint fine-tune not
  launched (mixed signal).  Infrastructure complete.

## 2026-09-04 Stage L88 — RMOT-aware GroundingDINO LoRA adaptation

- Hypothesis: a zero-initialized rank-16/alpha-32 LoRA on the final two
  GroundingDINO fusion layers and decoder layer 0, with the frozen L86/L87
  sidecar and causal history, would improve held-out target-bag and
  expression-to-candidate correspondence without changing the L69 bank.
- Contract: label-free, distributed, and temporal regressions passed; full
  V1/V2 fit used seed `20260829`, 40 epochs, 2,640 optimizer steps, world 4,
  effective group batch 8, and 20 even checkpoints. Local MMCV required FP32
  rather than the registered BF16 autocast. Candidate deletion/truncation was
  false and all frozen-base/LoRA gradient and reload checks passed.
- Dev: all 20 checkpoints were scored on 138 video-disjoint groups; the fixed
  selection chose epoch20/Rule B before fixed validation. Checkpoint SHA is
  `7012706140cfa94278ce15bb8da3e2318eb3efcfb8d26d3b1dbc5206f1145538`.
- Fixed semantic validation: immutable L29 was
  `recall=.7333333, precision=.0830189, FP/frame=10.125,
  pred/positive=8.8333, hard=.9166667, multi=.8194444`. L88 candidate-only
  was `recall=.4193548, precision=.1830986, FP/frame=2.4167,
  pred/positive=2.2903, hard=.8461538, multi=.3333333`; final Rule B was
  `recall=.1935484, precision=.1363636, FP/frame=1.5833,
  pred/positive=1.4194, hard=.8461538, multi=.1666667`, with empty rate
  `.5416667` and inactive false acceptance `.5`. The semantic gate failed on
  recall and multi-positive preservation; the lower volume is not a fix.
- Internal TrackEval: selected Rule B produced HOTA `26.0914` (V1) and
  `20.2386` (V2), below L86 `29.1663/21.6467` and L87-A
  `28.5752/22.1300`. No screening or official-test labels were read.
- Preserved failures/fixes: first formal causal-key drift, sparse seqinfo,
  merge, device, and score-shape retries remain in new directories with their
  provenance; no old bank/checkpoint/production entrypoint was overwritten.
- Status: `STOPPED_PENDING_SUPERVISOR_REVIEW`; no L88 long continuation,
  screening, official test, TrackEval beyond internal validation, or
  ordinary MOT/OVMOT change is authorized. Unique next action is supervisor
  review of one new RMOT correspondence/proposal design.

## 2026-09-05 Stage L88C — corrected candidate-vs-NULL replay

- Hypothesis: the prior L88 semantic failure may include an implementation
  error in the candidate-vs-NULL gate. Corrected emission is
  `candidate_energy >= candidate_threshold` AND
  `candidate_energy-null_logit >= null_margin` AND
  `presence_logit >= presence_threshold`; candidate-only uses only the first
  condition. No training, backward, optimizer, checkpoint update, bank
  change, top-k/NMS or NULL suppression was performed.
- Implementation: refit all 20 even checkpoints on the registered B/R/P
  grids, replayed the corrected shortlist (epochs 8/20/40/30/4) over complete
  V1/V2 dev video, ran the corrected internal TrackEval matrix, froze epoch
  30 Rule B before fixed validation, and ran the fixed 16-calibration/
  24-validation replay. A first TrackEval provenance alias failure was
  preserved; the minimal `scope_key/full_video` alias fix was pushed in
  `c5a6685` and the corrected matrix is attempt2.
- Result: L88C corrected final on validation was
  `recall=.354839, precision=.207547, FP/frame=1.75,
  pred/positive=1.7097, hard=.846154, multi=.305556,
  empty=.375, inactive-FA=.666667`. Immutable L29 remains
  `.733333/.083019/10.125/8.8333/.916667/.819444` for
  recall/precision/FP/pred-positive/hard/multi. Candidate-only recall was
  `.419355`; the corrected NULL comparison removed additional positives.
- Internal dev TrackEval for the frozen epoch30/Rule B was V1
  HOTA/DetA/AssA `.296615/.216472/.408302` and V2
  `.265682/.156622/.451938`; these are dev-only, not screening or official
  results. The semantic gate failed on recall and multi-positive recall.
- Root cause: existing candidate scores have useful average pairwise ranking,
  but the lowest-scoring positives in multi-target bags do not survive the
  fixed emission boundary; V2 is worse. This is a correspondence and
  multi-positive recall failure, not a universal NULL-acceptance,
  coverage, finite/reload, or implementation-smoke failure.
- Preserved outputs: corrected replay, two TrackEval attempts, frozen
  selection, fixed semantic JSON/gate, and offline A–I diagnosis under
  `outputs/l88c/`. Code was pushed to branch
  `codex/l88c-candidate-null-corrected-replay-20260905` at `e8f5f62`.
- Status: `STOPPED_PENDING_SUPERVISOR_REVIEW`. No screening/official-test
  labels, HOTA claim beyond internal dev TrackEval, ordinary MOT/OVMOT/TAO
  change, or new L88C training was run. The only next action is the single
  supervisor approval request in `reports/l88c/NEXT_TEST_APPROVAL_REQUEST.md`.

## 2026-09-06 — L88C recoverable storage cleanup

- While the authorized zero-training L88C final-internal replay was running,
  its open paths were checked. Three superseded L1 intermediate LoRA
  checkpoint directories (`outputs/l1_c/checkpoints/lora/checkpoint-100/200/300`,
  about 8.0G each) were not open by any process and were moved, not deleted,
  to `/data2/usr_for_deadline/locatemot_cleanup_20260906/`.
- Frozen banks, formal reports, current L88 epoch30 checkpoint/cache, and all
  ordinary MOT/OVMOT assets were left in place. `/data1` free space increased
  from about 76G to 114G. The recoverable inventory is documented in
  `reports/l88c/L88C_RESOURCE_CLEANUP_20260906.md`.

## 2026-09-07 Stage L89 — query-conditioned set correspondence decoder

- Hypothesis: a fresh QSC-D head over the frozen L84/L87-A Z1 candidate states,
  with candidate self-attention followed by cross-attention to pure frozen
  GroundingDINO/BERT language tokens, could improve same-frame correspondence
  without changing the bank, detector, tracker, or ordinary MOT/OVMOT.
- Implementation: isolated worktree at
  `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89`, base
  `ece9fc5549a9210424eb127fb10430c3fed7b7ba`; two set layers, eight heads,
  FFN 1024, hidden 256, history 8, 4,169,829 trainable parameters. Pure
  language cache has 3,418 label-free entries. Contract smoke passed after
  preserving two package/field failures. No token/span or static/motion
  supervision is verified (`UNALIGNED`).
- Fit: seed `20260829`, one GPU, BF16, 40 epochs/2,640 optimizer steps,
  finite loss and nonzero gradients, all 20 even checkpoints plus final.
  Only L49 fit supervision was used; no screening or official-test labels.
- Dev/selection: 20 checkpoints scored over 138 legal dev groups; five-item
  shortlist and full-video B/R/P matrix selected epoch 4, Rule R before fixed
  validation. Checkpoint SHA is
  `5ab3cb344b73b320b34de7b4bb41622ce665ecb17c4d90c1999640a318e69aa8`.
- Fixed semantic: authoritative retry4 failed. L29 was
  `recall=.7333333, precision=.0830189, FP/frame=10.125,
  pred/positive=8.8333, hard=.9166667, multi=.8194444`; L89 was
  `recall=.8709677, precision=.0794118, FP/frame=13.0417,
  pred/positive=10.9677, hard=.6923077, multi=.9027778`, with inactive
  false acceptance `1.0`. Hard-negative ranking moved, but deployable volume,
  precision, and no-match behavior failed.
- Internal TrackEval: frozen epoch4/Rule R on V1 validation (86 sequences)
  gave HOTA `2.2906`, V2 validation (537 sequences) HOTA `0.5894`, versus
  L87-A `28.5752/22.1300`. Full-video post-hoc target-row recall was only
  `.0131/.0026` and inactive acceptance was `1.0`. This is internal
  validation TrackEval evidence, not screening or official-test evidence.
- Root cause judgment: query-conditioned set representation gave limited
  frame-level ranking signal but failed calibrated persistent emission and
  absence handling. The branch is stopped pending supervisor review; no L89
  continuation, screening, official test, or ordinary MOT/OVMOT change is
  authorized. All L89 outputs are preserved under `outputs/l89/` and the new
  source/reports are on branch `codex/l89-query-conditioned-set-correspondence-20260907`.

## 2026-09-08 Stage L89C — corrected candidate-vs-NULL replay

- Hypothesis/repair: L89 training and the registered loss use `candidate_energy`
  against `null_logit`, but the old L89 evaluation/deployment path emitted
  using `presence_logit - null_logit`. L89C changed only the evaluation,
  selection, inference, and TrackEval contract to the shared
  `corrected_emission_mask`: candidate threshold, candidate-minus-NULL margin,
  and presence threshold. No training, scorer forward, bank/cache rebuild, or
  model/ordinary-MOT change was made.
- Inputs/selection: immutable L89 raw dev records (9,960; SHA recorded in
  `outputs/l89c/dev/corrected_shortlist_attempt1/`), 20 existing checkpoints,
  and the fixed manifest SHA
  `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
  Corrected dev TrackEval evaluated 5 checkpoints × B/R/P (15 results), then
  froze epoch 2 / Rule R, checkpoint SHA
  `f8b175597ece8aad1f0ec3ae9d05c70e0759a0f7480dff9af6ea2619a2e3f08b`, with
  candidate/presence/NULL thresholds `-0.75/-1.0/0.0` before fixed validation.
- Fixed semantic evidence: authoritative retry2 has 40 ordered records (16
  calibration, 24 validation), with preselection forbidden-label fields
  absent. Validation is recall `.8387097`, precision `.0869565`, FP/frame
  `11.375`, predictions/positive `9.6452`, hard violation `.6923077`,
  multi-positive recall `.8333333`, inactive false acceptance `1.0`.
  The gate is `semantic_gate_fail`: hard/recall/precision/multi floors pass,
  but volume and inactive/no-match conditions fail.
- Internal evidence: frozen epoch-2/Rule-R full-video inference covered V1
  `0004,0018` (86 sequences) and V2 `0016,0017,0020` (537 sequences).
  Corrected TrackEval HOTA is V1 `2.244718%`, V2 `0.583059%`; no screening or
  official-test labels were read. These are internal validation-scope metrics,
  not a final RMOT result. Historical buggy L89 was `2.2906%/0.5894%` and
  L87-A was `28.5752%/22.1300%` on V1/V2.
- First actionable root cause: corrected candidate-vs-NULL accounting still
  leaves no-match acceptance universal and output volume too high. All rows,
  keys, finite checks, and frozen boundaries passed; the initial GPU OOM is
  preserved as an implementation/resource attempt and the exact CPU retry
  completed. Unique next action: supervisor review and authorization of one
  new RMOT branch explicitly addressing absence/volume calibration with
  candidate correspondence. Do not extend L89/L89C or tune thresholds.
- Status: `STOPPED_PENDING_SUPERVISOR_REVIEW`; `zero_training=true`,
  `screening_gt_used=false`, `official_test_labels_read=false`,
  `ordinary_mot_ovmot_touched=false`.

## 2026-09-08 Stage L89D — true full-video timeline repair

- Hypothesis/scope: repair only the L89C sparse-prediction versus dense-GT
  timeline mismatch. No training, model, loss, bank, tracker, threshold, or
  ordinary MOT/OVMOT change was made. New code is confined to `tools/l89d_*`
  and new evidence is under `reports/l89d/` and `outputs/l89d/`.
- Contract repair: native L69 `frame_ids` and frame pointers now drive every
  video timeline; all query/frame pairs and all candidate rows are scored.
  The multi-video query-map overwrite was fixed and covered by a targeted
  regression. Dev inference attempt3 covers 1,152,690/1,152,690 candidate
  pairs across 2,385 frame groups; internal attempt1 covers
  243,550/243,550 pairs across 1,844 frame groups. Both passed full-video and
  pair-completeness checks. The dev and internal dense Z1 supplements were
  built from frozen query-independent caches and finalized with no labels.
- Dev evidence: the valid 5-checkpoint × B/R/P TrackEval matrix selected
  epoch 8 / Rule P on the preregistered dev tuple. The frozen checkpoint is
  `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/train/joint40/checkpoint_l89_epoch008.pt`,
  SHA256 `28d4c35a189474d446a26fc096bbafbad77e1298429674383ce7ce8b14677d99`.
  The fixed 16-calibration/24-validation semantic replay then failed: recall
  `.5483871`, precision `.1517857`, FP/frame `3.9583333`,
  predictions/positive `3.6129032`, hard violation `.6923077`,
  multi-positive recall `.4861111`, inactive false acceptance `.6666667`.
  Recall and multi-positive floors failed; no screening/test labels were read.
- Internal evidence: fixed epoch 8 / Rule P true full-video TrackEval gives
  combined HOTA `23.8586078862%`, DetA `15.1294290119%`, AssA
  `37.9461647913%`, DetRe `57.2559333985%`, DetPr `16.9954368707%`, IDF1
  `18.6931722536%`, and IDSW `6337.5`; V1 HOTA `25.0798607896%`, V2 HOTA
  `22.6373549828%`. These are internal validation-scope TrackEval results,
  not screening, official-test, or final ordinary-RMOT evidence.
- Attempts preserved: dev ENOSPC, internal exit-137/OOM retries, the expected
  smoke full-video rejection, the selector directory-argument error, and the
  earlier multi-video GT-map mismatch. The final boundary guard passed after
  the path contract correction. The fixed manifest SHA remained
  `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
- Status: `L89D_COMPLETE_ZERO_TRAINING_TRUE_FULLVIDEO_REPAIR /
  semantic_gate_fail / STOPPED_PENDING_SUPERVISOR_REVIEW`.
  Flags: `zero_training=true`, `screening_gt_used=false`,
  `official_test_labels_read=false`, `ordinary_mot_ovmot_touched=false`.
  Unique next action: supervisor review and authorization of one separate
  absence/volume-calibration study; do not extend L89/L89C or alter the
  production MOT/OVMOT paths.

## 2026-09-09 Stage L89E — phase-consistent temporal/history replay

- Hypothesis/scope: L89D evaluated every checkpoint with temporal history
  enabled even though L89 training used Stage-S epochs 1–8 with temporal off
  and zero history. L89E changed only this replay contract: S epochs 2/4/6/8
  use `temporal_enabled=false` and zero history; T epochs 10–20 and J epochs
  22–40 use causal last-four history. No training, model, loss, bank, tracker,
  threshold science, UIDM, ordinary MOT or OVMOT source was changed.
- Implementation: isolated branch/worktree based on L89D commit
  `d0960e1d1679414765f79203bef444f0f9968928`; phase policy and wrappers are
  in `tools/l89e_*`. The required CPU equality check passed for S and T
  `L86ClipStore` history construction. The first replay failure was a
  `FrameExample`/`_clip_history` interface mismatch; the minimal adapter fix
  passed a targeted S/T causality regression. A subsequent historical-score
  merge check found that the immutable sparse file contains S+T/J rows; the
  wrapper now filters only the historical T/J epochs and preserves their
  values. Launch-path and shortlist argument mistakes were preserved as
  `INCOMPLETE`/operator evidence; no affected model run was treated as data.
- Stage-S evidence: fresh phase-correct replay of epochs 2/4/6/8 completed
  `4 x 498 = 1,992` records, all finite, complete and zero-history. Merged
  phase-consistent fit/dev pool contains 20 epochs and 9,960 records. The
  registered maximum-five fit/dev shortlist was frozen at epochs
  `8(S),20(T),40(J),32(J),4(S)`.
- Dev selection: full native dev replay retained `11,003,527/11,003,527`
  candidate rows for all five shortlist candidates. The 15-result TrackEval
  matrix selected epoch 4 / phase S / Rule B by the registered tuple; dev
  HOTA was `0.296512` (internal fit/dev selection evidence only).
- Fixed semantic evidence: the new 16-calibration/24-validation replay used
  the frozen epoch-4/S zero-history policy and Rule B. Immutable L29 remains
  recall/precision/FP-frame/pred-positive/hard/multi
  `.7333333/.0830189/10.125/8.8333/.9166667/.8194444`. L89E validation is
  recall/precision/FP-frame/pred-positive/hard/multi
  `.4516129/.1458333/3.4166667/3.0967742/.7692308/.3611111`, with inactive
  false acceptance `.8333333` and empty rate `.2083333`. The semantic gate
  fails recall and multi-positive floors despite lower volume and higher
  precision; no threshold rescue was performed.
- Internal TrackEval: after the frozen selection, exact internal timeline
  replay covered `243,550/243,550` pairs for V1 0004/0018 and V2 0016/0017/0020.
  Phase-consistent HOTA/DetA/AssA/DetRe/DetPr were V1
  `28.7628/18.7680/44.3927/80.3045/19.5312%` and V2
  `21.8385/13.3358/35.9990/45.1160/15.8179%`; IDSW was 1,997 and 9,915.
  These are internal validation-scope TrackEval results, not screening or
  official-test results.
- Root cause: correcting S-phase temporal/history evaluation does not
  restore expression correspondence; the fixed emission remains recall- and
  multi-positive-limited, with substantial inactive acceptance. Status is
  `L89E_COMPLETE_ZERO_TRAINING_PHASE_CONSISTENT_REPLAY /
  semantic_gate_fail / STOPPED_PENDING_SUPERVISOR_REVIEW`.
- Preserved outputs: Stage-S attempts 1–4, merge attempts 1–2, shortlist
  attempts 1–4, dev/internal full-video and TrackEval outputs, fixed semantic
  output, and all reports under `reports/l89e/` and `outputs/l89e/`. The fixed
  manifest SHA remains
  `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
  Flags remain `zero_training=true`, `screening_gt_used=false`,
  `official_test_labels_read=false`, `ordinary_mot_ovmot_touched=false`.
- The only next action is supervisor review of a separately authorized
  absence/volume-calibration study. Do not extend L89/L89C/L89E, alter the
  bank, or touch production MOT/OVMOT.

## R0A — safe target contract, dense native indexes, and implementation smoke — 2026-09-09

- Hypothesis: isolate only allowlisted V1/V2 target records, repair the
  frame-specific target contract, and construct native-frame R0 data without
  parsing forbidden official-eval payloads.
- Safe-source retry3 passed: lexical monolithic-JSON isolation deserialized
  only allowlisted video values and recorded
  `forbidden_payload_deserialized=false` and
  `official_test_labels_read=false`.  The exact 5,314 sparse fit rows matched
  the authoritative frame map with zero sentence/target/missing-query/invalid-
  frame mismatches.
- Dense native-frame index retry2 and its contract audit passed for V1/V2;
  all L69 rows and duplicate candidate indices were retained.  The geometry
  `image_hw` and visual-cache dataset/group-key contracts were corrected in
  R0-only files.  Retry1 dense/smoke failures remain preserved with their
  first trace; retry2 is authoritative.
- Real R0 contract smoke retry2 passed on one V1 fit frame: loss
  `3.555824041366577`, 108/108 finite nonzero trainable gradients, frozen
  GroundingDINO, no tracker optimizer parameters, strict reload max diff `0`,
  and temporary visual item removed.  This is implementation evidence only;
  no semantic, screening, official-test, HOTA, or production MOT/OVMOT result.
- Next action: commit/freeze the R0A safe code and construct the registered
  query-independent visual cache on `/data2`; no old asset or production
  entrypoint is modified.

## R0B — grouped DDP driver repair and formal legal evidence — 2026-09-09

- Hypothesis: the old Q1 query-at-a-time driver was the actionable R0 training
  contract bottleneck. A frame-centric grouped driver with complete candidate
  sets, Q=8, deterministic strata, and DDP accumulation should make the
  registered R0 experiment executable without changing the frozen bank,
  tracker, detector, or ordinary MOT/OVMOT.
- Preserved/disqualified Q1: PID 39571 was stopped gracefully; the old partial
  `outputs/r0/train/v1_formal_retry1/` remains historical and is not
  selection-eligible.
- Repairs: grouped Q8 schedule, video-local tiles, world4 DDP, accumulation 4,
  frozen detector/language inputs, complete candidate rows, present-uncovered
  masking, and resumable `locatemot-r0-tcgh-checkpoint-v2`. The legal GT
  materialization gap was repaired only in the R0 wrapper by recording and
  skipping missing annotation rows; frozen GT was not edited.
- Formal training: V1 completed 12 epochs/3,058 optimizer steps and V2
  completed 12 epochs/3,046 steps. Both were finite with nonzero gradients,
  strict reload, all four strata and both domains. Checkpoints 2/4/6/8/10/12
  are retained. Formal jobs used the free physical GPUs 4–7; physical GPUs
  0–3, which had unrelated processes, were not touched.
- Legal dev selection: V1 shortlist epochs 12/6 and V2 12/10; full dev
  TrackEval selected epoch12 for both. Dev HOTA was V1 22.8681% and V2
  19.5801% for the selected checkpoints.
- Internal validation-scope TrackEval: V1 0004/0018 HOTA 26.4582%, V2
  0016/0017/0020 HOTA 15.6097%. These are not screening or official-test
  results. No screening/official labels were read, and ordinary MOT/OVMOT,
  UIDM, L69 and production entrypoints were untouched.
- Outputs: training remains under `outputs/r0/`; large dev/inference/TrackEval
  products are under `/data2/usr_for_deadline/locatemot_r0b_*` because `/data1`
  was nearly full. No raw/dense debug cache or detector copy was created.
- Status: `STOPPED_PENDING_SUPERVISOR_REVIEW`. This is valid grouped R0
  implementation plus legal dev/internal TrackEval evidence, not final
  ordinary-RMOT completion. The only next action is supervisor review and a
  separately authorized follow-up.

## R1 — aligned track conditioning legal-dev contract — 2026-09-10

- Hypothesis/scope: test a zero-initialized RMOT-only residual conditioned on
  aligned frozen L89E Stage-S Z0/Z1/Z4 states and causal observation history;
  the anchor, L69 bank, ordinary MOT/OVMOT, UIDM and all production paths stay
  frozen. This is not final RMOT completion.
- Read-only legal-dev audit: frozen epoch-004 Rule-B anchor selection was
  rechecked. The epoch-004 presence-only miss is `3/400=.0075`, below the
  preregistered `.05` activation threshold, so the formal R1 config freezes
  `presence_residual=false`. Existing full-video artifacts have no candidate
  score arrays for a valid EXACT_INDEX_DEDUP replay; the conservative RAW
  policy is therefore frozen without claiming a dedup comparison.
- Preserved evidence: `outputs/r1/audit/legal_dev_selection_attempt1/` and
  `reports/r1/R1_LEGAL_DEV_SELECTION.md`; no training or new checkpoint was
  created by the audit. The next action is the bounded wiring/reload smoke;
  formal fit remains conditional on that implementation gate.

## R1 — formal fit and legal-development replay — 2026-09-12

- Hypothesis/scope: a zero-initialized aligned track-conditioning residual
  over frozen L89E stage states, causal observation history, language-directed
  R0 detail and complete L69 candidate sets could improve expression-to-track
  correspondence without changing the anchor, bank, tracker or ordinary
  MOT/OVMOT paths.
- Formal fit completed in isolated worktree
  `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_R1`: six epochs, 2,250
  optimizer steps, all finite with nonzero gradients, strict reload for eight
  V1/V2 checkpoints, frozen anchor, and no persistent raw/dense debug cache.
- Prediction-only replays for legal-dev V1 videos `0008/0010/0020` and V2
  videos `0000/0008/0009` completed before legal GT was opened. The aggregate
  contains 2,385 groups and eight internal TrackEval results. Registered
  selection chose epoch 4 separately for V1 and V2. Selected HOTA was
  `24.2345%` (V1) and `23.6542%` (V2), versus the L89E anchor's `28.7628%`
  and `21.8385%`; this is not a stable breakthrough and remains legal-dev
  evidence only.
- Fixed 16-calibration/24-validation diagnostic retained all 2,435 L69 rows
  and 40 keys. R1 candidate-only recall/precision/FP-frame/pred-positive/
  hard/multi were `0.1000/0.1842/0.7750/0.5429/0.7727/0.1818`; the frozen
  Rule-B replay emitted zero rows for both anchor and R1 on this slice. The
  lower volume is a recall/multi-positive collapse, not a semantic repair;
  no threshold or NULL rescue was attempted.
- Status: `R1_NO_BREAKTHROUGH / STOPPED_PENDING_SUPERVISOR_REVIEW`.
  Outputs and failed technical attempts are preserved under `outputs/r1/`
  and `reports/r1/`. Flags remain
  `screening_gt_used=false`, `official_test_labels_read=false`, and
  `ordinary_mot_ovmot_touched=false`. The only next action is supervisor
  review and explicit authorization of one separate follow-up; no automatic
  continuation, screening, official test, or production merge is allowed.

## R1 — fixed semantic diagnostic closeout — 2026-09-12

- The fixed-slice evaluator used the native L62 order and attached labels only
  after row/feature construction and scoring. It used the frozen Rule-B values
  `candidate_threshold=1.0`, `presence_threshold=0.5`, `null_margin=0.0` and
  did not fit or tune a new threshold.
- This is a diagnostic, not a preregistered semantic gate and not HOTA. The
  authoritative files are in `outputs/r1/eval/fixed_semantic_attempt1/`; the
  corresponding report is `reports/r1/R1_FIXED_SEMANTIC_DIAGNOSTIC.md`.
# 2026-10-01 — new-server restoration started

- Hypothesis: restoring public V1/V2 data and exact base inference dependencies enables a new, attributable correspondence experiment; extending the failed R1 sidecar is unjustified.
- Evidence: fetched all remote branches; latest remains R1 `99090c8`. Read chronological historical endings and R1 reports: V1 legal-dev regression, limited V2 gain, fixed-slice score mismatch/output collapse. Old trained checkpoints are unavailable.
- Changes: started resumable official KITTI S3 **direct** download (15,813,146,295 bytes); downloaded and CRC-validated V1 expression/labels archives; isolated local data-tools environment. Full inventory/short plan: `reports/RESTORE_STATUS_20261001.md`.
- Failure/repair: shell sandbox lacks bubblewrap; execution requires tool escalation. Standard venv lacks ensurepip; local virtualenv bootstrap succeeded. Neither is a scientific result.
- Results: public V1 annotations recovered; KITTI/V2/runtime/project-generated assets pending. No historical checkpoint/bank reproduction claimed. Keep recovery scripts and new provenance; preserve migration AGENTS.md edit.

## LocateMOT-U U0 — unified specification and reference audit — 2026-10-01

- Hypothesis: a single specification-conditioned foundation can absorb the
  reusable persistent-query, temporal-language, open-vocabulary and reliability
  ideas from the ten requested repositories, provided the legal video boundary
  and one-checkpoint/shared-state contract are frozen before training.
- Code: cloned and shallow-audited MOTR, TransRMOT, TempRMOT, DKGTrack,
  FlexHook, iKUN, OVTR, COVTrack, ReferDINO and Open-GroundingDino; created
  `UnifiedSpecTrack` U0 shape skeleton, shared legal-scope guards, task/sample
  contracts, loss/runtime interfaces, parameter-sharing audit and reproducible
  audit/sanity tools. Exact source recipes and commit IDs are in
  `reports/unified/OPEN_SOURCE_TRAINING_AUDIT.md` and
  `outputs/unified/open_source_audit.json`.
- Data/protocol: froze V1/V2 fit/calibration/validation splits and reserved
  V1 `0005,0011,0013` plus V2 `0005,0011,0013,0019`; retained all 157 V2
  source target-ID-missing rows as unlabeled records. No official-test labels
  or screening ground truth were opened.
- Checkpoint/GPU: no LocateMOT-U checkpoint and no training run yet; GPU
  allocation was zero. External sanity was compile/help only (about 23 s wall
  time): 10/10 selected source files compiled, 1/10 entrypoints reached help,
  and the nine remaining failures are recorded dependency/version findings.
- Results: skeleton CPU forward finite with score shape `[1,100]`; legal guard
  smoke passed; skeleton non-head parameter fraction `0.9998259786` (not a
  final-model claim). No benchmark metric was produced. U0 scope/audit/sanity
  passed; U1 MM-GroundingDINO-B acquisition and proposal/expression gates are
  next.

## LocateMOT-U U1 — Swin-B foundation acquisition and adapter smoke — 2026-10-01

- Hypothesis: the audited GroundingDINO Swin-B CogCoOR weights can serve as the
  shared visual/specification foundation after selecting the checkpoint format
  compatible with the local MMDetection GroundingDINO implementation.
- Assets: acquired and hashed both the original release
  (`46270f7a822e6906b655b729c90613e48929d0f2bb8b9b76fd10a856f3ac6ab7`,
  938,057,991 bytes) and the OpenMMLab conversion
  (`55949c9c0f46339a73b415334765615d491ee6ed739ed3f568142b7fc5581143`,
  935,971,702 bytes). The MMDetection asset is the runtime candidate because
  its key layout matches the adapter; provenance is in
  `outputs/unified/protocol/foundation_asset.json`.
- Code/smoke: built the local Swin-B model (232,996,763 parameters), loaded the
  converted checkpoint with 8 expected initialization keys and 3 BERT metadata
  keys reported as non-fatal mismatch, then ran one CUDA label-free forward on
  legal V1 video `0004` frame `000063.png` using `car . pedestrian . cyclist .`.
- Results: 300 finite predictions, finite boxes and finite scores;
  `outputs/unified/u1_foundation_smoke.json` records the exact config,
  checkpoint hash, device and legal frame. This is a compatibility smoke only;
  no proposal recall, expression grounding metric or tracking result is
  claimed. Official-test labels and screening ground truth remain unread.
- Checkpoint/GPU/walltime: no LocateMOT-U checkpoint; one A800-equivalent GPU;
  model build/forward about 9 s. Next gate is label-free proposal and
  expression grounding on legal development frames.

## LocateMOT-U U1 — proposal gate diagnostic — 2026-10-01

- Hypothesis: the loaded Swin-B foundation would provide sufficiently localized
  open-vocabulary candidates for the shared semantic/tracking path on a fixed
  legal development slice.
- Protocol: 16 V1/V2 calibration/validation expression-frame rows, 50 target
  boxes, 300 candidates per row; model inference completed before each
  corresponding label file was opened. No threshold fitting, training,
  official-test labels or screening ground truth.
- Result after correcting the MMDetection BGR input contract: best-IoU
  `>=0.25` was `0.5400`, best-IoU `>=0.50` was `0.0400`, mean best IoU
  `0.2756`, and mean score at best IoU `0.01312`. Boxes and scores were finite.
  This remains below a usable proposal gate, so semantic-head/tracking training
  from this stream is disallowed. The initial RGB/BGR-swapped diagnostic was
  discarded and is not used as evidence.
- Diagnosis/next change: inspect prompt tokenization/category assignment, box
  decoding and proposal selection on this same legal slice. Preserve the split,
  rows, checkpoint and labels; do not tune a threshold or delete hard cases.
  Authoritative output: `outputs/unified/u1_proposal_gate.json` and
  `reports/unified/U1_PROPOSAL_GATE.md`.
- Selection-only check: retaining all 900 decoder queries raised IoU@0.50 to
  `0.1000` and mean best IoU to `0.3341`, so top-k selection contributes but
  does not repair localization. The top-900 output is retained separately and
  is not used as a training bank.
- A fixed class prompt (`car . pedestrian . cyclist .`) with all 900 queries
  reached IoU@0.25 `0.9000`, IoU@0.50 `0.1800`, mean best IoU `0.3635`; this is
  a useful direction for the repair but still below the stage gate and remains
  excluded from training. Output:
  `outputs/unified/u1_proposal_gate_generic_classes_top900.json`.

## LocateMOT-U U1.3 — proposal-selection repair diagnostics finalized — 2026-10-01

- Change: made the canonical 300-query output reproducible with explicit
  `max_per_img` and prompt metadata, and synchronized the report/status values
  with the corrected BGR run. Re-ran the 300-query baseline, the 900-query
  selection diagnostic, and the fixed class prompt (`car . pedestrian .
  cyclist .`) diagnostic on the identical legal 16-row slice.
- Results: the baseline stayed at IoU@0.25 `0.5400`, IoU@0.50 `0.0400`, and
  mean best IoU `0.2756`; retaining 900 queries reached `0.1000` at IoU@0.50;
  the fixed class prompt with 900 queries reached `0.1800` at IoU@0.50 and
  `0.9000` at IoU@0.25. All boxes/scores were finite, no rows were skipped,
  and no official-test labels or screening ground truth were read.
- Interpretation: prompt/category assignment and top-k selection are concrete
  repair directions, but the proposal gate remains below the stage criterion.
  No semantic-head/tracking training bank was created and no training was
  launched. The next experiment must keep this split/checkpoint fixed while
  inspecting prompt tokenization, category assignment and box decoding.
- Verification: unified modules/tools compile; CPU skeleton smoke is finite;
  parameter-sharing audit remains skeleton-only with non-head fraction
  `0.9998259786`; legal guard accepts V1 `0004` and rejects reserved `0013`.

## LocateMOT-U U0R/U1R measurement-contract repair — 2026-10-01

- Scope: resumed from `014b8a5ec81bb83c5993b56da171089f8a440a11` on
  `codex/u0r-u1r-foundation-repair-20261001`; no U2 work and no training was
  launched. The project root was verified as `/data2/user/LocateMOT`.
- Reproducibility hypothesis: a fresh checkout should import the unified data…140051 tokens truncated…rted before the new clients were launched, so OneJev cannot start while either client is unfinished.
- No partial rows are metrics. Full-frame order, the official scorer, model/manifest hashes and terminal receipts are still required before any result is promoted.

## 2026-10-23 — OneJev gate switched to the real native three-task course
- Audited `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/training_data/real_three_task_joint_native_v1/train.sources.json`: `native_ovmot_training_rows=true`, `ovmot_training_mode=native`, 131,618 rows total (131,214 MOT identity, 346 RMOT membership/presence, 58 native TAO category group/leaf rows). The native OVMOT component is a registered TRAIN subset sourced from the preserved candidate/annotation join; it is not a full-TAO claim and has zero optimizer updates so far.
- Patched `queue_real_three_task_native.py` to use the unified held-out finalizer, accept a fresh queue/checkpoint root and explicit GPU list, and require only the measured free-memory floor while recording foreign PIDs. The new course uses GPU5/7 after the held-out services terminate; GPU4/6 foreign jobs remain untouched.
- Patched `orchestrate_heldout_finalizers_and_onejev.py` to invoke the fresh native queue after all 27 owners and three finalizers complete. The historical teacher-control/step-3000 queue remains preserved and is not invoked.
- Compile/import checks passed and the native manifest loaded 131,618 rows in the source trainer preflight before CPU token-length planning was stopped to avoid occupying the host for an extended dry run. No optimizer or GPU training update was made by this preflight.

## 2026-10-23 — held-out owners advance under the expanded shared-service batch
- A current progress snapshot found G1 MOT20 at 1,413/1,657 rows through physical frame 3,072 and G200 MOT20 at 1,206/1,657 rows through physical frame 2,864. Recent-20 medians were about 45.1 s and 34.2 s per frame, giving rough remaining estimates of 3.1 h and 4.3 h.
- The two added shared-service scopes remain active: G1 RMOT 245/800, G200 RMOT 26/800, G1 OVMOT 2/40, and dedicated G200 OVMOT 5/40. These are row-count observations only; no official metric is promoted from partial output.
- The unified orchestrator remains in `WAITING_ALL_REGISTERED_HELDOUT_OWNERS`; native OneJev has not launched. The live MOT20 receipt was refreshed with the current counts and timing estimates.

## 2026-10-23 — two additional cross-model held-out controls launched
- User requested two more tasks in parallel. To avoid increasing `sglang::scheduler` count, both jobs reuse validated shared services: G1/GPU5 on `http://127.0.0.1:8240` and G200/GPU7 on `http://127.0.0.1:8238`.
- `g1_dancetrack_shared_gpu5_v1`: DanceTrack `dancetrack0004` validation, 1,203 frames, G1 bundle `GroundAnything-DLM-TrackProjG1-GPU5-V2`, official scoring parser.
- `g200_animaltrack_shared_gpu7_v1`: AnimalTrack `chicken_1` test, 306 frames, G200 bundle `GroundAnything-DLM-TrackProj200`, official scoring parser.
- Both scopes reuse the existing independent manifests from `heldout_inputs/g2_domain_v1`; outputs are isolated and use the same input/parser/decoder/tracker/scorer contract. GPU6 was left untouched because it has another user's active workload.
- Launch receipt: `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/heldout/parallel_two_shared_launch_20261023.json`; runner PIDs at launch were 3511892 and 3511893. No new SGLang service was created. Initial wrapper failure caused by precreating non-empty output dirs was cleaned without inference; relaunch uses directory-external launcher logs.
- Metrics are intentionally not claimed until each owner writes a terminal receipt and official metrics.

## 2026-10-23 — recoverable retries queued after shared-service failures
- The first G1 DanceTrack shared control reached a terminal failure at frame 1 because GPU5/8240 returned HTTP 503 while the existing MOT/RMOT/OVMOT clients occupied the service queue. The failed receipt at `heldout/g1_dancetrack_shared_gpu5_v1/failure_receipt.json` is preserved and is not treated as a metric.
- Started durable `queue_g1_dancetrack_shared_retry.py` (PID 3517597). It waits for the two extra G1 shared clients to reach terminal receipts, rechecks the existing service, and retries the identical 1,203-frame scope under `g1_dancetrack_shared_gpu5_v2` without creating a scheduler.
- The previous G200 OVMOT dedicated attempt is also preserved as failed: its log shows 500+ second OVMOT frames and a request timeout at frame 146 while GPU7 MOT/RMOT work was still active. Started durable `queue_g200_ovmot_dedicated_retry.py` (PID 3518544), which waits for G200 MOT20/RMOT termination and a 32-GiB free-memory gate, then starts one dedicated service on port 8243 and resumes the preserved causal prefix into `g200_ovmot_dedicated_gpu7_v2`.
- No existing process was killed, no failed output was overwritten, and no metric claim was made from either failed run.

## 2026-10-23 — OVMOT request-timeout fix
- The G200 OVMOT retry needs a bounded timeout larger than the previous 600-second hard-coded HTTP limit. Added an explicit `--grounding-request-timeout` option (default remains 600 seconds) to the tracked `run_video.py`/`run_g2_parallel_scope.py` path and set the queued dedicated G200 retry to 900 seconds.
- Applied the same timeout-only change to the runtime copy under `/data2/user/LocateMOT_GroundJEV`; the runtime's existing repaired parser implementation was preserved. Runtime `--help`, import, and constructor checks passed. Existing active clients keep their original default and were not restarted.

## 2026-10-23 — held-out gate extended to new parallel controls
- The durable orchestrator was restarted once after a code change (the old process was verified as our PID and was still in the pre-finalizer waiting stage). Its gate now includes the running G200 AnimalTrack owner and both retry queue state files, so native OneJev cannot claim GPU5/7 while the newly requested controls or their recoverable retries are still active.
- The restarted orchestrator is PID 3525198 and remains in `WAITING_ALL_REGISTERED_HELDOUT_OWNERS`; no finalizer or OneJev process was started by this restart.

## 2026-10-23 — receipt collector repair
- Audited `scripts/groundjev/collect_g2_heldout_receipts.py` before the finalizers can run. Two default owner entries referenced an undefined `RUNTIME` symbol, and one repaired OVMOT owner was duplicated. Replaced the stale symbol with `RUNTIME_ROOT` and removed the duplicate; the collector now compiles cleanly and preserves one record per owner. The expensive full model-hash smoke invocation was not used as an acceptance result.

## 2026-10-23 — reproducible collector for the two extra controls
- Added `scripts/groundjev/collect_extra_shared_control_receipts.py`, a non-blocking collector for the requested G1 DanceTrack/G200 AnimalTrack controls and their retry receipts. It records model/manifest hashes and official metrics only when a complete result block exists; queue states, failures, and partial outputs remain status evidence. A one-owner queue-state smoke check passed (`accepted_scope_count=0`, as required).

## 2026-10-23 — automatic post-run collection queued
- Started durable `queue_collect_extra_shared_controls.py` (PID 3535083). It waits for the G200 AnimalTrack owner and both G1/G200 retry queue states to become terminal, then writes `extra_shared_controls_final_receipt.json` through the non-blocking collector. It does not start a GPU service and does not affect the OneJev gate.

## 2026-10-23 — stopped accidental full collector smoke processes
- A prior collector smoke command that hashed the full model for every owner was still running (PIDs 3528263 and 3529284) and had never produced a receipt. Both command lines were verified as this project's `collect_g2_heldout_receipts.py` processes and were terminated explicitly; all GPU inference services, retry queues, and the durable extra-control collector were left untouched.
- The collector fix remains compile-tested; the full hash smoke is not accepted as an experiment result. Future collection is performed only by the bounded extra-control queue after its owners reach terminal states.

## 2026-10-23 — reclaimed orphaned failed G200 OVMOT service
- The failed `g200_ovmot_dedicated_gpu7_v1` receipt still had an orphaned GroundAnything service group on port 8242. Its command line was verified against the failed owner/model/output before cleanup; only that process group was terminated. The active G200 MOT/RMOT service on port 8238, G1 services, new AnimalTrack evaluation, retry queues, and foreign processes were not touched. A follow-up process check found no remaining 8242/failed-owner service.

## 2026-10-23 — reclaimed idle GPU3 G0 scheduler
- A single blocking status wait returned without a new terminal scope; the authoritative owner states remained six running held-out evaluations plus three durable wait queues, with no reported failure or restart.
- Audited the remaining GPU3 `GroundAnything-DLM` service on port 8223. Its owner was already terminal `G2_CLEAN_FULL_INDEPENDENT_HELDOUT_FAILED`, no `run_video.py` client or live owner existed, and the scheduler was an orphan reparented to PID 1.
- Terminated only that verified project-owned orphan. The active G1/G200 shared services (ports 8240/8238), evaluation clients, retry queues, orchestrator, and foreign GPU workloads were left untouched; no metric was promoted from the failed G0 receipt.

## 2026-10-23 — held-out owners continue after blocking wait
- A single 180-second blocking wait returned without a terminal owner transition; a follow-up authoritative audit confirmed all six active evaluation clients and both retry waiters remain live, with no new failure receipt.
- Current output progress from the legal manifests is G1 MOT20 1,443/1,657 frame decisions, G200 MOT20 1,256/1,657, G1 RMOT 278/800, G200 RMOT 80/800, G1 OVMOT 7/40, and G200 AnimalTrack 43/306. These are partial-row counts only; no official metric is promoted until the full frame contract and scorer receipt pass.
- The GPU3 orphan G0 service has been reclaimed; exactly two project-owned SGLang schedulers remain (G1/GPU5 port 8240 and G200/GPU7 port 8238). GPU6 and foreign workloads remain untouched. The unified orchestrator is still waiting for all registered owners before finalizers and native three-task OneJev.


## 2026-10-23 — held-out inference continues with advancing outputs
- A second 180-second blocking wait returned no terminal owner transition, but output files advanced and all six evaluation clients remained alive; this is progress rather than a restart or hang.
- The latest frame-decision counts are G1/G200 MOT20 1,448/1,259 of 1,657, G1/G200 RMOT 283/86 of 800, G1 OVMOT 8 of 40, and G200 AnimalTrack 49 of 306. These counts are partial evidence only and remain excluded from official metrics until terminal receipts pass.
- The two shared GroundAnything services and six clients are still the only project inference handles. The orchestrator remains gated; native MOT/RMOT/OVMOT OneJev optimization has not started early.


## 2026-10-23 — G1 OVMOT slow-frame boundary confirmed, not a hang
- The latest blocking wait had no terminal owner transition, while the other held-out outputs continued advancing. A focused log audit of `g1_ovmot_shared_gpu5_v1` found the client alive in `run_video.py` and the shared GPU5 service processing the next TAO sample (physical frame 233, the ninth sampled frame in the 40-frame manifest).
- The completed TAO samples take roughly 315–378 seconds each because the 1,203-class vocabulary is requested in chunks. Several raw outputs fail the canonical wire schema (duplicate coordinate tuples or a missing box-start token); the repaired official parser records the rejection and the contract emits an empty prediction set, without guessing boxes or claiming a metric.
- This is an observed throughput/parser limitation, not an infrastructure hang. The OVMOT client remains under the same two-service limit and is not restarted or duplicated.


## 2026-10-23 — held-out batch continues after extended wait
- A 240-second blocking wait produced no terminal owner transition, but post-wait file validation showed all six clients healthy and advancing; no retry or service was restarted.
- Current frame-decision counts are G1/G200 MOT20 1,455/1,274 of 1,657, G1/G200 RMOT 294/103 of 800, G1 OVMOT 9 of 40 sampled frames (the next TAO request is in flight), and G200 AnimalTrack 66 of 306.
- Exactly two project SGLang services and six evaluation clients remain. Partial rows, including OVMOT rejected outputs, remain excluded from official metrics; the orchestrator continues to gate finalizers and native three-task OneJev.


## 2026-10-23 — held-out inference remains healthy
- A further 180-second blocking wait returned no terminal owner transition; post-wait validation showed all active scopes continued writing and no failure receipt appeared.
- Current frame-decision counts are G1/G200 MOT20 1,459/1,280 of 1,657, G1/G200 RMOT 298/110 of 800, G1 OVMOT 10 of 40 sampled frames, and G200 AnimalTrack 73 of 306.
- The two shared services remain the only SGLang schedulers. OVMOT continues at its measured multi-minute per-sample rate; no process was duplicated or restarted, and the OneJev gate remains closed until terminal held-out receipts are available.


## 2026-10-23 — held-out scopes continue writing under the two-service cap
- A further blocking wait returned no terminal owner transition, but authoritative output validation showed all six active clients advancing with no failure receipt.
- Current frame-decision counts are G1/G200 MOT20 1,464/1,286 of 1,657, G1/G200 RMOT 303/117 of 800, G1 OVMOT 11 of 40 sampled frames, and G200 AnimalTrack 80 of 306.
- The two GroundAnything services remain the only project schedulers. No partial result is scored or promoted; the orchestrator still holds native three-task OneJev until terminal held-out contracts are satisfied.


## 2026-10-23 — held-out progress reaches the next TAO sample
- The next blocking wait returned no terminal owner transition, while all non-OVMOT scopes continued advancing: G1/G200 MOT20 1,469/1,293 of 1,657, G1/G200 RMOT 307/124 of 800, and G200 AnimalTrack 87/306.
- G1 OVMOT completed through sampled physical frame 291 and is processing frame 320 (the twelfth sampled frame of 40); its multi-minute latency is expected from the 1,203-class request and is not a hang.
- The two-service limit remains intact; no third SGLang scheduler or duplicate OVMOT client was started, and OneJev remains gated on terminal receipts.


## 2026-10-23 — held-out clients continue without service expansion
- Another 180-second blocking wait returned no terminal state change, while output files for all active clients advanced and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,474/1,300 of 1,657, G1/G200 RMOT 313/132 of 800, G1 OVMOT 12 of 40 sampled frames, and G200 AnimalTrack 95 of 306.
- The two existing GroundAnything services remain sufficient for the active load and no additional scheduler was created. The strict no-partial-metric and OneJev held-out gate remain in force.


## 2026-10-23 — held-out inference advances under stable resource envelope
- A 180-second blocking wait returned no terminal state change; the follow-up audit found all six clients healthy with advancing output files and no failure receipt.
- Current frame decisions are G1/G200 MOT20 1,477/1,305 of 1,657, G1/G200 RMOT 318/139 of 800, G1 OVMOT 13 of 40 sampled frames, and G200 AnimalTrack 101 of 306.
- The two existing GroundAnything schedulers remain the complete project service set. The same parser/decoder/tracker/scorer contract and no-partial-metric rule remain active while OneJev waits for terminal held-out receipts.


## 2026-10-23 — held-out batch reaches OVMOT sample 14
- The latest blocking wait returned no terminal owner transition, but the active scopes continued writing with no failure receipt.
- Current frame decisions are G1/G200 MOT20 1,482/1,312 of 1,657, G1/G200 RMOT 323/147 of 800, and G200 AnimalTrack 109 of 306. G1 OVMOT completed physical frame 349 and is processing frame 378, the fourteenth sampled frame of 40.
- The two-service resource cap remains intact. OVMOT parser rejections remain explicit empty predictions under the official contract; no partial result is scored or promoted and OneJev remains gated.


## 2026-10-23 — held-out clients continue toward terminal manifests
- The latest blocking wait returned no terminal transition, but all six clients continued to advance and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,488/1,318 of 1,657, G1/G200 RMOT 328/154 of 800, G1 OVMOT 14 of 40 sampled frames, and G200 AnimalTrack 116 of 306.
- The two existing schedulers remain the complete project service set. No partial rows are scored, and the OneJev launch gate remains closed until the registered held-out receipts and finalizers complete.


## 2026-10-23 — held-out batch continues with no failures
- The latest 180-second blocking wait returned no terminal owner transition, but all active clients continued to write and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,491/1,325 of 1,657, G1/G200 RMOT 333/161 of 800, G1 OVMOT 15 of 40 sampled frames, and G200 AnimalTrack 123 of 306.
- Both existing GroundAnything services remain healthy and no scheduler was added. Partial outputs remain excluded from official metrics; the OneJev gate stays closed.


## 2026-10-23 — held-out batch reaches OVMOT sample 16
- The latest blocking wait produced no terminal owner transition, but all non-OVMOT scopes advanced with no failure receipt: G1/G200 MOT20 1,492/1,332 of 1,657, G1/G200 RMOT 335/168 of 800, and G200 AnimalTrack 130/306.
- G1 OVMOT completed physical frame 407 and is processing frame 436, the sixteenth sampled frame of 40. The client and shared service remain live; the multi-minute request latency is expected under the 1,203-class vocabulary contract.
- The two-scheduler cap and no-partial-metric policy remain unchanged; OneJev is still gated on terminal held-out receipts.


## 2026-10-23 — held-out batch continues past MOT20 1.3k frames
- The latest blocking wait returned no terminal transition, but all six clients continued writing and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,495/1,339 of 1,657, G1/G200 RMOT 339/175 of 800, G1 OVMOT 16 of 40 sampled frames, and G200 AnimalTrack 137 of 306.
- The two GroundAnything schedulers remain the complete project service set. No partial metrics are promoted; native three-task OneJev remains held by the terminal held-out gate.


## 2026-10-23 — held-out batch advances beyond 1.3k MOT20 frames
- The latest blocking wait returned no terminal transition, but all active clients continued writing with no failure receipt.
- Current frame decisions are G1/G200 MOT20 1,499/1,345 of 1,657, G1/G200 RMOT 344/182 of 800, G1 OVMOT 16 of 40 sampled frames, and G200 AnimalTrack 144 of 306.
- The two existing GroundAnything services remain the only schedulers. Partial outputs remain excluded from scoring, and the native three-task OneJev launch remains gated on terminal held-out evidence.


## 2026-10-23 — held-out batch reaches 1.5k G1 MOT20 frames
- The latest blocking wait returned no terminal state change, but all active clients advanced and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,503/1,353 of 1,657, G1/G200 RMOT 349/189 of 800, G1 OVMOT 17 of 40 sampled frames, and G200 AnimalTrack 151 of 306.
- The two GroundAnything services remain the only schedulers. Partial outputs remain excluded from official scoring, and the native three-task OneJev gate remains closed until terminal held-out receipts.


## 2026-10-23 — held-out batch reaches 1.5k/1.3k frame milestones
- The latest blocking wait returned no terminal transition, but all active clients advanced and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,507/1,359 of 1,657, G1/G200 RMOT 354/196 of 800, G1 OVMOT 18 of 40 sampled frames, and G200 AnimalTrack 158 of 306.
- The two GroundAnything schedulers remain the complete project service set. No partial metric is promoted and the OneJev gate remains closed until terminal held-out receipts.


## 2026-10-23 — held-out outputs continue without scheduler growth
- The latest blocking wait returned no terminal state change; post-wait validation found all clients advancing and no failure receipt.
- Current frame decisions are G1/G200 MOT20 1,509/1,366 of 1,657, G1/G200 RMOT 357/203 of 800, G1 OVMOT 18 of 40 sampled frames, and G200 AnimalTrack 165 of 306.
- The two GroundAnything services remain the only schedulers. No partial metric is promoted, and native three-task OneJev remains gated on the complete held-out receipt set.


## 2026-10-23 — held-out batch passes 1.5k G1 MOT20 frames
- The latest blocking wait returned no terminal transition, but every active client continued writing and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,512/1,373 of 1,657, G1/G200 RMOT 360/210 of 800, G1 OVMOT 19 of 40 sampled frames, and G200 AnimalTrack 172 of 306.
- The two GroundAnything schedulers remain the only services. Partial outputs are not scored, and the native three-task OneJev gate remains closed until complete held-out receipts.


## 2026-10-23 — held-out batch reaches OVMOT midpoint
- The latest blocking wait returned no terminal state change, but all active clients advanced and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,516/1,378 of 1,657, G1/G200 RMOT 366/216 of 800, G1 OVMOT 20 of 40 sampled frames, and G200 AnimalTrack 178 of 306.
- The two GroundAnything schedulers remain the only services. Partial outputs remain excluded from official scoring, and the native three-task OneJev gate remains closed pending terminal held-out receipts.


## 2026-10-23 — held-out batch continues beyond 1.5k G1 MOT20 frames
- The latest blocking wait returned no terminal state transition; all active outputs advanced and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,521/1,385 of 1,657, G1/G200 RMOT 371/223 of 800, G1 OVMOT 20 of 40 sampled frames, and G200 AnimalTrack 185 of 306.
- The two GroundAnything schedulers remain the only services. Partial outputs remain excluded from metrics, and native three-task OneJev remains held by the terminal held-out gate.


## 2026-10-23 — held-out batch continues toward complete manifests
- The latest blocking wait returned no terminal state change, but all active clients advanced and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,524/1,392 of 1,657, G1/G200 RMOT 375/230 of 800, G1 OVMOT 21 of 40 sampled frames, and G200 AnimalTrack 192 of 306.
- The two GroundAnything schedulers remain the only services. No partial metric is promoted; native three-task OneJev remains gated on the complete held-out receipt set.


## 2026-10-23 — held-out batch continues past 1.5k/1.3k frame milestones
- The latest blocking wait returned no terminal transition, but all active clients advanced and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,529/1,398 of 1,657, G1/G200 RMOT 381/236 of 800, G1 OVMOT 22 of 40 sampled frames, and G200 AnimalTrack 198 of 306.
- The two GroundAnything schedulers remain the only services. Partial outputs remain excluded from official scoring, and the native three-task OneJev gate remains closed until terminal held-out receipts.


## 2026-10-23 — held-out batch advances toward terminal scopes
- The latest blocking wait returned no terminal transition; all active clients continued writing and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,534/1,406 of 1,657, G1/G200 RMOT 386/243 of 800, G1 OVMOT 22 of 40 sampled frames, and G200 AnimalTrack 206 of 306.
- The two GroundAnything schedulers remain the only services. Partial rows remain excluded from official metrics, and native three-task OneJev stays gated on terminal held-out receipts.


## 2026-10-23 — held-out inference continues beyond 1.4k G200 MOT20 frames
- The latest blocking wait returned no terminal transition; all active clients continued writing and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,540/1,411 of 1,657, G1/G200 RMOT 391/248 of 800, G1 OVMOT 23 of 40 sampled frames, and G200 AnimalTrack 210 of 306.
- The two GroundAnything schedulers remain the only services. No partial metric is promoted, and native three-task OneJev remains gated on terminal held-out receipts.


## 2026-10-23 — held-out batch keeps advancing under the two-service cap
- The latest blocking wait returned no terminal transition, but every active client continued writing and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,545/1,417 of 1,657, G1/G200 RMOT 396/255 of 800, G1 OVMOT 24 of 40 sampled frames, and G200 AnimalTrack 217 of 306.
- The two GroundAnything schedulers remain the only services. Partial results remain excluded from official metrics, and the native three-task OneJev gate remains closed until terminal held-out receipts.


## 2026-10-23 — held-out batch crosses G1 RMOT 400 frames
- The latest blocking wait returned no terminal state transition, but every active client continued writing and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,549/1,422 of 1,657, G1/G200 RMOT 401/260 of 800, G1 OVMOT 25 of 40 sampled frames, and G200 AnimalTrack 222 of 306.
- The two GroundAnything schedulers remain the only services. Partial outputs remain excluded from official scoring, and the native three-task OneJev gate remains closed until terminal held-out receipts.


## 2026-10-23 — held-out batch continues through G1 RMOT 400 frames
- The latest blocking wait returned no terminal state transition, while all active clients continued writing and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,551/1,426 of 1,657, G1/G200 RMOT 405/265 of 800, G1 OVMOT 25 of 40 sampled frames, and G200 AnimalTrack 227 of 306.
- The two GroundAnything schedulers remain the only services. Partial outputs remain excluded from official metrics, and native three-task OneJev remains gated on terminal held-out receipts.


## 2026-10-23 — held-out clients continue toward terminal receipts
- The latest blocking wait returned no terminal transition, while all active clients advanced and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,555/1,431 of 1,657, G1/G200 RMOT 410/271 of 800, G1 OVMOT 26 of 40 sampled frames, and G200 AnimalTrack 233 of 306.
- The two GroundAnything schedulers remain the only services. Partial outputs remain excluded from metrics, and native three-task OneJev remains gated until terminal held-out receipts.


## 2026-10-23 — held-out batch reaches G1 RMOT 416 frames
- The latest blocking wait returned no terminal state transition, but all active clients continued writing and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,561/1,436 of 1,657, G1/G200 RMOT 416/276 of 800, G1 OVMOT 27 of 40 sampled frames, and G200 AnimalTrack 238 of 306.
- The two GroundAnything schedulers remain the only services. Partial outputs remain excluded from official metrics, and native three-task OneJev remains gated on terminal held-out receipts.


## 2026-10-23 — held-out batch continues toward completion
- The latest blocking wait returned no terminal state transition, but all active clients advanced and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,566/1,442 of 1,657, G1/G200 RMOT 421/282 of 800, G1 OVMOT 27 of 40 sampled frames, and G200 AnimalTrack 244 of 306.
- The two GroundAnything schedulers remain the only services. Partial rows remain excluded from official scoring, and native three-task OneJev remains gated on terminal held-out receipts.


## 2026-10-23 — held-out batch advances toward full-frame contracts
- The latest blocking wait returned no terminal state transition, while all active clients advanced and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,571/1,448 of 1,657, G1/G200 RMOT 426/288 of 800, G1 OVMOT 28 of 40 sampled frames, and G200 AnimalTrack 250 of 306.
- The two GroundAnything schedulers remain the only services. Partial outputs remain excluded from official metrics, and native three-task OneJev remains gated on terminal held-out receipts.


## 2026-10-23 — held-out batch continues beyond 1.5k G1 MOT20 frames
- The latest blocking wait returned no terminal transition, but all active clients advanced and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,577/1,453 of 1,657, G1/G200 RMOT 432/295 of 800, G1 OVMOT 29 of 40 sampled frames, and G200 AnimalTrack 257 of 306.
- The two GroundAnything schedulers remain the only services. Partial outputs remain excluded from official metrics, and native three-task OneJev remains gated on terminal held-out receipts.


## 2026-10-23 — held-out batch advances toward terminal receipts
- The latest blocking wait returned no terminal transition, but all active clients advanced and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,582/1,459 of 1,657, G1/G200 RMOT 437/301 of 800, G1 OVMOT 30 of 40 sampled frames, and G200 AnimalTrack 263 of 306.
- The two GroundAnything schedulers remain the only services. Partial outputs remain excluded from official metrics, and native three-task OneJev remains gated on terminal held-out receipts.


## 2026-10-23 — held-out batch reaches 1.5k G200 MOT20 frames
- The latest blocking wait returned no terminal state transition, but all active clients advanced and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,587/1,467 of 1,657, G1/G200 RMOT 442/308 of 800, G1 OVMOT 31 of 40 sampled frames, and G200 AnimalTrack 270 of 306.
- The two GroundAnything schedulers remain the only services. Partial outputs remain excluded from official metrics, and native three-task OneJev remains gated on terminal held-out receipts.


## 2026-10-23 — held-out batch continues under the two-scheduler cap

- One blocking wait completed; the post-wait integrity check found all six active evaluation clients still writing and no failure receipt.
- Current frame decisions are G1/G200 MOT20 1,592/1,472 of 1,657, G1/G200 RMOT 448/316 of 800, G1 OVMOT 31 of 40 sampled frames, and G200 AnimalTrack 278 of 306. Prediction-line counts are 241,309 / 221,633 / 1,097 / 1,241 / 0 / 6,316 respectively; these are partial evidence only.
- The retry queues and finalizer orchestrator remain in their registered waiting states. Exactly two project SGLang services are present; no third scheduler was started. No official metric is promoted yet, and native three-task OneJev remains gated until terminal held-out receipts.


## 2026-10-23 — held-out scopes remain live after another blocking wait

- The next 240-second blocking wait produced no owner-status transition, but post-wait validation showed continued writes from the active scopes and no failure receipt.
- Current frame decisions are G1/G200 MOT20 1,599/1,480 of 1,657, G1/G200 RMOT 455/325 of 800, G1 OVMOT 32 of 40 sampled frames, and G200 AnimalTrack 287 of 306. Prediction-line counts are 242,423 / 222,961 / 1,111 / 1,262 / 0 / 6,502; these remain partial evidence and are not scored.
- Exactly two project SGLang schedulers remain. The retry queues and held-out finalizer orchestrator are still waiting; no native OneJev optimizer update or official metric has been promoted.


## 2026-10-23 — held-out clients continue writing; OVMOT remains latency-bound

- A further 240-second blocking wait produced no owner-status transition, but all non-OVMOT scopes continued writing and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,605/1,488 of 1,657, G1/G200 RMOT 461/333 of 800, G1 OVMOT 33 of 40 sampled frames, and G200 AnimalTrack 295 of 306. Prediction-line counts are 243,408 / 224,304 / 1,123 / 1,270 / 0 / 6,714; these are partial evidence only.
- G1 OVMOT has advanced one sampled frame since the previous audit and remains within the expected multi-minute TAO vocabulary latency. Exactly two SGLang schedulers remain; no partial metric or OneJev optimizer update is promoted.


## 2026-10-23 — held-out inference remains healthy under two schedulers

- The latest 240-second blocking wait had no owner-status transition; output validation still found active writes and no failure receipt.
- Current frame decisions are G1/G200 MOT20 1,611/1,492 of 1,657, G1/G200 RMOT 468/339 of 800, G1 OVMOT 34 of 40 sampled frames, and G200 AnimalTrack 297 of 306. Prediction-line counts are 244,335 / 225,159 / 1,145 / 1,276 / 0 / 6,771; partial evidence is not scored.
- Exactly two SGLang schedulers remain. The held-out finalizer and retry queues remain gated on terminal receipts; no native OneJev optimizer update or official metric is claimed.


## 2026-10-23 — held-out scopes continue toward terminal completion

- The latest 240-second blocking wait produced no owner-status transition, but every active scope continued writing and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,618/1,498 of 1,657, G1/G200 RMOT 474/346 of 800, G1 OVMOT 35 of 40 sampled frames, and G200 AnimalTrack 303 of 306. Prediction-line counts are 245,439 / 226,201 / 1,157 / 1,304 / 0 / 6,920; partial evidence is not scored.
- Exactly two SGLang schedulers remain. The terminal gate, retry queues and native three-task OneJev remain waiting for complete receipts; no official metric is promoted.


## 2026-10-23 — G200 AnimalTrack held-out scope completed with official metrics

- The G200 shared GPU7 AnimalTrack scope reached `SHARED_SERVICE_SCOPE_COMPLETE` after all 306 frames were processed. The contract reports no perception rejection and keeps the same parser/decoder/tracker/scorer path.
- Official AnimalTrack chicken_1 metrics: HOTA=0.4430628110, DetA=0.5497700434, AssA=0.3673280145, IDF1=0.4720116618, IDSW=142, FP=1,581, FN=1,323; end-to-end FPS=0.0289838. These values are promoted only for this complete scope, not as a cross-task claim.
- The remaining G1/G200 MOT20, RMOT and G1 OVMOT scopes are still running; the finalizer/OneJev gate remains closed until the registered complete receipt set is available.


## 2026-10-23 — post-AnimalTrack validation; remaining scopes still active

- After G200 AnimalTrack completed, another blocking wait produced no additional owner transition. The completed scope remains at all 306 frames; no duplicate client or scheduler was launched.
- Remaining frame decisions are G1/G200 MOT20 1,624/1,505 of 1,657, G1/G200 RMOT 482/361 of 800, and G1 OVMOT 36 of 40 sampled frames. Partial outputs remain excluded from official metrics.
- Exactly two SGLang schedulers remain. The finalizer and retry queues remain gated until all registered held-out owners have terminal receipts.


## 2026-10-23 — held-out batch advances after AnimalTrack completion

- A further blocking wait produced no additional owner transition; the completed G200 AnimalTrack scope remains intact at 306/306 frames.
- Remaining frame decisions are G1/G200 MOT20 1,630/1,512 of 1,657, G1/G200 RMOT 489/377 of 800, and G1 OVMOT 37 of 40 sampled frames. All active scopes continue writing and no failure receipt is present.
- The two-scheduler resource cap remains intact. No partial output is promoted and the finalizer/OneJev gate remains waiting for all registered held-out receipts.


## 2026-10-23 — remaining held-out scopes continue writing

- Another blocking wait returned no owner-status transition, but every unfinished scope advanced and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,635/1,519 of 1,657, G1/G200 RMOT 494/393 of 800, and G1 OVMOT 38 of 40 sampled frames. G200 AnimalTrack remains complete at 306/306.
- Exactly two SGLang schedulers remain. No partial result is promoted; finalizer and native OneJev remain gated on terminal receipts.


## 2026-10-23 — held-out batch reaches final TAO sample

- The latest blocking wait produced no owner-status transition, but all unfinished scopes advanced and no failure receipt appeared.
- Current frame decisions are G1/G200 MOT20 1,641/1,527 of 1,657, G1/G200 RMOT 500/411 of 800, and G1 OVMOT 39 of 40 sampled frames. G200 AnimalTrack remains complete at 306/306.
- The two-scheduler cap is unchanged. G1 OVMOT is processing its final sampled frame; partial outputs remain excluded until the complete receipt and scorer finish.


## 2026-10-23 — G1 OVMOT held-out scope completed; all 40 frames rejected

- The G1 GPU5 OVMOT scope reached `SHARED_SERVICE_SCOPE_COMPLETE` with all 40 registered TAO validation samples processed. The official TETA recipe receipt is complete, but every frame was rejected by the strict wire/parser contract (40/40, rejection rate 1.0), so predictions=0.
- Official registered OVMOT metrics are explicitly zero: TETA=0, LocA=0, AssocA=0, ClsA=0, and all recall/precision components=0. End-to-end FPS=0.00307586. This is a valid failure-boundary result, not a detector or generalization gain.
- Remaining G1/G200 MOT20 and RMOT scopes continue running; finalizer and native OneJev remain gated until their complete receipts are available.


## 2026-10-23 — G1 MOT20 inference complete; scorer/remaining scopes pending

- G1 MOT20 has written all 1,657 frame decisions (251,751 prediction lines); its owner is still marked running, so the final scorer/receipt has not yet been promoted.
- Current unfinished counts are G200 MOT20 1,541/1,657, G1/G200 RMOT 584/440 of 800. G1 OVMOT is complete at 40/40 with the documented all-rejected zero TETA receipt; G200 AnimalTrack is complete at 306/306.
- Exactly two SGLang schedulers remain and no duplicate client was started. Official metrics are reported only after each owner reaches its terminal receipt.


## 2026-10-23 — G1 MOT20 full held-out scope scored

- G1 GPU5 MOT20-05 reached `G2_PARALLEL_SCOPE_COMPLETE` after all 1,657 frames were processed. Official metrics: HOTA=0.3502305038, DetA=0.6632209880, AssA=0.1852801070, IDF1=0.2993288355, IDSW=10,993, FP=14,516, FN=81,990. Six perception-rejected frames remain explicitly in the denominator; this is a complete scope result, not a partial estimate.
- The result is below the preserved G0/G2 references on this scope and is retained as an honest G1 adaptation measurement; no gain claim is made.
- Remaining G200 MOT20 and both RMOT scopes are still running. Two SGLang schedulers remain and the terminal gate is still closed.


## 2026-10-23 — G1 complete; G200 MOT20/RMOT continue under shared services

- After G1 MOT20 completion, the active unfinished scopes continued writing with no failure receipt: G200 MOT20 1,551/1,657, G1 RMOT 660/800, and G200 RMOT 460/800.
- G1 OVMOT (40/40, official TETA=0 due 40 rejected frames) and G200 AnimalTrack (306/306, official metrics already recorded) remain complete.
- The retry queues correctly remain dependency-gated on the still-running RMOT/MOT20 owners. The two-scheduler cap is unchanged.


## 2026-10-23 — G1 RMOT timeout isolated and safe v3 resume launched

- The G1 shared RMOT v2 owner failed at frame 660 with a real HTTP `TimeoutError` during native perception. Its completed 660-frame prefix and failure receipt are preserved; no metric was promoted.
- Added `queue_g1_rmot_shared_retry.py`: it reuses the same GPU5 service, restores the exact v2 causal prefix into fresh `g1_rmot_shared_gpu5_v3`, raises only the per-request timeout to 900 seconds, reruns the unchanged official parser/scorer, and verifies the full 800-frame contract before accepting a receipt.
- The G1 DanceTrack retry was moved to fresh v3 output and now waits for RMOT v3 plus the completed OVMOT scope. The finalizer/OneJev owner list and collector were updated to use RMOT v3 while retaining v2 as historical failure evidence. Compilation and diff checks pass; exactly two SGLang schedulers remain.


## 2026-10-23 — G1 RMOT v5 completed after service-lifecycle repair

- The first retry (v3) stalled because the completed G1 MOT20 owner orphaned its SGLang service in graceful shutdown; v4 also exposed an output-root collision because the new service log was placed inside the retry root. Both failures are preserved with receipts.
- The repaired v5 path starts a replacement G1 service on the same GPU5/8240 endpoint with an isolated service output directory, restores the exact 660-frame v2 prefix, and completes all 800 RMOT frames. Official metrics: HOTA=0.0377768821, DetA=0.0056875540, AssA=0.2524360693, IDF1=0.0089435439, IDSW=3, FP=1,760, FN=7; 45 perception-rejected frames are retained in the denominator. This is a complete but weak result, with no gain claim.
- The v5 scope now has both queue and scope owner receipts; DanceTrack v3 is running on the same replacement service. Exactly two SGLang schedulers remain.


## 2026-10-23 — post-RMOT repair scopes continue under two schedulers

- G1 RMOT v5 remains complete at 800/800 with its official receipt; G1 DanceTrack v3 is now at 252/1,203 frames on the replacement GPU5 service.
- G200 MOT20 is at 1,615/1,657 frames and G200 RMOT at 591/800; both continue writing with no failure receipt.
- Exactly two SGLang schedulers remain. The retry collector and held-out finalizer are waiting for terminal G1 DanceTrack/G200 scopes; no partial metric or OneJev update is promoted.


## 2026-10-23 — G1 DanceTrack and G200 scopes continue after RMOT repair

- G1 DanceTrack v3 advanced to 387/1,203 frames on the replacement GPU5 service; G200 MOT20 advanced to 1,624/1,657 and G200 RMOT to 610/800.
- G1 RMOT v5 remains complete at 800/800 with the official low-score receipt. Exactly two SGLang schedulers remain; all partial outputs stay outside official metrics and OneJev remains gated.


## 2026-10-23 — consolidated-result gate added before JEV coupling

- The active held-out batch remains the only prerequisite for the next phase. Current progress after the latest blocking wait: G1 DanceTrack 496/1,203 frame decisions, G200 MOT20 1,630/1,657, and G200 RMOT 622/800; completed scopes retain their terminal official receipts. The G200 OVMOT retry and extra shared-control collector remain dependency-gated.
- The orchestrator now waits for the registered owners plus the retry/extra-control queue, writes and prints a terminal-only consolidated receipt, and only after that runs the held-out finalizers and starts the real native MOT/RMOT/OVMOT JEV coupling queue. Partial JSONL rows cannot trigger training.
- Added `scripts/groundjev/print_heldout_consolidated_result.py` and updated `orchestrate_heldout_finalizers_and_onejev.py`; both compile cleanly. The replacement orchestrator is live under a new PID and exactly two project SGLang schedulers remain. No JEV optimizer update has started.


## 2026-10-23 — held-out clients advanced during the verified wait

- The blocking wait timed out without a terminal owner transition, but the post-wait audit confirmed continued writes and no failure: G1 DanceTrack 900/1,203, G200 MOT20 1,657/1,657 frame decisions (scorer/receipt still running), and G200 RMOT 772/800.
- G200 OVMOT retry remains correctly gated on terminal G200 MOT20/RMOT controls; the extra shared-control collector and consolidated-result gate remain waiting. No consolidated result has been printed and no JEV optimizer update has started.
- Exactly two project SGLang schedulers remain. Partial frame rows are still excluded from official metrics.


## 2026-10-23 — G200 MOT20 terminal; G200 RMOT remains live

- G200 MOT20 reached `G2_PARALLEL_SCOPE_COMPLETE` after all 1,657 frames; its scorer receipt is now terminal and the completed output is immutable.
- The shared GPU7 service remains reachable for the still-running G200 RMOT client, which advanced to 777/800 frame decisions; G1 DanceTrack advanced to 919/1,203. No service replacement or extra scheduler was started.
- G200 OVMOT retry remains gated on terminal G200 RMOT; the consolidated-result printer and JEV coupling gate remain closed. Exactly two GroundAnything service endpoints (GPU5/8240 and GPU7/8238) remain in use.


## 2026-10-23 — finalizer owner lists repaired before terminal collection

- The three held-out finalizer scripts were updated to include the current G1 DanceTrack v3, G200 AnimalTrack shared, and G200 OVMOT retry-v2 owner receipts. This prevents the final evidence collection from silently omitting the retry scopes that the orchestrator already gates.
- All three scripts compile cleanly. The running orchestrator will execute the updated files after the remaining owners become terminal; no active inference process was restarted.


## 2026-10-23 — G200 RMOT timeout isolated and causal retry resumed

- The G200 shared RMOT owner reached a real `TimeoutError` at frame 777 after writing the exact 0–776 prefix. The failed owner, failure receipt and prefix remain preserved; no metric is promoted from the incomplete run.
- Added `queue_g200_rmot_shared_retry.py`, which reuses GPU7/8238, raises only the per-request timeout to 900 seconds, restores the exact prefix into `g200_rmot_shared_gpu7_v2`, and writes a separate owner/official-scoring receipt. The retry is currently running from frame 777 with no additional scheduler.
- The pre-start G200 OVMOT queue was stopped and marked superseded before it could pass the old failed RMOT dependency. Its new queue state waits on the repaired RMOT owner. The orchestrator and extra-control collector were restarted with the repaired dependency paths; JEV remains gated.


## 2026-10-23 — superseded OVMOT child group reclaimed before RMOT retry

- The pre-start OVMOT queue had already spawned a detached GPU7 service/client group on port 8243 before its parent was stopped. The command lines and output root were verified as this project; the group was terminated without touching foreign processes.
- Its two-frame partial output was moved to `g200_ovmot_dedicated_gpu7_v2_superseded_before_rmot_repair` with a superseded failure receipt, so the fresh v2 OVMOT retry cannot overwrite it. Port 8243 is now free; GPU7 is serving only the G200 RMOT retry on 8238.
- A new OVMOT queue is waiting on `g200_rmot_shared_gpu7_v2/owner_state.json`; the consolidated-result and JEV gates remain closed.

## 2026-10-23 — RMOT v2 timeout reproduced and bounded rejection repair activated

- The second G200 RMOT attempt reproduced the same real HTTP TimeoutError at frame 777 after preserving an exact 777-frame prefix. Its failed owner and queue receipt remain immutable at g200_rmot_shared_gpu7_v2; no incomplete output is eligible for official metrics.
- The runner was repaired at the request boundary: under the registered empty_and_record policy, a request timeout now writes an explicit empty rejected frame, retains that frame in the metric denominator, and emits no guessed box. The stop policy remains unchanged. The runtime copy was patched and compile/import checked.
- The detached OVMOT v2 child group that had survived its failed parent was reclaimed after command-line/output ownership verification. Its two-frame partial output was moved to g200_ovmot_dedicated_gpu7_v2_superseded_before_rmot_retry_v3 with a superseded receipt; port 8243 is free again.
- Restarted the single held-out orchestrator and extra-control collector with v3 dependency paths, launched queue_g200_rmot_shared_retry_v3.py on the existing GPU7/8238 service, and launched the gated OVMOT v3 queue. At launch the RMOT v3 owner entered RUNNING_G200_RMOT_RETRY; OVMOT remains waiting on its terminal RMOT dependency. No extra SGLang scheduler was created.

## 2026-10-23 — repeated RMOT timeouts bounded without guessing predictions

- The first repaired RMOT v3 client recorded explicit timeout rejections at frames 777–779, each taking the full 900-second request budget; its immutable 780-row prefix was moved to g200_rmot_shared_gpu7_v3_superseded_before_timeout_circuit and marked ineligible for metrics.
- Added max-consecutive-timeouts to the runner. The new retry keeps the exact prefix, issues at most one further request, and then records the remaining frames as empty request_timeout_circuit_open rejections while preserving the official denominator and tracker state. This prevents a dead service from consuming hours on repeated invalid requests.
- The shared GPU7/8238 service was kept alive; the old RMOT client/queue were terminated only after ownership verification. The existing OVMOT gate will continue to wait on the terminal RMOT v3 owner, and no extra scheduler was created.

## 2026-10-23 — G200 RMOT full legal scope completed after bounded recovery

- RMOT v3 completed all 800 registered frames and passed the official unmodified TrackEval scorer. The terminal receipt reports HOTA 0.4011940792, DetA 0.3276041719, AssA 0.4921537914, IDF1 0.4630333756, IDSW 83, FP 1349, FN 839.
- Frames 777–799 are explicit perception rejections (23/800, 2.875%) after the first timeout circuit opened; they remain in the denominator and contain no guessed boxes. Because the output resumed from an interrupted prefix, end-to-end FPS is not claimed; the resumed pipeline timer is recorded separately.
- The G200 OVMOT dedicated queue is now released by the terminal RMOT owner. Its runner also receives the one-time timeout circuit breaker so repeated invalid requests cannot consume hours; all OVMOT frames and official TETA components remain pending.

## 2026-10-23 — OVMOT launch resource race preserved and made waitable

- The first post-RMOT OVMOT launch passed the queue's 32-GiB probe but lost memory to concurrent GPU7 work before the child scope's second probe (25,310 MiB); it exited before inference. The empty owner and resource receipt were moved to g200_ovmot_dedicated_gpu7_v2_superseded_resource_gate and are excluded from metrics.
- Patched run_g2_parallel_scope.py to wait up to 24 hours for the same 32-GiB floor rather than fail on this race. Foreign GPU7 processes are untouched. The dedicated OVMOT queue remains bounded by the timeout circuit and will be relaunched after the waitable gate is live.

## 2026-10-23 — OVMOT slow-response bound tightened after direct timing audit

- The first waitable OVMOT retry reached only five output frames after 26 minutes; completed requests took 482–773 seconds and the next request was still active. The 900-second timeout was therefore too permissive for this service behavior.
- Stopped only the project-owned OVMOT groups, preserved the five-row causal prefix at g200_ovmot_dedicated_gpu7_v2_superseded_before_ovmot_timeout180, and reduced the dedicated request timeout to 180 seconds with the one-time circuit breaker. No partial output is promoted to metrics.

## 2026-10-23 — OVMOT socket timeout replaced by full-call wall timeout

- The 180-second socket timeout still did not interrupt a server that kept the HTTP connection alive while taking 634 seconds on frame 146. Six output rows were preserved at g200_ovmot_dedicated_gpu7_v2_superseded_before_ovmot_wall_timeout; no metric is promoted.
- Added a SIGALRM wall-clock guard around the entire GroundAnything client call. The dedicated OVMOT retry remains at 180 seconds and the one-time circuit breaker; this bounds server-side slow responses as well as idle sockets without guessing boxes.

## 2026-10-23 — held-out evaluation batch closed and result printed before JEV

- The final G200 dedicated OVMOT retry completed its 40 registered TAO annotation frames with the official unmodified TETA recipe. All 40 frames were explicit perception rejections, predictions were 0, and TETA/LocA/AssocA/ClsA were all 0. The slow-response prefixes and resource-gate failures remain archived separately and are not mixed into this result.
- The terminal-only consolidated receipt was generated at outputs/groundjev/heldout/groundjev_heldout_consolidated_result_20261023.{json,md} and published under docs/GROUNDJEV/results/ on the research branch. It includes G2/G1/G200 MOT20, DanceTrack, AnimalTrack, RMOT and OVMOT rows, frame counts, rejection counts, official metrics and failure/supersession evidence.
- The orchestrator reached RUNNING_HELDOUT_FINALIZERS; JEV coupling is still gated behind the finalizers and will start only after this result receipt has been printed and the finalizer records complete.

## 2026-10-23 — JEV launch permission failure isolated and fixed

- All three held-out finalizers completed successfully and the orchestrator then attempted the real native three-task queue on GPUs 5/7. The first launch made zero optimizer updates and failed before training because it executed run_with_resources.py directly even though that Python file has no executable bit (PermissionError(13)).
- Patched queue_real_three_task_native.py to invoke the resource wrapper through the validated OneJev Python interpreter. The failed queue state and preflight args.json remain preserved; no training checkpoint or metric was claimed from the failed launch.


## 2026-10-23 — native coupling context repair and compact retry

- The first native coupling attempt failed before optimizer update 1 because the v1 runtime rendered 2,099 of 131,618 rows over max_len=32768. A 98,304-token retry completed 5 real updates but was impractical at roughly 68–150 seconds per update; its zero-checkpoint log is preserved and the owner is marked superseded.
- Restored the registered compact_competition_v1 loader/module and created an immutable v2 manifest from the same 131,618 native MOT/RMOT/OVMOT rows. Candidate boxes, row IDs, targets, media and all positive-IoU edges are retained; only reconstructible dense matrices are omitted. The CPU plan proves 0 over-length rows at max_len=32768 (p50/p90/p99/max 7948/12182/16050/28156; 8,227 planned steps).
- The current compact native coupling run is active on GPUs 3 and 7 with the same full source exposure. At the last receipt it reached step 50/8,227, with 0.57–0.90 rows/s and no checkpoint yet (save interval 500). GPU5 was excluded because its free memory was 1.7 GiB; the old GPU5/7 wait and uncompact retry remain preserved.


## 2026-10-23 — completed held-out services reclaimed for joint training

- Ports 8238, 8240 and 8243 were terminal held-out GroundJEV scopes with no active client connections. Their project-owned scheduler groups were stopped after exact command-line verification; no foreign process was touched. Residual GPU memory is now available to the live native three-task OneJev run.
## 2026-10-24 — real native 三任务联合训练 step4000 checkpoint 完整

联合 GroundJEV/OneJev native MOT/RMOT/OVMOT 训练已到 step4000/8227（examples=64,000），最近事件 loss=1.276689、CE=0.919309、Brier=0.357380、slot_acc=0.7125、sec/step=23.770、峰值显存约 7.80 GiB。`adapter-step4000` 已完整落盘；`adapter_model.safetensors` 为 43,346,432 bytes，SHA-256 `8a7001f0478b7ba0afd80c258fe8483a8e910f894941a0546b4451d1c54dbedc`，372 个 tensor 的 CPU `safe_open` 校验通过。owner 仍为 `REAL_NATIVE_THREE_TASK_TRAINING_RUNNING`，尚无 terminal receipt 或 JEV held-out 指标；该训练进度不构成性能提升结论。
## 2026-10-25 — real native 三任务联合训练 step4500 checkpoint 完整

联合 GroundJEV/OneJev native MOT/RMOT/OVMOT 训练已到 step4500/8227（examples=72,000），最近事件 loss=1.043941、CE=0.734483、Brier=0.309458、slot_acc=0.7725、sec/step=21.204、峰值显存约 7.80 GiB。`adapter-step4500` 已完整落盘；`adapter_model.safetensors` 为 43,346,432 bytes，SHA-256 `e5d133ff665043051fed5a07c6c49191c7b90c1d487b4437fa7d1f01b05b082f`，372 个 tensor 的 CPU `safe_open` 校验通过。owner 仍为 `REAL_NATIVE_THREE_TASK_TRAINING_RUNNING`，尚无 terminal receipt 或 JEV held-out 指标；该训练进度不构成性能提升结论。
## 2026-10-25 — real native 三任务联合训练 step5000 checkpoint 完整

联合 GroundJEV/OneJev native MOT/RMOT/OVMOT 训练已到 step5000/8227（examples=80,000），最近事件 loss=1.113642、CE=0.795815、Brier=0.317827、slot_acc=0.7575、sec/step=20.697、峰值显存约 7.80 GiB。`adapter-step5000` 已完整落盘；`adapter_model.safetensors` 为 43,346,432 bytes，SHA-256 `202003ef431dcde6a0c75205c43afe395f3d0dfb9e85cd9065d720983367d298`，372 个 tensor 的 CPU `safe_open` 校验通过。owner 仍为 `REAL_NATIVE_THREE_TASK_TRAINING_RUNNING`，尚无 terminal receipt 或 JEV held-out 指标；该训练进度不构成性能提升结论。
## 2026-10-25 — real native 三任务联合训练 step5500 checkpoint 完整

联合 GroundJEV/OneJev native MOT/RMOT/OVMOT 训练已到 step5500/8227（examples=88,000），最近事件 loss=0.823151、CE=0.570731、Brier=0.252420、slot_acc=0.8125、sec/step=16.493、tokens/s=7805.095、峰值显存约 7.80 GiB。`adapter-step5500` 已完整落盘；`adapter_model.safetensors` 为 43,346,432 bytes，SHA-256 `13404900e546c08443790e5c3b496cd1403916b5b5724857a0254e9f5366dee0`，372 个 tensor 的 CPU `safe_open` 校验通过，adapter/config 文件可读。owner 仍为 `REAL_NATIVE_THREE_TASK_TRAINING_RUNNING`，尚无 terminal receipt 或 JEV held-out 指标；该训练进度不构成性能提升结论。
## 2026-10-25 — real native 三任务联合训练 step6000 checkpoint 完整

联合 GroundJEV/OneJev native MOT/RMOT/OVMOT 训练已到 step6000/8227（examples=96,000），最近事件 loss=0.903464、CE=0.639255、Brier=0.264209、slot_acc=0.805、sec/step=19.505、tokens/s=6634.281、峰值显存约 7.80 GiB。adapter-step6000 已完整落盘；adapter_model.safetensors 为 43,346,432 bytes，SHA-256 c20f8e0bfd0065febc82f8ffed77ac97495cf60ec8dd42a90f87929c570101cb，372 个 tensor 的 CPU safe_open 校验通过。owner 仍为 REAL_NATIVE_THREE_TASK_TRAINING_RUNNING，尚无 terminal receipt 或 JEV held-out 指标；该训练进度不构成性能提升结论。

## 2026-10-25 — real native 三任务联合训练 step7000 checkpoint 完整

真实 native MOT/RMOT/OVMOT 联合训练已到 step7000/8227（examples=112,000）。最近事件 loss=0.932824、CE=0.657154、Brier=0.275671、slot_acc=0.790、sec/step=26.412、tokens/s=4994.005、峰值显存约 7.80 GiB。adapter-step7000 已完整落盘；adapter_model.safetensors 为 43,346,432 bytes，SHA-256 33a8f43144df3813400f32ed96e1924da6c6920d16b89302a768ad9f1fbb6415，372 个 tensor 的 CPU safe_open 校验通过。owner 仍为 REAL_NATIVE_THREE_TASK_TRAINING_RUNNING，尚无 terminal receipt 或 JEV held-out 指标；该训练进度不构成性能提升结论。

## 2026-10-25 — real native 三任务联合训练 step7500 checkpoint 完整

真实 native MOT/RMOT/OVMOT 联合训练已到 step7500/8227（examples=120,000）。最近事件 loss=0.738868、CE=0.519953、Brier=0.218915、slot_acc=0.8375、sec/step=22.963、tokens/s=5418.052、峰值显存约 7.80 GiB。adapter-step7500 已完整落盘；adapter_model.safetensors 为 43,346,432 bytes，SHA-256 f4ddd56ff955516f775ba347e8eeaf9cb5e990da1c13a2b97b58272cf8779675，372 个 tensor 的 CPU safe_open 校验通过。owner 仍为 REAL_NATIVE_THREE_TASK_TRAINING_RUNNING，尚无 terminal receipt 或 JEV held-out 指标；该训练进度不构成性能提升结论。

## 2026-10-26 — native three-task OneJev training terminal and TrackJEV batching repair

The real native MOT/RMOT/OVMOT joint run terminated cleanly at 8227/8227 optimizer updates with 131,618 manifest rows and no failure. The final event (the last partial nine-row update) is loss=1.661405, CE=1.299044, Brier=0.362360, slot_acc=0.6875; the preceding step8225 event was loss=0.792641, CE=0.549075, Brier=0.243566, slot_acc=0.8200. These are training-health values, not held-out performance. The terminal receipt reports 189,998.784 seconds elapsed, exact manifest SHA `7c92778fb4417ea1e49bf286cab7758ceef1443d683ab059e5e758e5bd823aa3`, physical GPUs [3,7], and `failed=null`. The final root `adapter/adapter_model.safetensors` is 43,346,432 bytes, 372 tensors, SHA-256 `6f71d15e001b2a3600ceb3adc3e012e74d10b47069cbb5a13e2866b7010b18a6`; the step7500 checkpoint remains preserved with SHA `f4ddd56ff955516f775ba347e8eeaf9cb5e990da1c13a2b97b58272cf8779675`. CPU merge into `merged-final-step8227` completed with base checkpoint step600 and `ONEJEV_LORA_MERGE_COMPLETE`, preserving base and adapter.

A real TrackJEV MOT20-05 four-frame smoke using merged step8227 and the exact independent G2 candidate replay completed without GT reads. The existing adapter implementation first constructed an overlong 220-candidate request and took 231--272 seconds per non-empty frame. The registered `batch_candidates=8` parameter was unused; it is now applied before request construction, with all candidates retained and global assignment unchanged. Adapter/model/decision-layer selftests and compileall pass. The repaired smoke processed 220/220/210 candidates in 15.47/15.29/14.56 seconds per non-empty frame (45.71 seconds total), with no JEV benchmark claim; official metrics remain pending full held-out evaluation.

## 2026-10-26 — final OneJev held-out multitask evaluation started

The merged final native OneJev step8227 was evaluated with the repaired TrackJEV adapter on frozen independent candidate replays, with no GT read by runtime. Registered RMOT multi0007 (800 frames) completed and the unmodified official TrackEval receipt reports HOTA=0.0039727608, DetA=0.0004734848, AssA=0.0333333333, IDF1=0.0009465215, IDSW=0, FP=0, FN=2111; 2,112 release query GT boxes, 1,657 candidate-covered at IoU>=0.5 (post-hoc coverage recall 0.7845644), and 3.1557 association-replay FPS. Registered RMOT single0007 (800 frames) completed with HOTA=0, DetA=0, AssA=0, IDF1=0, IDSW=0, FP=0, FN=18; release query GT boxes=18 and post-hoc candidate coverage recall=0.8888889 at IoU>=0.5, 3.0860 association-replay FPS. Both reports use TrackEval commit `12c8791b303e0a0b50f753af204249e622d0281a`, preserve all frames, and make no ordinary-model claim.

Registered TAO OVMOT video105 (40 annotation frames) completed through the official non-exhaustive TETA evaluator. The upstream GroundAnything candidate replay rejected all 40 frames, produced zero predictions, and the receipt reports TETA/LocA/AssocA/ClsA and all recall/precision fields equal to 0; this is a measured perception rejection boundary, not evidence about the JEV decoder. The TETA evaluator commit is `b498aa87b252bfb75d7cc0d8d6435c3e1e74e4de`. Full independent MOT20-05 TrackJEV evaluation is still running on GPU7 with the same merged step8227 model, exact G2 candidate replay and no GT feedback; its official TrackEval step will be run only after all 1,657 frames finish.

The repaired `locatemot/groundjev/trackjev/native_onejev_adapter.py` was synchronized to the research branch at GitHub commit `93cd1f9f2cae0f3f907fd7771965edb54d10f7a7` (blob SHA `30dfaf0c036cd2eb2234781d09fcdd5abe37331a`). This is the bounded native candidate-request batching fix validated by compileall and the TrackJEV/OneJev self-tests; it keeps all candidates and leaves global assignment unchanged.

A temporary batch-size comparison on GPU4 used the same merged step8227 model and frozen G2 candidate replay for a four-frame functional scope. `batch_candidates=16` completed without GT reads but took 16.71/16.48/16.08 seconds on the non-empty frames, slower than the preserved batch-8 smoke (15.47/15.29/14.56 seconds). The configurable CLI experiment is not retained and no second full-video run was started; the existing GPU7 batch-8 evaluation remains the registered scope.
