Warning: truncated output (original token count: 137554)
Total output lines: 4003

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
- Reproducibility hypothesis: a fresh checkout should import the unified data
  package using tracked source only. The first clone failed because the broad
  `data/` ignore rule also hid `locatemot/unified/data/`. Added a narrow ignore
  exception, committed the package source, pushed it, and re-cloned the remote
  branch at `26f9c989c8b13773d681bbe6962b77e73a0026bd`. Imports, skeleton smoke,
  and foundation smoke passed in the clone. Decision: keep the repair and mark
  U0 legal-scope reproducibility PASS.
- Checkpoint hypothesis: the MMDetection Swin-B conversion must have only
  documented non-core load mismatches. `tools/unified/audit_foundation_checkpoint.py`
  now requires the exact one training-only missing key
  `dn_query_generator.label_embedding.weight`, the exact three BERT metadata
  unexpected keys, and zero backbone/encoder/decoder/bbox-head/language-fusion
  missing keys. Audit result: `PASS_CONDITIONAL_TRAINING_ONLY_MISSING`.
- Preprocessing hypothesis: the formal runtime must be equivalent to the
  official MMDetection path. Ten deterministic legal images were compared with
  the same fixed generic vocabulary; all produced 300 finite predictions,
  maximum box difference `0.0`, and maximum score difference `0.0`. Decision:
  preprocessing contract PASS; the prior hand-built tensor smoke is not formal
  evidence.
- Split hypothesis: a deterministic video-disjoint split balancing expression
  volume, physical-target volume, and motion proxy is adequate before formal
  evaluation. `tools/unified/build_legal_split.py` froze v2 with seed `20261001`
  and split SHA
  `8ee806048ccbb227919f27edf68cf0bebc1fa50f824a8108878fc5f39fc6e1c0`.
  V1 fit/calibration/validation are `9/3/3` videos and V2 are `9/3/3`; V2
  source-target-ID-missing references remain `157` (fit only in this split),
  with no replacement or silent drop. Decision: freeze this split before any
  U1 adaptation or formal gate.
- Hardware/wall time: one CUDA device (A800-equivalent) was used for the
  checkpoint audit, smoke, and equivalence check; these short checks completed
  in under a minute each. Official-test labels and screening ground truth were
  not opened. External baseline execution remains incomplete and is the next
  blocking U0 task.

## LocateMOT-U U0R — TempRMOT legal-development execution — 2026-10-01

- Hypothesis: at least one published external RMOT method can be executed on a
  frozen legal-development subset without reading official-test labels or
  mixing its environment into LocateMOT-U.
- Protocol: copied TempRMOT at commit
  `6a65640d849fdee4a32bb055945ee34c3b0edeb1` into the isolated environment
  `/data2/user/reference_envs/temprmot`, loaded the released checkpoint
  `checkpoint_rk.pth` (SHA-256
  `55e9b2cecbc1590aface4c9d8639008e694e8c4556d1d3c27bf90549f408c7e8`), and
  ran the adapter on V1 validation videos `0012` and `0014`, one legal
  expression per video. TrackEval then evaluated the two generated sequences
  with HOTA, CLEAR, and Identity.
- Result: prediction and evaluation completed with 94 prediction rows and 93
  ground-truth rows. Combined HOTA was `94.679`, MOTA `98.925`, and IDF1
  `99.465`. Official-test labels, screening ground truth, and training were not
  used. The raw outputs and environment remain outside the repository;
  `outputs/unified/external_baselines/temprmot_legal_v1.json` records the
  reproducible metadata and `reports/unified/EXTERNAL_BASELINE_LEGAL_DEV.md`
  records the bounded scope.
- Compatibility work was confined to the isolated copy: a torchtext shim,
  current PyTorch `torch.load` behavior, disabling a redundant pretrained
  download, a CUDA scalar-type API update, and NumPy aliases in TrackEval.
  These changes do not alter the LocateMOT-U model or the frozen split.
- Decision: keep the TempRMOT run as real external-baseline evidence, while
  retaining `U0_EXTERNAL_BASELINE=INCOMPLETE` until a second independent
  method also produces predictions and evaluation. DKGTrack and FlexHook
  remain pending because their published weight links are not yet available
  through the isolated download path.

## LocateMOT-U U1R — formal generic and expression measurements — 2026-10-01

- Measurement contract: the frozen split SHA is
  `8ee806048ccbb227919f27edf68cf0bebc1fa50f824a8108878fc5f39fc6e1c0`; the
  foundation checkpoint SHA is
  `55949c9c0f46339a73b415334765615d491ee6ed739ed3f568142b7fc5581143`; every
  forward used `LoadImageFromFile`, `FixScaleResize(800,1333)`,
  `PackDetInputs`, and `DetDataPreprocessor` with BGR-to-RGB conversion.
- Generic protocol: one fixed query-independent vocabulary, all legal
  calibration/validation frames with materialized labels, physical units
  `(dataset, video, frame, track_id)`, Top100/150/300/900 diagnostics, and
  small/medium/large height strata. The run covered 4,201 unique images,
  33,526 physical GT units, and 5,992 unique physical objects. Top150 recall
  was V1 `0.0803` and V2 `0.0669` at IoU 0.50 (V1/V2 IoU .25: `.5309/.4940`,
  IoU .75: `.0045/.0062`). Decision: formal generic gate FAIL; this is below
  the `.80` adaptation early-pass, so a detection adaptation probe is required
  before any grounding adaptation.
- Expression protocol: every legal calibration/validation query contributed
  one deterministic middle labeled frame and, when available, one first empty
  target frame. Units were `(dataset, video, query_id, frame, target_id)`;
  source-missing references stayed in the source denominator. The run covered
  7,160 query-frame rows and 7,199 valid target units with zero source-missing
  units in this calibration/validation sample. Top1 IoU .50 recall was V1
  `0.0030` and V2 `0.0028`; AP50 and empty-frame FP/query are recorded in
  `reports/unified/U1_EXPRESSION_GROUNDING_GATE.md`. Decision: formal
  expression gate FAIL; no expression adaptation has started.
- No official-test labels or screening ground truth were read. No U2 or
  universal track core work was started. The external-baseline state remains
  incomplete because only TempRMOT has real predictions/evaluation; DKGTrack
  and FlexHook public Baidu links require an authenticated download path.

## LocateMOT-U U1-A — protocol correction, valid adaptation, and stop — 2026-10-01

- The first eight-process launch was invalid because MMEngine's default
  `--launcher none` left the processes outside DDP; it was stopped before any
  formal evidence. A later exploratory FP32 run also used `frozen_stages=1`.
  MMDetection's Swin implementation shows that value freezes patch embedding
  and stage0 only, while the registered contract requires stages 0 and 1.
  Those exploratory metrics are excluded from the final record.
- Corrected hypothesis/protocol: a fixed legal-fit generic stream can repair
  the foundation proposal localization while freezing Swin stages 0/1 and
  BERT. The legal-fit stream has 3,606 rows and fixed `label 0 -> car`; the
  split, vocabulary, official `FixScaleResize(800,1333)` pipeline, and all
  denominators stayed unchanged. The valid run used
  `frozen_stages=2`, BF16 AMP, eight-GPU PyTorch DDP, batch 2/GPU, and no
  accumulation (global batch 16). The logged peak was 5,945 MB/GPU. A two-step
  smoke passed before the full run.
- A batch-2/two-step-accumulation attempt passed a one-step smoke but failed in
  the full run with PyTorch's `expect_autograd_hooks_` DDP reducer assertion.
  The smallest working fix was accumulation 1; this kept the recommended
  microbatch 2 and did not alter the official input resolution. This failure
  and the corrected smoke are retained in the adaptation report.
- The valid schedule completed epochs 1, 2, 4, and 6, each formally evaluated
  on the unchanged 4,201-image, 33,526-physical-unit scope. Top150
  Recall@IoU0.50 (V1/V2) was foundation `0.0803/0.0669`, then
  `0.8729/0.8515` (epoch1), `0.8873/0.8766` (epoch2),
  `0.8945/0.8843` (epoch4), and `0.8938/0.8815` (epoch6). IoU0.25 reached
  about `.99` in both domains, while epoch-6 IoU0.75 was `0.7215/0.6119`.
  Size-stratum IoU0.50 was small `0.9136`, medium `0.9445`, and large
  `0.7085`.
- Training logs contain occasional `grad_norm: nan` values, but loss values
  remained finite, all six checkpoints were written, and every formal shard
  produced finite predictions. This telemetry is reported rather than
  filtered. The final epoch-6 checkpoint SHA-256 is
  `a1ff39f9c0bd59a5fd2b14968a528fc11e882b8fa6a3e85a9b64253a1cdc5ccc`.
- Decision: the valid V1/V2 generic gate did not reach `.90/.90` after the
  registered six-epoch maximum. Expression adaptation was not started because
  it is downstream of the generic gate. Added the corrected adaptation report,
  failure decomposition, and four valid formal JSONs. Final status is
  `LOCATEMOT_U_BLOCKED_GENERIC_GATE_AFTER_MAX_ADAPTATION` with
  `STOPPED_PENDING_SUPERVISOR_REVIEW`; U2 was not launched.

## LocateMOT-U U1R2 — measurement repair, class-aware refinement, and foundation completion — 2026-10-01

- Contract hypothesis: the remaining U1 failures were caused by a measurement/data contract mismatch rather than a need to replace the MM-GroundingDINO-B foundation. The source `labels_with_ids` fields are normalized top-left `xywh`; the repaired formal gates preserve source extents and use Top150 IoU.50 for generic proposals and Top100 IoU.50 target coverage for expressions. The previous U1-A-v1 files remain preserved as historical artifacts and were not overwritten.
- Coordinate/source audit: legal calibration/validation contains 33,526 physical GT boxes. 195 V1 boxes extend below the image at `y1>height` (166 large, 27 medium, 2 small); no x-side overflow exists. Same-frame geometry matching to official KITTI labels gives mean IoU `0.7414883`; Refer and official track IDs are separate namespaces. Decision: retain source extents and recover classes by geometry; do not clip evaluation GT.
- Data hypothesis: a class-aware physical-frame union can repair localization while removing V1/V2 duplicate training. The fit stream contains 2,545 images and 10,502 boxes; 3,424 V1/V2 overlaps were removed, 580 conflicting boxes and 21 missing class matches were recorded, and observed classes are car 6,787, pedestrian 3,700, cyclist 13, van 2. Decision: retain the stream with fixed class policy and explicit low-confidence audit.
- Numerical stability: the prior BF16 log has three finite-loss `grad_norm:nan` records (epoch/iter 1/50, 3/100, 6/100). A fresh FP32 one-step audit from epoch4 found 981 trainable parameters, total loss `40.27210998535156`, zero nonfinite gradients, and zero no-gradient parameters. Decision: R2 uses FP32/TF32, no AMP/BF16.
- Generic refinement: one bounded FP32 epoch from the frozen epoch4 checkpoint passed the corrected gate: V1/V2 Top150 IoU.50 `0.982716/0.971837`; small/medium/large `0.937467/0.989984/0.997825`. Decision: freeze the R2 epoch1 generic checkpoint and skip epoch2/mixed fallback.
- Expression hypothesis: whole-expression positive spans with all referred boxes, legal empty frames, and preserved V2 missing-target references can add language-conditioned candidate coverage without erasing generic proposals. One FP32 epoch completed from the selected generic checkpoint. Expression Top100 IoU.50 is V1/V2 `1.000000/0.961367`; the same checkpoint's generic regression is V1/V2 Top150 IoU.50 `0.986592/0.958739`.
- Language audit: on exactly 500 legal validation same-image pairs with a same-video sentence from a different target set, true versus shuffled Top10 recall is `0.959333/0.871133` and AP50 is `0.736901/0.491159`. Both strict comparisons pass, so language conditioning is measurable rather than generic-box-only behavior.
- External evaluator: synthetic perfect/empty/ID-switch and cross-evaluator arrays pass with maximum difference `0.0`. TempRMOT evaluated videos 0012/0014 occur in its declared training pool; its HOTA 94.679 is reclassified as `PIPELINE_SANITY_ONLY_TRAINING_OVERLAP`.
- Decision: the final shared expression checkpoint satisfies all U1 hard gates. `official_test_labels_read=false`, `screening_gt_used=false`, V2 source-target-ID-missing references remain explicit (157, silently dropped=false), and no U2/U3/U4 work started. Set `LOCATEMOT_U_READY_FOR_U2` and stop pending supervisor review.

## LocateMOT-U U2 — shared persistent track core and portability stop — 2026-10-02

- Provenance: working branch `codex/u2-shared-persistent-track-core-20261001`,
  authoritative local base `56507f4aa9a12e76d6f9f6702f0f0b38163666ec`, and
  immutable U1 foundation SHA
  `4efc35ce70be5cddf2c4d9a0f91bc4edef3e9ec5277a819211ba4ac42e0d7d26`.
- Implementation: replaced the U0 tracking placeholders with one shared U2
  core (`PerceptionBridge`, independent identity/semantic candidate encoders,
  `UnifiedTrackState`, four-layer identity set interaction, GRU identity
  memory/anchor/EMA, association/new/no-match, lifecycle, motion, Hungarian
  rollout, and bounded OVMOT residual). The runtime is
  `locatemot/unified/runtime/u2_online_tracker.py`; no RMOT or prompt tracking
  was launched.
- Contract evidence: `tools/unified/u2_causal_contract.py` passed birth,
  occlusion, no-future, MOT specification invariance, OVMOT isolation, and
  online-state save/reload. Actual-model sharing audit passed with
  `shared_non_head_fraction=0.9378111093950003`.
- U2-P data manifest SHA-256 is
  `575bd349e00bab23634c6929c04513149afec838de419bb3f1a53469fafc7fda`.
  KITTI legal development portability passes (V1/V2 Top150 @.50
  `0.986592/0.958739`), while BDD100K, DanceTrack, MOT17, MOT20, TAO, and
  C-TAO are missing locally. Candidate-cache SHA is `not built`.
- Training command: not launched; GPU count `0`, optimizer steps `0`, wall
  clock training runtime `0`. Simple MOT, MOT-only, and joint OVMOT metrics
  are intentionally `null` because the registered portability protocol is
  incomplete.
- Decision: `LOCATEMOT_U_BLOCKED_U2_PERCEPTION_PORTABILITY`; stop pending local
  acquisition/audit of the missing streams. Official test labels and
  Refer-KITTI official evaluation labels remain unread.
- Validation note: the 2-process CUDA forward/backward smoke passed with the
  same MOT task on both ranks. An 8-process smoke could not initialize because
  all eight A800 devices were occupied by an unrelated process; it was
  terminated before any training step and is not counted as a pass.

## LocateMOT-U U2 — acquisition recovery — 2026-10-11

- Revalidated local descendant `666ab44` and the actual U1R2 checkpoint SHA
  `4efc35ce70be5cddf2c4d9a0f91bc4edef3e9ec5277a819211ba4ac42e0d7d26`.
  U1 references and the checkpoint remain unchanged.
- Observed MOT17/MOT20 extraction counts of 33,810/13,430 files, with nested
  release roots. TAO's four volume roots exist but contain zero files; no
  training or download worker was live. The old download module spawned
  workers while being imported by each child, recursively launching itself.
- Replaced that launcher with one guarded entry point and four thread workers
  running bounded download clients. Keep the registered four-volume assignment,
  prioritize modal train/validation inputs, validate downloaded sizes, and
  record actual per-file SHA-256 for source provenance. A fresh direct-network
  README transfer passed: 9,189/9,189 bytes, so existing Baidu authentication is
  usable. This does not imply that the dataset or portability gate is complete.
- Identified a separate eligibility bug: the old portability artifact could
  return PASS solely from directory availability while all new-domain metrics
  were null. Correct the audit and eligibility calculation before any training;
  metric definitions, split rules, thresholds, and seeds remain unchanged.

## U2 frozen-foundation MOT portability — 2026-10-11

Hypothesis: the selected U1R2 foundation may need bounded adaptation on crowded/non-road pedestrian domains. Registered ByteTrack inclusive train-half ranges, FRCNN physical-video deduplication, `person .`, positive mark-1/class-1 GT and unmodified source box geometry before measurement. The loaded checkpoint SHA matched the immutable U1R2 reference, with no missing/unexpected weights.

Ran all 7,115 dev frames on GPUs 1–7 in one parent launch/wait command, leaving an externally occupied GPU 0 alone. MOT17: 2,652 images / 53,890 GT; Top150 0.787233, Top300 0.806253. MOT20: 4,463 images / 615,137 GT; Top150 0.731010, Top300 0.884673. Exact coverage and monotonic count checks passed; partial measurements are rejected. MOT17-02/04 and crowded MOT20-03/05 dominate the shortfall. Retain the measured failure; do not alter visibility filtering, vocabulary, frame scope or gates to improve the reported values.

Track training is held. At most two epochs of mixed U2-P perception adaptation are justified after the missing domains/protocols are ready, with U1 replay and full retention gates. MOT20 Top300 has not met the preferred 0.90 or been accepted as a runtime override. TAO four-volume acquisition/extraction continues. Cloud BDD resources are static detection rather than MOT 2020. Started official DanceTrack train/val and C-TAO base-only acquisition, pinning release revisions; aria2 TLS failed, curl direct HTTP-206 probe succeeded and curl replaces it. No tracker or RMOT/prompt training was launched.

U2-P follow-up: C-TAO base-only source downloaded and inspected: 490,210 train images, 1,489,637 base-category boxes, 500 videos; supervision frequencies f/c only and no validation paths. LVIS v1 full vocabulary frozen (1,203 names; 866 f/c base, 337 rare novel). Native MMDetection chunking passed a real TAO-image smoke with 38 chunks and exactly one backbone call, followed by class-agnostic IoU-0.8 merge and global Top300. This is runtime evidence, not a substitute for full TAO coverage. Original modal TAO validation has 36,375 images / 113,112 boxes; 314 boxes lack a LVIS-v1 synset mapping, so the benchmark conversion/audit must explicitly account for them. Original BDD MOT 2020 image host dl.yf.io is live; started eight train/val archives (38,886,924,336 bytes) with original published MD5 checks, four bounded workers and eight connections/file. Official 2021 tracking-label S3 URL returns 404; keep BDD labels missing while locating a valid original release.

## U2 TAO/C-TAO portability registration — 2026-10-12

Continue from the published full MOT-half failure measurements, preserving the immutable U1R2 checkpoint. Plan: freeze all official TAO validation images plus every original annotated C-TAO training-audit frame; preserve the full dense C-TAO set for later identity training. Use all 1,203 LVIS-v1 names, the already verified 32-name chunks, one image backbone extraction, class-agnostic IoU-0.8 merge and global Top300. Count all original modal TAO GT including the 314 annotations without a v1 synset mapping; report those separately with source-frequency grouping rather than dropping them. Wait for CRC-completed source archives internally in the single evaluation launcher while downloads/extraction continue. In parallel, finish Dance extraction and locate original BDD tracking labels. No track training is eligible yet.

## U2 data completion and full portability launch — 2026-10-12

Hypothesis: recovering the original data and measuring complete dev scopes will distinguish foundation domain failure from incomplete measurement. DanceTrack download/CRC extraction completed; preserved 32/8/25 project videos and exact 33,772/8,024/25,508 frames. TAO modal train/val and all 490,210 C-TAO training images now pass availability coverage (six available streams, BDD pending). Frozen full TAO validation (36,375 images, 110,055 base / 3,057 novel GT) and original C-TAO train audit (17,131 images, 53,263 base GT); retain all 314 unmapped original TAO GT units. Immutable-foundation real smokes passed in both chunked domains and DanceTrack; full audits run on GPUs 1–6 and 7 respectively. No partial recall promoted; MOT gates still fail and track training remains held.

BDD root cause: old official annotation URLs are dead and the user cloud BDD folder is static detection. Found a pinned HF archive whose directory contains all 1,400/200 raw tracking JSON videos and validation images. Download/hash/selected-member CRC extraction is active; source labels have not yet established availability. Added original toolkit category/ignored/crowd and inclusive Box2D handling; raw-label full-scope validation will precede BDD measurement. Original seven train-image archives continue downloading, with an internal blocking extractor. Keep these recovery changes; no learned model, seeds or gates changed.

## U2 BDD raw tracking recovery validated — 2026-10-12

Recovered and SHA/CRC-validated the pinned mirror: all 1,400 train and 200 val raw JSON videos; every 39,973 validation image matches the original release ZIP member size/CRC. Frozen 278,079 train / 39,973 val frames and the fixed eight-category vocabulary before inference. Full official-reference comparison covers all validation objects: 442,814 positive boxes, 18,625 crowd and 4,073 official ignored-category objects. First strict comparison exposed a 2.84e-14 rounding difference from direct x2+1; changed arithmetic to the official xywh reconstruction order, then every positive box/category/ID matched exactly (no tolerance relaxation). Keep the correction. Full BDD portability is queued after the DanceTrack worker on GPU 7 with a real-image smoke first. The first original train-image archive is CRC-extracted (200 videos, 39,418 frames); remaining train-image downloads/extraction continue. Numerical MOT gate failures and track-training hold remain unchanged.

## U2-P bounded perception adaptation preparation — 2026-10-12

Plan before any adaptation result: use a fixed 20,000-anchor replay epoch, maximum two epochs, seed 20261012. U1 generic/V1-expression/V2-expression proportions 0.10/0.15/0.25; BDD/Dance/MOT17/MOT20/TAO base detection each 0.10. Stratified video sampling covers every training video where the allocated budget permits; cycle deterministically when replay pools are smaller than their allocations. Preserve original U1 replay boxes/phrases; use only official train or registered MOT-half train frames. TAO class supervision uses mapped LVIS-v1 f/c categories only, with novel/unmapped annotations excluded from class and identity supervision. Encode detection classes as positive phrase spans through the existing native ODVG grounding path. Empty positives retain a fixed absent base-class capti…97554 tokens truncated…ndidate MOT identity rows over the 32,768-token limit. The cause was repeated dense N×N IoU and center-distance CSV matrices in every MOT row, not missing media or a changed candidate source. Added the explicit `compact_competition_v1` manifest transform: native JSONL remains immutable; exact row IDs, targets, media paths and candidate boxes are preserved; all positive-IoU edges retain native rounded IoU and center distance; zero-IoU geometry is derivable from the retained boxes. The v2 CPU plan rendered all 131,990 rows with zero drops (p50/p90/p99/max 7,939/12,173/16,050/28,156 tokens; 8,250 world-4 steps), with no GPU or optimizer updates. The old v1 waiting owner was superseded before any training; a separate v2 owner is waiting on G2/G1/bundle/held-out gates and exclusive GPUs. Native TAO category rows remain pending, so this is still a registered Base teacher-control course rather than a final OVMOT claim.

## 2026-10-21 — native YesTrack v2 queued behind an exclusive GPU7 gate

Registered the real native YesTrack corpus with 726 train rows and 390 fixed
validation rows (train SHA
`065337d2fedca2377d02b94e57c9eb3778aeca09aeb1a8b9463429fa15bbee66`, validation
SHA `d63a9fb0d54b8a1dc5d12b0960a48cb75c56ab786671f955fb317e196b3033d8`). The
CPU rendering plan passed with zero rows over 32,768 tokens (max 26,159; 91
planned optimizer steps per epoch, 273 over three epochs). Added a real GPU
owner that waits for GPU7 to have no
compute PID and at least 32 GiB free, trains native multimodal LoRA for three
epochs, then evaluates frozen base and tuned adapter on identical held-out rows
using AUROC, PR-AUC, accuracy, Brier, ECE, false rejection and false
acceptance. The owner does not stop other users and writes terminal receipts
only after training and both evaluations succeed. At registration GPU7 was
occupied, so no YesTrack metric is claimed; prior 24-row readout controls remain
engineering evidence only. Tracking HOTA/AssA/IDF1/IDSW still requires a later
fixed-candidate physical-baseline integration.

## 2026-10-21 — live G2 Clean checkpoint refresh at step 12002

A single authoritative read of the formal event log found G2 Clean at event step 12002/18227, with finite loss 0.298694, gradient norm 10.215737, peak memory 12.829 GiB, and the registered 8-micro-batch/16-sample contract. The periodic adapter checkpoint is `adapter-step012000.pt` and `checkpoints/latest_training_state.pt` is present; the terminal receipt is still absent. The recent 500-step rate is 11.615 s/step, a linear remainder of approximately 20.1 hours. G1, bundle, held-out, YesTrack, and zero-drop three-task v2 remain waiting owners; no metric claim was promoted and no protected process was modified. Evidence: `docs/GROUNDJEV/STATUS_SNAPSHOT_20261021_G2_STEP12002.{json,md}`.

## 2026-10-21 — process-scope audit confirms no live MOT-only/J1 optimizer

A fresh process-table audit found G2 Clean as the only active GroundJEV optimizer (event step 12012/18227 on GPUs 3/5). G1 GPU5, deployment bundle, independent held-out, YesTrack and zero-drop three-task v2 are live dependency/resource waiters. No independent MOT-only or legacy J1 OneJev optimizer is live; historical checkpoints and logs remain preserved. Foreign PRISM jobs were untouched and no process was stopped. Evidence: `docs/GROUNDJEV/PROCESS_AUDIT_20261021.{json,md}`.

## 2026-10-21 — G2 step-12000 recovery checkpoint validated

Read the protected G2 `latest_training_state.pt` and `adapter-step012000.pt` with the registered local trainer environment. Both load successfully and carry format `groundanything-g2-projector-qwen3-lora-v2-strict`, step 12000, manifest SHA `f682519a…06e71`, 293 trainable tensors/signature entries, optimizer and scheduler state, RNG state, two rank RNG paths, sampler position 96000 and world size 2. The terminal receipt is still absent, so this proves recoverability only and no performance claim. PyTorch 2.6's default weights-only reader rejects the stored NumPy RNG object; the controlled audit used trusted `weights_only=False` and did not modify files. Evidence: `docs/GROUNDJEV/G2_CHECKPOINT_VALIDATION_20261021.{json,md}`.

## 2026-10-21 — G2 Clean dataset scope audited

The protected G2 Clean manifest contains 583,251 TRAIN rows from AerialMind (69,358), AnimalTrack (11,543), DanceTrack (41,796), MOT17 (15,948), MOT20 (5,616), RMOT26 (294), Refer-KITTI-V2 (379,341), SportsMOT (28,574) and TAO-Amodal (30,781). It is GroundAnything perception adaptation (projector + Qwen3 LoRA), not J1/OneJev identity training. MOT20-05 is explicitly held out; BURST is excluded for lack of a legal TRAIN image root; TAO novel annotations are excluded. Manifest SHA is `f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71`. Evidence: `docs/GROUNDJEV/G2_CLEAN_DATASET_SCOPE_20261021.{json,md}`.

## 2026-10-21 — GroundJEV Phase A real G2 single-backbone smoke

Implemented the isolated `locatemot/groundjev/coupled` path: frame/model-provenance checked shared state, G2 feature bridge, typed dynamic identity slots, dot-product JEV readouts, explicit Yes/No/admit/lifecycle/memory heads, causal identity memory, safe lifecycle control, assignment-conditioned memory writes and a one-to-one runtime. CPU backward and causal runtime tests passed. A real GPU smoke loaded the official GroundAnything visual module from the registered G2 base checkpoint (`3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2`), encoded one MOT17 image once, pooled two real boxes into 2x2560 region tokens, and completed finite typed loss/backward/optimizer-step: total loss 4.179936, 3.07 GiB peak allocated, 14.43 s, bridge forward count 1, independent VLM instances 0. This is engineering evidence only; no tracking or scientific gain is claimed, and contextual language-state return plus identity-conditioned grounding remain pending. The first py310 attempt failed before model construction because `pybase64` is absent; the successful run used the existing GroundAnything serving environment and the same immutable checkpoint. Evidence: `docs/GROUNDJEV/PHASE_A_REAL_SMOKE_20261021.{json,md}`.

## 2026-10-21 — GroundJEV Phase B choice/Yes-No pilot completed

A bounded real pilot reused 32 MOT17-02-FRCNN TRAIN frames and 992 prior native GroundAnything proposal boxes, re-pooled once per frame with the G2 base visual checkpoint SHA `3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2`. The frozen G2 evidence fed typed Identity Choice, Admit, Yes/No, Lifecycle and Memory readouts; 24 frames received 192 optimizer updates and the final 8 frames were a same-video temporal development holdout. Identity accuracy changed train 0.0618->0.6720 and temporal-dev 0.0363->0.6492; admit accuracy changed train 0.1022->0.9247 and temporal-dev 0.1331->0.9153. An actual eight-frame replay without teacher memory produced three persistent predicted tracks with one cached G2 feature forward per frame. This is a teacher-forced engineering control, not an independent HOTA/generalization result; language hidden-state coupling and identity-conditioned grounding remain unmeasured. The first replay failed on the legacy/new memory field mismatch and was fixed without changing data or budget. Evidence: `docs/GROUNDJEV/PHASE_B_PILOT_20261021.{json,md}`.

Checkpoint receipt finalized for the Phase B pilot: `adapter-step000192.pt`, SHA256 `c96f27c9c8de5a01a36036c78c4e4a36c467bb96a0ca5f7d0eee27d93e32c904`; the binary remains local under the registered output directory and is not pushed to GitHub.

## 2026-10-21 — native G2 Clean hidden-state probe completed

The protected G2 Clean base plus `adapter-step012000.pt` (adapter SHA `f0dfe4f5efd6bd2f0e8ebe8f224ae060e8b0422b36571697fb4fa9efa6d85ae2`, step 12000, manifest SHA `f682519a…06e71`) completed the native GroundAnything DLM clean path on real row `mot17_MOT17-02-DPM_000001_dense_clean_v2` from MOT17 TRAIN. The real image was encoded once; the clean embedding and hybrid-language hidden state were both `[1, 1152, 2560]`, with 1,008 image-pad hidden vectors. No independent VLM instance was created. The probe used the formal BF16 autocast boundary and the existing process-local Transformers 5.17 `cache_position` compatibility shim. The first dtype and API failures were preserved and fixed at the probe boundary. This establishes native hidden-state availability for the shared-backbone work only; it is not a tracking metric, identity gain, grounding improvement, or generalization result. Evidence: `docs/GROUNDJEV_COUPLED/PHASE_A_NATIVE_HIDDEN_PROBE_20261021.{json,md}`.

## 2026-10-21 — native G2 clean DLM coupled Phase C smoke completed

The new `NativeG2CleanDLMProvider` restores the protected G2 step-12000 adapter in memory, renders one real MOT17 TRAIN row, and uses one native clean DLM forward to produce both 1,008 image hidden tokens and a `[1, 2560]` non-image contextual token. The same-backbone bridge pooled two real candidate boxes, fused the contextual token into the typed GroundJEV readout, and completed a finite GPU backward plus optimizer step: total loss 5.533803, clipped gradient norm 73.300224, 12.059 GiB peak allocation, 16.91 s, bridge/native forward counts both 1, independent VLM instances 0. This proves the visual+clean-language coupling interface runs on a real checkpoint; it is still an engineering smoke with no HOTA/IDF1, grounding recall, identity gain or generalization claim. PyTorch emitted a BF16/FP32 RMSNorm fallback warning in the frozen language path; it did not produce a non-finite value and remains recorded in the receipt. The formal G2 run directory was untouched. Evidence: `docs/GROUNDJEV_COUPLED/PHASE_C_NATIVE_DLM_COUPLED_SMOKE_20261021.{json,md}`.

## 2026-10-21 — native G2 context readout C1 pilot completed

A bounded eight-frame MOT17-02-FRCNN pilot used G2 step 12000 with the official DLM frozen. Each of 8 real frames had one native DLM forward; the downstream GroundJEV typed readout received the pooled image regions plus same-frame clean-language context. Four frames were train and four same-video temporal development, with previous-frame GT-matched region tokens as explicit teacher-forced causal slots. Twelve readout updates changed identity accuracy train 0.032258→0.725806 and temporal-dev 0.032258→0.403226; admit accuracy changed train 0.911290→0.911290 and temporal-dev 0.790323→0.911290. This is a bounded learnability control only: no HOTA/IDF1, independent-video or final tracking claim. The first attempt completed the loop but failed at receipt writing because the output directory was absent; the directory fix was the only rerun change and the failed attempt is excluded. Evidence: `docs/GROUNDJEV_COUPLED/PHASE_C_NATIVE_CONTEXT_PILOT_20261021.{json,md}`.

## 2026-10-21 — active goal snapshot at G2 step 12153

The protected formal G2 Clean owner is alive on GPUs 3/5 at event step 12153/18227; recent event rate is 11.614 s/step and the linear remaining estimate is 19.6 h. The latest durable checkpoint is `adapter-step012000.pt`, with no terminal receipt. G0 MOT20-05 validation-half export is terminal for 1,657 frames (HOTA 0.073346961, IDF1 0.059598451, IDSW 1933); this remains a validation-half control rather than a full TEST claim. G1, bundle, held-out, YesTrack and compact three-task OneJev owners are alive and waiting on their registered dependencies/resource gates with no recorded failure; compact OneJev optimizer updates remain 0. The process audit found no live MOT-only optimizer. Evidence: `docs/GROUNDJEV/ACTIVE_GOAL_STATUS_20261021_G2_STEP12153.{json,md}`.

## 2026-10-21 — merged Phase C causal memory input contract

Merged the isolated single-backbone memory-input implementation into the GroundJEV evidence branch. `MemoryTokenProjector` used by the typed readout remains compatible; the new `IdentityMemoryTokenProjector` and `inject_identity_memory` insert only causal identity tokens before the registered answer index, return the insertion offset for label shifting, and leave the disabled path tensor-identical to the G2 baseline. The focused contract tests passed in the serving PyTorch environment (3 tests) and the existing Phase A core functions passed (3 checks); coupled sources compile. This is an input-contract result only: the official DLM decoder hook, generated-box recall and C0/C1/C2 HOTA/IDF1 evaluation remain pending.

## 2026-10-21 — blocked accidental teacher-only OVMOT launch

The registered real three-task v2 manifest contains 131,214 native MOT rows, 346 native RMOT rows, and 430 OVMOT Base teacher-control rows; its 40-frame native TAO archive has no category labels and is validation-only. The waiting four-GPU OneJev owner had zero optimizer updates. I stopped that waiting owner before any launch, preserved its state snapshot, and marked it `PAUSED_FOR_NATIVE_OVMOT_ROWS`. The queue gate now requires an explicit native OVMOT TRAIN candidate/annotation join (`native_ovmot_training_rows=true`, `ovmot_training_mode=native`) and returns `WAITING_NATIVE_OVMOT_ROWS` otherwise. No three-task training or OVMOT gain is claimed; the historical source course remains preserved.

## 2026-10-21 — active goal snapshot at G2 step 12195

A single authoritative read of the protected G2 event log found event step 12195/18227 with finite loss 0.225930949, gradient norm 9.566365, peak memory 12.828853 GiB, and a 500-step rate of 11.618 s/step (linear remainder about 19.5 h). The latest durable checkpoint remains adapter-step012000.pt and the terminal receipt is absent. G0 MOT20-05 remains a complete 1657-frame validation-half natural-visibility control with HOTA 0.073346961 and IDF1 0.059598451; it is not a full TEST claim. The compact three-task queue is paused before optimizer launch because native OVMOT TRAIN rows are missing; its optimizer update count remains zero and the guard requires native OVMOT provenance/mode. No final G1/held-out/OneJev/YesTrack result was promoted. Evidence: docs/GROUNDJEV/ACTIVE_GOAL_STATUS_20261021_G2_STEP12195.{json,md}.

## 2026-10-21 — native OVMOT TRAIN join pipeline prepared

Added `tools/groundjev/prepare_native_ovmot_train_join.py`, a CPU-only joiner for a future complete native GroundAnything TAO TRAIN stream. It validates `/frames/train/` provenance, requires native region features and GT-free model metadata, preserves candidate boxes, assigns official Base category targets only through deterministic one-to-one IoU >=0.5 matching to TAO TRAIN annotations, and emits the existing hierarchical category questions with on-demand real-image crop recipes. The registered TAO TRAIN source contains 18274 images, 54639 annotations and 765 Base categories mapped to the 1203-class vocabulary; 11 categories are explicitly unmapped. The current 40-frame/35228-candidate native TAO archive is validation-only and is rejected by the TRAIN path guard. `py_compile`, one-frame synthetic preflight, and validation-scope rejection passed; no GPU or optimizer update occurred. The compact three-task owner remains paused pending a complete native OVMOT TRAIN archive. Evidence: docs/GROUNDJEV/NATIVE_OVMOT_TRAIN_JOIN_20261021.{json,md}.

## 2026-10-21 — TAO TRAIN native OVMOT scope registered

Registered a CPU-only scope manifest for all 18274 TAO TRAIN images across 500 videos. The runtime manifest contains only absolute image paths and the fixed 1203-category query vocabulary; it carries `training_on_this_scope=true` for an explicit export gate but no GT fields, labels or evaluation data. `stream_tao_native_candidates.py` now requires `--allow-training-scope` for this manifest and rejects validation manifests with that flag; the scope gate was tested without importing the model service. The exporter was not launched because current GPUs remain occupied, so native candidate rows and optimizer updates are still zero. Evidence: docs/GROUNDJEV/TAO_TRAIN_NATIVE_SCOPE_20261021.{json,md}.

## 2026-10-21 — corrected process audit at G2 step 12235

A corrected process-table audit (the first username-filtered attempt was invalid because the account is `lyq`) found the protected G2 parent/ranks alive at 12235/18227 on GPUs 3/5, G1/bundle/YesTrack waiters alive, and no registered MOT-only or legacy J1 OneJev optimizer command. The compact three-task owner remains paused before optimizer updates because native OVMOT TRAIN rows are absent. Foreign GPU jobs were not touched. Evidence: docs/GROUNDJEV/PROCESS_AUDIT_20261021_G2_STEP12233.{json,md}.

## 2026-10-21 — active goal snapshot at G2 step 12241

The protected G2 Clean event log reached 12241/18227 with finite loss 0.195041649; the last 500-step rate is 11.626 s/step and the linear remainder is about 19.3 h. The durable checkpoint remains step12000 and no final receipt exists. G0 remains a 1657-frame MOT20-05 validation-half control (HOTA 0.073346961, IDF1 0.059598451); the compact three-task queue remains paused with zero optimizer updates pending native OVMOT TRAIN rows. Evidence: docs/GROUNDJEV/ACTIVE_GOAL_STATUS_20261021_G2_STEP12241.{json,md}.

## 2026-10-21 — GroundJEV fixed-candidate MOT20-03 replay establishes failure boundary

Added a reproducible CPU evaluator for the immutable native GroundAnything candidate/region cache. The current 192-step coupled readout replayed all 128 registered MOT20-03 TRAIN frames (14,784 candidates, 128 cached feature calls) without a second VLM and generated 152 predictions on 76 frames. TrackEval returned HOTA 0.0279006409, DetA 0.0099894107, AssA 0.0779470005, IDF1 0.0052606731, IDSW 1, FP 1, FN 14,524; 11,462 candidates were explicit REJECT. The cache predates the current G2 Clean terminal checkpoint and overlaps TRAIN scope, so this is a fixed-candidate association/runtime diagnostic, not independent G2 or end-to-end evidence. The first evaluator attempts exposed only direct-entry `sys.path` and NumPy-2/old-TrackEval compatibility issues; both were fixed locally without changing metric code or inputs. Evidence: `docs/GROUNDJEV_COUPLED/PHASE_B_FIXED_CANDIDATE_EVAL_20261021.{json,md}` and the registered output receipt.

The same-input historical compact control (checkpoint `rmot_native72_stage_v1/broad_native_training/checkpoint.pt`) returned HOTA 0.336514881, DetA 0.309884333, AssA 0.365702041, IDF1 0.359321689, IDSW 159, FP 119 and FN 9,832. It is an offline multi-component reference, not a single-backbone result. The gap confirms that the coupled readout's 0.027900641 HOTA is a real association/admission failure and should trigger a bounded repair study rather than a long joint finetune.

## 2026-10-21 — Phase C causal memory prefill contract

Added `locatemot/groundjev/coupled/causal_memory.py`, which inserts causal identity tokens only at the terminal prompt boundary used by GroundAnything causal bbox generation, extends attention/position/cache inputs, and preserves the disabled path by tensor identity. A seven-test CPU contract suite passed; a fake language tower showed a changed next-token logit after memory insertion with exactly one prefill call and no second VLM. Real G2 GPU prefill, correct-versus-shuffled memory box generation, and HOTA/IDF1 effects remain pending because GPUs 3/5 are protected by formal G2 and the other devices have foreign compute owners. Evidence: `docs/GROUNDJEV_COUPLED/PHASE_C_MEMORY_PREFILL_CONTRACT_20261021.{json,md}`.

## 2026-10-21 — authoritative G2 progress refresh at step 12294

A single non-polling audit found the protected formal G2 Clean parent/ranks alive on GPUs 3/5 at event step 12294/18227. The event is finite (loss 0.205467, gradient norm 9.551106, peak memory 12.828834 GiB); `adapter-step012000.pt` and `latest_training_state.pt` remain the latest durable checkpoints and `final_receipt.json` is absent. G1, bundle, held-out, YesTrack and compact three-task owners remain dependency/resource waiters. No protected process or experimental protocol was changed. The causal-memory prefill contract remains engineering-only pending an exclusive GPU window for real generated-box comparison.

## 2026-10-21 — fixed-candidate action-distribution repair completed

The first MOT20-03 fixed-candidate replay showed a candidate pool with 86.85% GT-box coverage at IoU >= 0.5, but the short MOT17-trained readout selected `REJECT` for 77.54% of 14,784 candidates and created only two births. A bounded repair trained only on the first 64 frames of the registered MOT20-03 TRAIN clip, weighted identity-choice classes from that training half with an inverse-frequency cap of 4x, and froze the final 64 frames before optimization. After 180 CPU updates, temporal-dev readout identity accuracy was 0.871720 and admit accuracy 0.903860; predicted-memory runtime replay returned HOTA 0.302815, DetA 0.235861, AssA 0.388886, IDF1 0.257729, IDSW 18, FP 3, FN 5,634 on 1,891 predictions across 61 frames. The 64 feature calls used the cached single-backbone state and no second VLM. This is a TRAIN-scope mechanism repair, not an independent benchmark or G2 Clean result; its scope and teacher-forced classification boundary are explicit in `docs/GROUNDJEV_COUPLED/PHASE_B_FIXED_CANDIDATE_REPAIR_20261021.{json,md}`.
## 2026-10-21 — redirect goal audit at G2 step 12326

A single authoritative audit confirmed the registered MOT20-05 G0 native export is terminal for 1,657 validation-half frames (HOTA 0.073347, IDF1 0.059598) and found no live GroundJEV MOT-only optimizer; no historical process was stopped or overwritten. Formal G2 Clean remains alive on GPUs 3/5 at step 12326/18227, with a 500-step rate of 11.6 seconds per step and a linear remainder of about 19.0 hours. The latest durable checkpoint is `checkpoints/adapter-step012000.pt`, and the final receipt is absent. G1 GPU5, bundle, held-out and YesTrack owners remain waiting; the corrected real three-task OneJev owner remains paused at zero optimizer updates because native TAO TRAIN OVMOT rows are not present. Evidence: `docs/GROUNDJEV/ACTIVE_GOAL_STATUS_20261021_G2_STEP12326.{json,md}`.

## 2026-10-21 — cross-disk native OVMOT archive audit at G2 step 12345

A read-only search of `/data0/user`, `/data1/user` and `/data3/user` found no complete TAO TRAIN native candidate archive. The only existing native TAO candidate archive remains a 40-frame `frames/val` stream and is ineligible for TRAIN joining. The corrected real three-task queue remains paused with zero optimizer updates; no validation archive was relabeled and no teacher-only fallback was launched. Evidence: `docs/GROUNDJEV/NATIVE_OVMOT_ARCHIVE_AUDIT_20261021.{json,md}`.

## 2026-10-21 — GroundJEV joint identity corpus rebuild contract and Phase B ablations

Added `tools/groundjev/coupled/build_joint_identity_corpus.py` and its focused
contract tests. The builder normalizes registered MOT/RMOT/OVMOT rows without
inventing boxes or targets, records candidate provenance and causal memory,
rejects future/test evidence, and has an explicit native-only gate that rejects
teacher crops. A real 12-row sample from the registered Phase-B corpus produced
5 MOT, 3 RMOT and 4 OVMOT rows; 8 rows had native candidates and 4 remained
explicit Base teacher/empty-candidate rows, so `native_ovmot_training_rows` was
false. The existing G2 step-11000 B0/B1/B2 bounded readout receipts were
consolidated into `docs/GROUNDJEV_COUPLED/PHASE_B_ABLATIONS_20261021.{json,md}`.
The three 200-update arms and 8-frame causal replay are engineering evidence
only; no official tracking metric or OVMOT native-training claim was promoted.

The native-only corpus gate was then strengthened to reject rows carrying an
explicit `gt_in_model_input`/`GT_IDs_in_model_input` flag even when their
candidate-source string looks native. The focused contract suite now has 12
passing checks; no source rows or training queues were changed.

The Phase-B joint corpus was also audited against all 583,251 G2 Clean TRAIN
rows. There are 678 exact image-path overlaps covering 290 images (221
identity-pair, 119 occlusion-recovery, 338 RMOT); the 430 OVMOT teacher rows
have no exact overlap. The audit reads no official-test labels and marks the
joint corpus as TRAIN-scope controls only. Evidence:
`docs/GROUNDJEV_COUPLED/DATA_OVERLAP_AUDIT_20261021.{json,md}`.

## 2026-10-21 — targeted native TAO archive rescan

Re-read every discovered TAO candidate archive under `/data2/user/LocateMOT_GroundJEV/outputs/groundjev`. The two small candidate controls and the 40-frame streams all resolve to `frames/val`; the larger 40-frame stream has 35,228 candidates and a native feature cache but is explicitly a validation stream. One full Base/Novel stream is empty and failed, and the registered 18,274-image/500-video TAO TRAIN manifest has `native_candidate_export_started=false` and `native_ovmot_training_rows=false`. No complete TAO TRAIN native candidate archive exists, so the corrected three-task queue remains paused at zero optimizer updates. Validation outputs were not relabeled and no teacher-only fallback was promoted. Evidence: `docs/GROUNDJEV/NATIVE_OVMOT_ARCHIVE_RESCAN_20261021.{json,md}`.
## 2026-10-21 — single authoritative G2 progress refresh at step 12417

One non-polling status read found the protected formal G2 Clean ranks alive on GPUs 3/5 at event step 12,417/18,227 with finite loss `0.1803909168`. The latest durable adapter remains `checkpoints/adapter-step012000.pt`; `checkpoints/latest_training_state.pt` is present and `final_receipt.json` is absent. G1, the deployment bundle, held-out evaluation and YesTrack remain dependency/resource waiters. The corrected three-task queue remains `PAUSED_FOR_NATIVE_OVMOT_ROWS` with zero optimizer updates. No protected process or experimental protocol was changed. Evidence is the live G2 run directory plus the refreshed matrix.

## 2026-10-21 — TAO TRAIN native export shards registered

Because the all-GPU snapshot was occupied by protected/foreign jobs, no native TAO export was launched. Instead, added `scripts/groundjev/prepare_tao_train_native_shards.py` and registered eight contiguous, disjoint, label-free manifests covering all 18,274 TAO TRAIN images (500 videos) with parent manifest SHA `835adaadff282088599987de4c5bc906e7d8aca826deb7426cda05598e9db3a4`. CPU coverage validation reports 18,274 covered frames, zero missing and zero duplicates. Every shard remains `native_ovmot_training_rows=false`; this only makes the future native stream resumable/parallel and does not unblock training. The first preflight exposed a slice-length assertion bug, which was fixed before accepting the v2 registration; the failed partial directory was preserved separately. Evidence: `docs/GROUNDJEV/TAO_TRAIN_NATIVE_SHARDS_20261021.{json,md}` and the external shard receipts.

Added `scripts/groundjev/merge_tao_native_shards.py` as the only promotion path after streaming. It checks every shard receipt, parent SHA, exact image order, GT-free metadata, and disjoint full coverage before producing a combined candidate archive. A preflight against the registered-but-unstreamed v2 directory rejected the missing shard receipt at shard 0, as intended; no native TRAIN rows or optimizer updates were created.

The merge contract also passed a synthetic four-frame/two-shard GT-free fixture, covering the successful receipt path without touching dataset or GPU state.

## 2026-10-21 — resumable TAO TRAIN native owner started

Added `scripts/groundjev/queue_tao_native_train_shards.py` and synchronized the runtime stream copy so shard-specific GroundAnything and native-feature ports are supported while the default validation ports remain unchanged. The owner is live in `WAITING_G2_CLEAN_SUCCESS`, restricted to physical GPUs 4/6, and treats any existing compute application as unavailable; GPUs 3/5 and 7 are excluded to protect G2/G1 and held-out/YesTrack. It will run the eight native TRAIN shards and strict merge only after the exact G2 terminal receipt. A first background launch was reaped by the terminal wrapper before writing state; the persistent-session launch wrote the waiting state and made no GPU or foreign-job change. Evidence: `docs/GROUNDJEV/TAO_TRAIN_NATIVE_QUEUE_20261021.{json,md}`.

## 2026-10-21 — authoritative G2 refresh at step 12442

The protected G2 Clean ranks remain alive on GPUs 3/5 at event step 12,442/18,227 with finite loss `0.2287370411`; the latest durable adapter is still `checkpoints/adapter-step012000.pt`, the training-state checkpoint is present, and `final_receipt.json` is absent. G1, clean-bundle, held-out and YesTrack owners remain live dependency/resource waiters; the corrected three-task owner remains paused at zero updates because native TAO TRAIN rows are not available. No protected process or protocol was changed.

## 2026-10-21 — native TAO join owner and native three-task launch gate repaired

Added a CPU-only owner for the complete native TAO TRAIN stream and a separate
resumable owner for the future real MOT/RMOT/OVMOT OneJev run. The join owner
waits for the strict eight-shard merge receipt, runs the checked-out
`prepare_native_ovmot_train_join.py` and `prepare_real_three_task_joint_manifest.py`,
and never launches a model or optimizer. Its initial state is
`WAITING_COMPLETE_NATIVE_TAO_STREAM`. The native three-task owner is now live
as a dependency waiter at `WAITING_NATIVE_OVMOT_MANIFEST`; it has launched no
GPU process or optimizer and will apply the joined-manifest plus terminal
G2/G1/bundle/held-out and exclusive-GPU gates later. The historical
teacher-control queue remains paused at zero updates.

During the audit, the join owner was corrected to resolve repository joiners
from `/data2/user/LocateMOT` because the runtime output tree does not mirror
those source files. The paused compact queue was also corrected to accept the
actual bundle terminal status `G2_CLEAN_DEPLOYMENT_BUNDLE_EXPORTED` while
retaining the historical spelling for old receipts. No data, checkpoint, or
running process was changed. Evidence:
`docs/GROUNDJEV/TAO_NATIVE_JOIN_QUEUE_20261021.{json,md}` and
`docs/GROUNDJEV/REAL_THREE_TASK_NATIVE_QUEUE_20261021.{json,md}`.

## 2026-10-21 — TAO native owner corrected to use a terminal G2 bundle

The TAO TRAIN shard owner was audited before its first GPU launch. Its
original model constant pointed at the published G0 bundle, which would have
made a later native candidate archive impossible to attribute to G2. The owner
now waits for the exact G2 receipt, merges `adapter_projector_final.pt` into
the private `GroundAnything-DLM-G2-CLEAN-NATIVE-TAO-V1` bundle, validates the
569-tensor/export/manifest contract, and only then starts shard services. The
private bundle is never mixed with or written over G0, G1, or the ordinary G2
deployment bundle. The owner process had not started a shard, so it was
restarted with the corrected code while remaining in its G2 wait state.
Evidence: `scripts/groundjev/queue_tao_native_train_shards.py` and
`docs/GROUNDJEV/TAO_TRAIN_NATIVE_QUEUE_20261021.{json,md}`.

## 2026-10-21 — decoupled the terminal G2 bundle from the G1 comparison arm

The clean deployment owner previously waited for both the exact G2 receipt and
G1 GPU5 completion. That ordering could block independent G2 held-out
measurement and native G2 candidate export behind an optional comparison arm.
The owner now waits only for the exact-once G2 terminal receipt, performs the
same CPU-only 569-tensor merge into the separate G2-CLEAN-V2 bundle, and leaves
G1 as an independent GPU5 adaptation/control. The existing owner was still in
its pre-launch wait state, so no model or output was changed; it will be
restarted with the corrected gate. Evidence: `tools/groundjev/queue_g2_clean_bundle_export.py`.

The owner restart also exposed a resumability defect: its old waiting directory
was treated as an existing export and refused to start. The restart guard now
accepts the preserved pre-launch WAITING_G2_AND_G1_TERMINAL state (as well as
WAITING_G2_CLEAN_SUCCESS), refuses live duplicate owners and any partial
bundle, and resumes without deleting the historical state. The corrected owner
is live again in WAITING_G2_CLEAN_SUCCESS.

## 2026-10-21 — native three-task owner no longer logically depends on G1

The new real native MOT/RMOT/OVMOT owner was narrowed to require the native
TAO manifest, terminal G2, independent G2 bundle and held-out receipts. G1
GPU5 adaptation remains an optional comparison receipt; its physical GPU5
ownership is still respected by the four-GPU resource probe. This preserves
G1/G200/G0 controls without making the final-G2 joint route wait on a separate
adaptation arm. The historical paused teacher-control queue was not changed.

## 2026-10-21 — G1 GPU5 waiting owner made resumable

The G1 GPU5 owner had the same pre-launch recovery defect: after a process
restart, its preserved WAITING_G2_CLEAN_SUCCESS directory was treated as an
existing artifact and the owner could not be resumed. The startup guard now
rejects a live duplicate owner and any partial output, but accepts the empty
preserved waiting directory and records `resumed_wait_owner`. The old waiting
owner was stopped and restarted with the registered
`GroundAnything-DLM-TrackProjG1-GPU5-V2` bundle path. It remains at
`WAITING_G2_CLEAN_SUCCESS`; no G1 optimizer or GPU process has started.

## 2026-10-21 — native three-task owner made resumable

The real native MOT/RMOT/OVMOT owner had a weaker restart boundary: a dead
waiting process could rewrite its queue directory without distinguishing a
clean wait from partial training output. Startup now accepts only the known
pre-launch waiting states, preserves the optional runtime config when it was
already created, rejects unexpected files, refuses live duplicates, and records
`resumed_wait_owner`. The existing owner is being restarted only in its
CPU-only `WAITING_NATIVE_OVMOT_MANIFEST` state; no optimizer or GPU process is
started.

## 2026-10-21 — YesTrack waiting owner made resumable

The native YesTrack v2 owner was still waiting for an exclusive GPU7 and had
no optimizer or evaluation artifacts. Its startup guard now resumes only the
known empty `WAITING_GPU7_EXCLUSIVE` queue (and the pre-created `args.json`),
rejects partial queue/training outputs, refuses live duplicates, and records
`resumed_wait_owner`. The waiting owner is restarted without allocating GPU7;
the independent YesTrack run remains pending.

## 2026-10-21 — held-out owner made resumable

The GPU7 G0/G2 independent held-out owner was still CPU-only in
`WAITING_G2_CLEAN_BUNDLE_AND_GPU7`; its directory contained only the current
wait receipt and the preserved historical pre-MOT receipt. Startup now accepts
that exact clean wait state, preserves the historical receipt, rejects any
unexpected partial evaluation outputs, refuses live duplicates, and records
`resumed_wait_owner`. The owner is restarted without allocating GPU7.
## 2026-10-21 — G2 progress evidence refreshed

The protected formal G2 Clean owner remains live on physical GPUs 3/5. The
latest observed training event is step 12,567/18,227 with finite loss
0.3145263; `adapter-step012500.pt` and `latest_training_state.pt` load with
the registered clean manifest SHA. No terminal receipt or final adapter exists
yet, so no downstream bundle, held-out metric, G1 update, or joint optimizer
update is claimed. The waiting owners remain untouched and recoverable.

## 2026-10-21 — G2 progress evidence refreshed (step 12578)

An authoritative event read at 05:04:48 +0800 records the protected formal G2
Clean run at step 12,578/18,227 with finite loss 0.24055790062993765. The two
distributed ranks are still the only active optimizer in this route on physical
GPUs 3/5. The latest durable checkpoint remains `adapter-step012500.pt` with
`latest_training_state.pt`; `final_receipt.json` and a terminal adapter are
still absent. No G1, bundle export, held-out metric, YesTrack update, or native
three-task update is claimed before that terminal receipt. This is evidence of
continued healthy progress only; the prior historical result boundaries remain.

## 2026-10-21 — GroundJEV coupled regression refreshed

The bounded coupled-core regression was rerun after the queue-owner changes: 13
unittests covering joint identity corpus, Phase A core, memory-conditioned input,
fixed-candidate contract, and causal-memory prefill all passed. The registered
G1/G2-bundle/TAO/native-joint/YesTrack queue entrypoints also passed `py_compile`,
and the coupled plus TrackJEV packages passed `compileall`. This verifies the
recoverable launch surface only; it does not promote any model or imply a
tracking metric while G2 lacks its terminal receipt.

## 2026-10-21 — waiting-owner source resolution audit

A read-only `/proc` audit of the seven registered waiting owners confirmed that
all commands run with cwd `/data2/user/LocateMOT`: the G1 and G2-bundle owners
resolve `tools/groundjev/...`, while the TAO-shard, held-out, YesTrack, native
join, and native three-task owners resolve `scripts/groundjev/...` or the
registered join entrypoint from this project root. The sibling runtime checkout
was not used to launch these owners. This closes a reproducibility ambiguity
without changing any owner, checkpoint, or GPU allocation.


## 2026-10-21 — G2 event-integrity audit (step 12594)

The formal G2 event stream contains 12,594 optimizer events from step 1 through
step 12,594 with no step gaps. Loss, gradient norm, memory, learning rates, and
all other recorded numeric fields are finite across the stream. The recent
throughput is stable at 309.8 steps/hour over 100 steps, 311.9 over 500 steps,
and 310.9 over 1,000 steps. Checkpoint events are present at each 500-step
boundary through `adapter-step012500.pt`; the terminal receipt remains absent.
This supports healthy continuation and an estimated 5,633 remaining updates at
roughly 311 steps/hour, but it is not a model-quality or generalization claim.

## 2026-10-21 — tightened G1 post-G2 terminal gate

The resumable GPU5 G1 owner previously accepted a G2 receipt with steps at or
above the registered budget and did not explicitly require the terminal adapter
file. Its post-G2 gate now requires status
`G2_EXACT_COVERAGE_TRAINING_COMPLETE`, exactly 18,227 steps, the registered clean
manifest SHA, and `adapter_projector_final.pt`. A temporary-receipt regression
confirmed that missing adapter and wrong-step receipts are rejected while the
complete contract is accepted. The live waiting owner was not restarted or
modified; this patch protects future recovery and launch paths.


## 2026-10-21 — G2 progress snapshot (step 12610)

The protected G2 Clean run reached step 12,610/18,227 at 05:10:55 +0800 with
finite loss 0.2049703631. The three expected training processes remain live and
`final_receipt.json` is still absent. This snapshot changes only progress
evidence; no downstream model, held-out metric, or generalization claim is
made.

## 2026-10-21 — coupled runtime lifecycle execution repair

Phase-A review found that the single-backbone runtime decoded lifecycle logits
but ignored them: every assigned track was forced to `ACTIVE`, every absent
track only accumulated `LOST`, and an assignment-conditioned memory write could
reactivate a track whose lifecycle decision was `LOST`.  The smallest aligned
fix now executes `ACTIVE/REACTIVATE/LOST` decisions, treats `TERMINATE` on a
matched observation as recoverable `ACTIVE`, blocks memory writes for
`LOST/TERMINATE`, and applies a registered timeout before irreversible
termination of an unmatched track.  A controlled causal replay verified one
birth, a matched `LOST` decision with a held memory write, and timeout
termination; the one-feature-forward and memory-causality regressions still
pass.  This is a runtime correctness repair, not a tracking-performance claim.

## 2026-10-21 — GPU gate for the next real pilot

The single resource audit found all eight A800 GPUs occupied.  Physical GPUs
3/5 remain the protected G2 ranks, GPU7 retains an unrelated compute owner, and
the remaining cards also have active compute workloads; no card has an
exclusive launch window for the Phase-B real-G2 pilot.  The latest durable G2
adapter remains `checkpoints/adapter-step012500.pt` with
`latest_training_state.pt`; the terminal receipt is still not present.  The
pilot was therefore not started on a shared card, and no process was killed or
reused.  This preserves the registered training contract while the existing
G2 completion waiter remains the only long-running observer.

## 2026-10-21 — corrected G0/G2 evidence boundary for early coupled pilots

An audit of the historical Phase-A smoke and Phase-B readout pilot found that
their `checkpoint` was the unadapted `GroundAnything-DLM` bundle and neither
receipt recorded a G2 adapter.  The metrics and loss values are preserved as
G0 native-visual controls only; they are not G2 Clean evidence and are not used
for a G2 performance claim.  The future Phase-A and Phase-B entrypoints now
require both an explicit adapter and the native clean-DLM loader, and record
the adapter/runtime provenance.  This fixes a provenance error without
overwriting the historical control artifacts.

## 2026-10-21 — protected G2 progress snapshot (step 12701)

The authoritative event stream now contains contiguous rows through step
12,701/18,227; the latest row at 05:28:19 +0800 has finite loss
0.2224419505.  The durable checkpoint set still ends at
`checkpoints/adapter-step012500.pt` with
`checkpoints/latest_training_state.pt`, while `final_receipt.json` is absent
and `trainer_contract.json` remains `G2_FORMAL_TRAINING_RUNNING`.  The launcher
is still live.  This is only a healthy-progress record; it does not promote a
checkpoint or claim a tracking metric.

## 2026-10-21 — native three-task training step-contract audit

The registered `queue_real_three_task_native.py` writes `epochs=1` and
`max_steps=0` into its generated config.  The pinned `train/sft.py` contract
interprets a falsy `max_steps` as the full planned epoch
(`steps_per_epoch * epochs`), rather than a zero-step plan.  The queue then
requires the terminal `done` record to report `examples == native manifest
rows` and a non-missing terminal record before declaring completion.  This
keeps the real MOT/RMOT/native-OVMOT run from being mistaken for a CPU-only or
empty optimizer launch.

## 2026-10-21 — G1 GPU5 plan provenance repair

The G1 queue is explicitly bound to physical GPU5, but the older
`native_ground_tracking200_20261014.json` retained a historical
`physical_training_GPU: 7` field.  That plan is preserved unchanged; the G1
queue now references a versioned GPU5 plan with matching physical-device
metadata and an explicit queue-owner provenance note.  The corrected plan will
be read only when its exact G2 terminal gate opens.

The already-running wait-only owner had loaded the old plan path in memory, so
it was safely replaced before any GPU work began (old PID 516321 exited; new
wait owner PID 579251 is `WAITING_G2_CLEAN_SUCCESS`).  Its output directory,
bundle, controls and historical state were preserved; the new owner records
`native_ground_tracking200_gpu5_20261021.json` and still owns no GPU.

## 2026-10-21 — protected G2 throughput snapshot (step 12731)

The formal G2 event stream is contiguous through step 12,731/18,227 with no
non-finite loss, gradient, memory or learning-rate fields.  The measured rates
are 313.73, 312.31 and 311.00 steps/hour over the last 100, 500 and 1,000
steps, respectively.  This leaves 5,496 updates and a linear estimate of
about 17.5–17.7 hours.  `final_receipt.json` and the terminal adapter remain
absent, so this is a throughput/progress record only and does not open any
downstream performance claim.

## 2026-10-21 — G2 durable checkpoint contract audit

The latest durable G2 recovery pair remains intact: `adapter-step012500.pt`
and `latest_training_state.pt` are present at 469,243,990 and 469,247,597
bytes.  The loaded state reports step 12,500, format
`groundanything-g2-projector-qwen3-lora-v2-strict`, the registered manifest
SHA, world size 2, optimizer and scheduler state, both rank RNG paths, sampler
state, and the full training contract.  This validates recoverability only;
the formal run is still not terminal and no downstream model claim is made.

## 2026-10-21 — historical G2 midterm scope audit

Indexed the Clean V2 TRAIN manifest (583,251 rows; SHA-256 `f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71`) against all 24 historical step-3000 G2-PL input frames. All 4 MOT17-02-FRCNN frames are TRAIN-scope; the 8-frame and 12-frame Refer-KITTI-0007 RMOT clips have zero exact image-path overlap with the Clean fit index. Every frame path and content SHA-256 is recorded. The short clips remain diagnostics and do not permit an independent generalization claim. Evidence: `docs/GROUNDJEV/G2_MIDTERM_SCOPE_AUDIT_20261021.{json,md}`.

## 2026-10-21 — J1 terminal checkpoint correction

The historical `trackjev_3task_J1_v4_birth` course is terminal at 1,200/1,200 updates, not active: 19,143 rows were consumed, the final adapter and full FP32/Adam/scheduler/RNG state are present, and no J1 optimizer is live. The earlier budget note had only observed step 650 and is corrected without restarting the run. Evidence: `docs/GROUNDJEV/J1_TERMINAL_RECEIPT_20261021.json` and the updated `J1_J2_TRAINING_BUDGET_AUDIT.md`.

## 2026-10-21 — OVMOT evaluator repair reproduced on fresh copies

The official TETA evaluator was rerun on fresh copies of the existing video-63 control and video-4 native diagnostic. The current configuration keeps `OUTPUT_SUMMARY=True`, `OUTPUT_TEM_RAW_DATA=True`, and `OUTPUT_PER_SEQ_RES=True`, which fixes the two previously recorded evaluator failures without changing GT or predictions. Video 63 completed its independent 40-frame TAO validation scope with TETA 18.640, LocA 7.0114, AssocA 28.618, and 35,228 predictions; it is a simple-backend control, not G2/G1/OneJev adaptation evidence. Video 4 remains only a four-frame native diagnostic: all four frames were safely rejected for duplicate/unsorted canonical coordinate tuples and produced zero predictions. The full 40-frame video-4 native evaluation is not claimed until that parser/output blocker is repaired. Receipt: `docs/GROUNDJEV/OVMOT_EVALUATOR_REPAIR_20261021.{json,md}`.

## 2026-10-21 — conservative OVMOT native-output parser repair

The strict native parser rejects video-4 because otherwise complete GroundAnything responses contain duplicate/unsorted output plus malformed fragments (dangling box tokens and nested object references). Added `locatemot/groundjev/native_output_repair.py` and the `official_scoring_repaired` adapter policy. The repair keeps only complete object-ref/box pairs with valid four-coordinate geometry, drops and records malformed fragments, and never fills or reorders coordinates. A targeted self-test passed; offline preflight of the four preserved raw responses recovered 104/129/133/98 complete pairs. A fresh live retry timed out at the GroundAnything service before frame 751, so no new native metric is claimed and video-4 remains pending service recovery. Evidence is in `docs/GROUNDJEV/OVMOT_EVALUATOR_REPAIR_20261021.{json,md}`.

## 2026-10-21 — G2 Clean progress snapshot at step 12873

The formal G2 Clean event stream advanced to 12873/18,227 during the audit; the latest event at 2026-10-21 06:01:24 +0800 has finite loss 0.272433674, gradient norm 11.859652, and peak memory 12.828881 GiB. The last 100/500/1,000-update windows measure 310.88/312.23/311.53 steps/hour, giving a linear ETA near 2026-10-21T23:12:34.824000+08:00. The durable adapter/state pair remains at adapter-step012500.pt plus `latest_training_state.pt`; no terminal receipt exists and all downstream queues remain correctly gated. Evidence: `docs/GROUNDJEV/G2_PROGRESS_SNAPSHOT_20261021_STEP12873.{json,md}`.

## 2026-10-21 — native joint queue config contract repair

The post-G2 native three-task owner was found to inject the audit-only
`native_manifest_rows` field into its generated OneJev config. The pinned
`train.sft` parser rejects unknown keys, so this would have failed before the
first model load when all dependency gates opened. Removed that field from the
generated config; the native manifest row count remains in the owner state and
is checked against the terminal `done.examples` receipt. Validated the
normalized config against `train.sft.build_parser`, confirmed `max_rows=1` is
only a per-microbatch cap with `accum_rows=16`, and ran compile/import smokes.
No G2 state or process was changed. Receipt:
`docs/GROUNDJEV_COUPLED/REAL_NATIVE_CONFIG_AUDIT_20261021.{json,md}`.

## 2026-10-21 — corrected coupled architecture boundary

The architecture/source audit had stale wording that described the bridge as
visual-only. The current code includes `NativeG2CleanDLMProvider`, and the
Phase-C receipt records one real G2 clean visual-plus-language forward with
same-frame context consumed by the typed readout. Updated the audit to state
that native context is complete while SGLang integration and trained
pre-grounding memory feedback remain open gates. No metric or model claim was
changed.

## 2026-10-21 — opened C2 prefill training gradient path

The causal memory prefill adapter previously wrapped every call in
`torch.inference_mode`, which was correct for deployment but would silently
block gradients for a memory-conditioned training run. Added an explicit
`inference_mode` switch that preserves default inference behavior and allows
training callers to pass `False`. A new backward test confirms the inserted
memory token receives finite gradient through the prefill and readout. This
is an interface correction, not a tracking or grounding gain.

## 2026-10-21 — G2 Clean progress snapshot at step 12907

The protected G2 event stream is contiguous through 12,907/18,227. The latest
event at 06:07:55 +0800 has finite loss 0.1906903072, gradient norm 9.2506351,
and peak memory 12.8288 GiB. Last-100/500/1,000-step throughput is
312.084/312.309/311.484 steps per hour, leaving 5,320 updates and a linear
ETA around 23:10–23:13 Asia/Shanghai. The durable recovery pair remains
`checkpoints/adapter-step012500.pt` and `checkpoints/latest_training_state.pt`;
there is no final receipt. All downstream owners remain gated. Evidence:
`docs/GROUNDJEV/G2_PROGRESS_SNAPSHOT_20261021_STEP12907.{json,md}`.

## 2026-10-21 — repaired G1 GPU5 waiter runtime paths

The G1 waiter was correctly idle but its future command resolved the model,
GroundAnything source and ground-train Python environment under the code
checkout, where they do not exist. Updated the queue to run training, export
and controls from `/data2/user/LocateMOT_GroundJEV` while preserving the code
checkout for the registered plan and data/protocol files. Verified the runtime
paths, terminated only the old wait-only owner PID 579251, and started PID
631075 with status `WAITING_G2_CLEAN_SUCCESS`, zero optimizer updates and no
GPU launch. G2 and historical artifacts were untouched. Evidence:
`docs/GROUNDJEV/G1_GPU5_RUNTIME_PATH_REPAIR_20261021.{json,md}`.

## 2026-10-21 — repaired G2 bundle and TAO exporter runtime paths

The G2 deployment-bundle owner and the native TAO shard owner were waiting on
the formal G2 receipt, but both future export calls resolved the base model and
exporter under the source checkout. That checkout intentionally has no
`models/GroundAnything-DLM`; the dedicated GroundJEV runtime does. Both queues
now invoke the runtime GroundAnything Python and runtime exporter from
`/data2/user/LocateMOT_GroundJEV`, while the source checkout remains the code
and manifest authority. The two wait-only owners were replaced without
touching G2: bundle PID 634486 and TAO PID 634487 are both still gated at
`WAITING_G2_CLEAN_SUCCESS`, with no optimizer updates or GPU allocation.
`py_compile` and a runtime Torch import passed. Evidence:
`docs/GROUNDJEV/G2_BUNDLE_RUNTIME_PATH_REPAIR_20261021.{json,md}`.

## 2026-10-21 — G2 Clean progress snapshot at step 12938

The protected event stream advanced to 12938/18227. The latest finite event at
06:13:50 +0800 has loss 0.1918672598, gradient norm 10.0799599, and peak
allocated memory 12.8289 GiB. Recent 100/500/1000-step rates are
313.316/312.283/311.850 steps per hour, giving a linear ETA around
23:06–23:11 Asia/Shanghai. The final receipt is absent; adapter-step012500 and
latest_training_state.pt remain the durable recovery pair. This is a training
health snapshot only, not a tracking result. Evidence:
`docs/GROUNDJEV/G2_PROGRESS_SNAPSHOT_20261021_STEP12938.{json,md}`.

## 2026-10-21 — corrected G2 native evaluation identity and runtime paths

The short-control entrypoint still hard-coded its adapted arm as `G2-PL` and
resolved future inference/scoring scripts from the historical OneJev checkout.
It now derives the Clean arm as `G2-CLEAN-V2`, validates the terminal adapter
step 18227, Clean TRAIN manifest SHA, merged model SHA and export receipt before
launch, and executes from `/data2/user/LocateMOT_GroundJEV`. The historical
`GroundAnything-DLM-G2-PL` path deliberately keeps the `G2-PL` label and is not
reclassified. The code was compiled and the historical identity path was
smoke-tested; no service or GPU was started. Evidence:
`docs/GROUNDJEV/G2_NATIVE_EVAL_IDENTITY_REPAIR_20261021.{json,md}`.

## 2026-10-21 — relaunched exited G1 GPU5 wait owner

The G1 dependency owner had exited with only its initial
`WAITING_G2_CLEAN_SUCCESS` state; no training output, optimizer update or GPU5
launch existed. After verifying that the registered output directory still
contained only the owner state, relaunched it with the same G2 root, runtime
GroundAnything base, output, bundle and control paths. New PID 642505 is live
and waiting; protected G2 was untouched. Evidence:
`docs/GROUNDJEV/G1_GPU5_WAIT_OWNER_RELAUNCH_20261021.{json,md}`.

## 2026-10-21 — G2 Clean live health snapshot at step 12964

The protected two-rank G2 process remains running on physical GPUs 3/5; both
ranks are in the compute loop with about 19.3 GiB allocated each. The event
stream is contiguous through 12964/18227, with finite loss 0.3086177218,
gradient norm 13.7076988 and peak memory 12.8289 GiB. Recent throughput is
312.500/312.067/311.850 steps per hour over 100/500/1000 updates, giving a
linear ETA around 23:09–23:11. No terminal receipt exists and no metric claim
is made. Evidence:
`docs/GROUNDJEV/G2_PROGRESS_SNAPSHOT_20261021_STEP12964.{json,md}`.

## 2026-10-21 — downstream pre-launch contract preflight

While G2 remained non-terminal at step 12970, imported every registered G1,
G2 bundle, held-out, TAO, native join, three-task and YesTrack queue module;
all targeted Python compilation checks passed. The dedicated runtime Python
environments, GroundAnything base, service interpreter, exporter, video runner
and Clean TRAIN manifest were present. No GPU, optimizer or model service was
started. Evidence:
`docs/GROUNDJEV/DOWNSTREAM_PRELAUNCH_PREFLIGHT_20261021.{json,md}`.

## 2026-10-21 — G2 Clean live progress snapshot at step 12980

The protected two-rank G2 Clean process remains alive and its event stream is
finite and contiguous through 12980/18227. The latest event at
2026-10-21 06:21:56 +0800 has loss 0.2029841521, gradient norm
11.7594414, and peak event memory 12.8288527 GiB.
The measured 100/500/1000-step rates are
312.5/312.012/311.877
steps per hour, giving a linear estimate of about 16.817 hours from the
500-step window. The final receipt is still absent; adapter-step012500 and
latest_training_state.pt remain the durable recovery pair. Downstream owners are
still dependency-gated and have no optimizer updates. This is a training-health
snapshot only and makes no tracking or benchmark claim. Evidence:
`docs/GROUNDJEV/G2_PROGRESS_LIVE_20261021_STEP12980.{json,md}`.

## 2026-10-21 — YesTrack GPU7 resource gate confirmed

The live YesTrack owner remains `WAITING_GPU7_EXCLUSIVE` because physical GPU7
has an unrelated process (PID 4008192, user `ljh`, PRISM/CUB training) using about
16.2 GiB at 100% utilization. The queue's rule is no compute PIDs and at least
32768 MiB free; the external process was left untouched. This is an observed
resource gate, not a training or metric result. Evidence:
`docs/GROUNDJEV/YESTRACK_GPU7_RESOURCE_GATE_20261021.{json,md}`.

## 2026-10-21 — MOT-only long-training safe-close audit

A process-name audit over all `lyq` Python/torch/train/eval/service processes
found no residual MOT-only or old OneJev long optimizer. The only project
optimizer is the protected G2 Clean two-rank run at step 12991/18227
on physical GPUs 3/5. Dependency queues and the GroundAnything service remain
registered; no legacy MOT-only process was restarted. Evidence:
`docs/GROUNDJEV/MOT_ONLY_SAFE_CLOSE_AUDIT_20261021.{json,md}`.

## 2026-10-21 — G0 MOT20 long export terminal receipt audited

The frozen G0 long export has an authoritative official-scope receipt for
MOT20-05: 1658 legal TRAIN frames, HOTA 0.433441,
DetA 0.638348, AssA 0.294655, IDF1
0.441085, IDSW 6239. The receipt is a
G0 TRAIN/validation-scope reference only; it is not MOT20 TEST and is not an
independent G2 held-out claim. The older 1,657-frame diagnostic artifact is
kept separate. Evidence:
`docs/GROUNDJEV/G0_MOT20_LONG_EXPORT_AUDIT_20261021.{json,md}`.

## 2026-10-21 — live completion matrix refreshed at G2 step 12997

The previous matrix was stale at step12002. A new execution-state matrix records
G0 MOT20-05 long export complete in its legal TRAIN scope, MOT-only safely closed,
G2 running at 12997/18227 on GPUs3/5, and all downstream G1/TAO/held-out/
three-task/YesTrack gates with their current statuses. Scientific gain remains
unproven and independent metrics remain pending. Evidence:
`docs/GROUNDJEV/GOAL_COMPLETION_MATRIX_LIVE_20261021_STEP12997.{json,md}`.

## 2026-10-21 — G2 recovery checkpoint validated on CPU

While the protected G2 run continued, the latest durable training checkpoint
was loaded with the CPU-only runtime. It is step 13000 at world size
2, and contains model trainable state, optimizer, scheduler,
sampler position, Python/NumPy/Torch/CUDA RNG state, the training contract and
trainable audit. The adapter-step012500 file is also present. No GPU was touched
and no metric claim is made. Evidence:
`docs/GROUNDJEV/G2_CHECKPOINT_RECOVERY_AUDIT_20261021.{json,md}`.

## 2026-10-21 — G2 Clean live progress at step 13065

The protected two-rank G2 Clean run advanced to step 13065/18227 while both ranks remained live on GPUs 3/5. The event stream remains finite with finite loss and gradient norm; no final receipt exists yet, so downstream G1, native TAO/OVMOT joining, three-task OneJev optimization and independent held-out evaluation remain gated. This receipt records training health only and makes no scientific metric claim. Evidence: `docs/GROUNDJEV/G2_PROGRESS_LIVE_20261021_STEP13065.{json,md}`.

## 2026-10-21 — G2 throughput estimate at step 13072

Using actual event timestamps from the protected two-rank G2 Clean run, the recent 100/500/1000-event windows give approximately 5.20 steps/min and 16.52 hours to step 18227. This is an operational ETA only; it is not a metric or quality claim. Evidence: `docs/GROUNDJEV/G2_THROUGHPUT_ESTIMATE_20261021_STEP13072.{json,md}`.

## 2026-10-21 — G2 adapter checkpoint step 13000 audited at event step 13076

The protected G2 Clean run advanced to event step 13076/18227 and produced the durable `adapter-step013000.pt` checkpoint. Its SHA-256 is `08cf28930eced61b2086785494728221e113ac75cd7dc6aefe02ab1bc2ebfe59`; the latest full recovery state remains step 13000 with SHA-256 `4a79cbf3e64e69ffc56aaf964754d0209382b87c2124d46f13506325ca52fd22`. The final receipt is absent, so downstream gates remain closed. This is checkpoint and training-health evidence only. Evidence: `docs/GROUNDJEV/G2_CHECKPOINT_PROGRESS_20261021_STEP13076.{json,md}`.

## 2026-10-21 — live goal-completion matrix at G2 step 13116

The execution matrix was refreshed from the current G2 event stream and downstream owner states. G0 MOT20 long export remains complete in legal TRAIN scope and MOT-only training remains safely closed; G2 is still running at step 13116/18227, while G1, native TAO/OVMOT joining, three-task OneJev, YesTrack and held-out evaluation remain gated. This is an execution audit and makes no scientific metric claim. Evidence: `docs/GROUNDJEV/GOAL_COMPLETION_MATRIX_LIVE_20261021_STEP13116.{json,md}`.

## 2026-10-21 — G2 adapter checkpoint step 15500 audited at event step 15737

The protected two-rank G2 Clean run advanced to event step 15737/18227 on physical GPUs 3 and 5 and produced the durable adapter-step015500 checkpoint. The adapter and latest full recovery state both loaded on CPU with `weights_only=False`; optimizer, scheduler, sampler and RNG state are present. The terminal receipt is still absent, so G2-dependent bundle export, G1, native TAO/OVMOT joining, three-task OneJev optimization and held-out/YesTrack evaluation remain gated. This is checkpoint and training-health evidence only; it does not establish a coupled JEV metric improvement. Evidence: `docs/GROUNDJEV/G2_CHECKPOINT_PROGRESS_20261021_STEP15500.{json,md}`.

## 2026-10-21 — G2 adapter checkpoint step 16000 audited at event step 16014

The protected two-rank G2 Clean run advanced to event step 16014/18227 on physical GPUs 3 and 5 and produced the durable adapter-step016000 checkpoint. The adapter and latest full recovery state both loaded on CPU with `weights_only=False`; optimizer, scheduler, sampler and RNG state are present and the registered Clean manifest SHA matches. The terminal receipt is still absent, so G2-dependent bundle export, G1, native TAO/OVMOT joining, three-task OneJev optimization and held-out/YesTrack evaluation remain gated. This is checkpoint and training-health evidence only; it does not establish a coupled JEV metric improvement. Evidence: `docs/GROUNDJEV/G2_CHECKPOINT_PROGRESS_20261021_STEP16000.{json,md}`.

## 2026-10-21 — G2 adapter checkpoint step 16500 audited at event step 16501

The protected two-rank G2 Clean run advanced to event step 16501/18227 on physical GPUs 3 and 5 and produced the durable adapter-step016500 checkpoint. The adapter and latest full recovery state both loaded on CPU with `weights_only=False`; optimizer, scheduler, sampler and RNG state are present and the registered Clean manifest SHA matches. The terminal receipt is still absent, so G2-dependent bundle export, G1, native TAO/OVMOT joining, three-task OneJev optimization and held-out/YesTrack evaluation remain gated. This is checkpoint and training-health evidence only; it does not establish a coupled JEV metric improvement. Evidence: `docs/GROUNDJEV/G2_CHECKPOINT_PROGRESS_20261021_STEP16500.{json,md}`.
## 2026-10-21 — 坐标加权耦合 step60 仍未恢复真实框

在冻结 G2 step17000 的前提下，从双序列耦合 step48 恢复到 step60，使用坐标 token 权重 5.0，MOT20-03/05 共 6 个 transition、10 次视觉编码，正式 G2 未改动。Grounding teacher-forcing loss 末步为 2.09848，JEV structured loss 为 0.000779；两个序列的 MATCH/ADMIT/Yes-No/ACTIVE 仍为 1.0，但 C2 greedy 框平均 IoU 仍为 0。对 step60 做无更新 token trace，C2 坐标均值损失降到 2.5554/2.5421，却仍无法贪心生成首个坐标 `<214>`/`<339>`（分别输出 `<1>`/`<|box_start|>`）。结论：耦合链路真实训练有效影响 token 概率，但输出坐标接口/课程仍未打通，停止盲目加步，下一步改坐标输出目标或显式候选框条件并重新做独立视频验证；不宣称 MOT/RMOT/OVMOT 提升。证据：`docs/GROUNDJEV_COUPLED/PHASE_C_REAL_G2_COUPLED_COORDINATE_WEIGHT_STEP17000_20261021.{json,md}`。
## 2026-10-21 — native visual-prompt memory also leaves G2 geometry unchanged

用 GroundAnything 官方 `build_visual_prompt` 把上一帧合法框作为当前帧 visual reference，MOT20-03/05 各 5 帧共 10 次视觉编码，G2 step17000 冻结、无参数更新、无第二个 VLM。所有 C2 输出都能被官方 parser 恢复为合法框，但两序列 C2 平均 IoU 均为 0；首个坐标仍接近冻结 G2 错误基线。结论：原生 reference-box 接口本身不能修复当前坐标头，不能把该诊断称为跟踪收益。证据：`docs/GROUNDJEV_COUPLED/PHASE_C_VISUAL_PROMPT_MEMORY_STEP17000_20261021.{json,md}`。

## 2026-10-21 — identity + previous-box state projector 双序列耦合失败边界

新增小型 `IdentityBoxStateTokenProjector`，将上一帧共享视觉摘要和持久框坐标一起插入 G2 causal prompt；冻结 G2 step17000，MOT20-03/05 共 6 个 transition，真实 GPU 更新 12 步。Grounding loss 降至 1.70039，typed JEV MATCH/ADMIT/Yes-No/ACTIVE 在两序列均为 1.0，但 C2 greedy 框 IoU 仍为 0，首坐标仍输出 `<1>`/`<0>`。结论：只加入数值框状态仍不足以控制 G2 坐标头，下一步需要合法的 coordinate-output adaptation/候选条件训练，而不是盲目增加 JEV 步数。证据：`docs/GROUNDJEV_COUPLED/PHASE_C_STATE_CONDITIONED_MEMORY_STEP17000_20261021.{json,md}`。
## 2026-10-21 — corrected dense G2 contract exposes one-box baseline

审计发现此前 target-only 耦合 pilot 将 `Locate all persons` 与单目标 answer 混用，不能据此解释 G2 坐标头。新增多框 causal decoder：每个框的四个坐标受官方 token 范围约束，只在 `<|im_end|>` 停止，并在 MOT20-03/05 各 3 个真实帧保留全部 GT person boxes。冻结 G2 step17000、6 次视觉编码、无参数更新时，模型仍在每帧只输出一个合法框；IoU0.5 召回均值 MOT20-03=`0.010345`、MOT20-05=`0.004038`。结论：这是 G2 dense 前端的真实基线；后续耦合训练必须使用完整多框合同或合法 referring/visual-prompt 合同，之前 target-only IoU 结果降级为协议不匹配的接口诊断。证据：`docs/GROUNDJEV_COUPLED/PHASE_C_REAL_G2_DENSE_MULTIBOX_STEP17000_20261021.{json,md}`。

## 2026-10-21 — G2 Clean live progress at step 17488

The protected two-rank G2 Clean run remains live at `17488/18227` on GPUs 3/5.
The latest event is finite (loss `0.1523213747`, gradient norm `8.3949747`, peak
`12.828824 GiB`); recent 500-step throughput is `307.919 steps/hour`, giving a
linear remainder estimate of about `2.40 h`. The durable recovery state and
`adapter-step017000.pt` are present, while the terminal receipt is absent.
G1, bundle export, native OVMOT joining, three-task optimization and held-out
evaluation remain gated; no independent OneJev/YesTrack optimizer was started.
This is a training-health/recoverability record only. Evidence:
`docs/GROUNDJEV/G2_PROGRESS_LIVE_20261021_STEP17488.{json,md}`.

## 2026-10-21 — GroundJEV fixed-candidate cross-sequence diagnostic on MOT20-05

冻结在 MOT20-03 TRAIN repair split 上训练的 180-step readout，在不同的
MOT20-05 TRAIN 视频帧 766–893（128 帧）上零更新运行；仍使用同一注册的
native candidate/region cache 合同和单主干缓存回放。TrackEval 得到
HOTA=`0.115666`、DetA=`0.147839`、AssA=`0.090593`、IDF1=`0.071458`、
IDSW=`341`、FP=`19`、FN=`21407`，126 帧有输出，关联回放约 `9.06 FPS`。
相比 MOT20-03 同视频 temporal repair 的 HOTA=`0.302815`、IDF1=`0.257729`、
IDSW=`18`，跨视频明显退化，说明当前 fixed-candidate readout 尚未泛化。
两个范围均为 TRAIN，缓存早于 G2 Clean 终点，不能当作 G2 Clean、TEST 或
最终 GroundJEV 增益。证据：
`docs/GROUNDJEV_COUPLED/PHASE_B_FIXED_CANDIDATE_CROSS_SEQUENCE_MOT20_05_20261021.{json,md}`。

## 2026-10-21 — registered GroundJEV coupled video entrypoint

Added `tools/groundjev/coupled/run_coupled_video.py` as the clean-checkout
entrypoint for the already-tested fixed-candidate GroundJEV video evaluator.
It forwards the real clip/checkpoint/output/device arguments to the existing
single-backbone runtime, keeps the TrackEval scope explicit, and does not start
a second VLM or relabel cached features as terminal G2 Clean output. The new
entrypoint passes `py_compile` and `--help`; no GPU job was launched by this
change.

## 2026-10-21 — G2 Clean step 17500 checkpoint recovery audit

在受保护的两卡 G2 训练继续运行期间，用注册 CPU 环境只读加载
`adapter-step017500.pt` 和 `latest_training_state.pt`。两者均报告 step
17500、world size 2、严格 G2 格式和一致的 Clean manifest SHA；adapter 与
完整 state 各有 293 个 trainable tensors，optimizer/scheduler/sampler/RNG
状态均存在，adapter 浮点参数全部有限。终态 receipt 仍未生成，训练进程
未修改。证据：`docs/GROUNDJEV/G2_CHECKPOINT_PROGRESS_20261021_STEP17500.{json,md}`。

## 2026-10-21 — post-G2 downstream queue contract audit

在正式 G2 仍运行期间，对终态后自动执行的 G2 bundle、GPU5 G1、native
OVMOT join、native MOT/RMOT/OVMOT 三任务和独立 held-out 五个入口执行只读
合同审计。五个脚本均通过注册 Python 环境的 `py_compile`；终态 G2
status/step/manifest 校验、bundle receipt、native OVMOT manifest gate、
held-out scope gate 和资源等待条件均存在，未启动任何 GPU 或 optimizer。
这是终态前置审计，不代表下游已完成。 

## 2026-10-21 — step17500 coupled decoder regression and DanceTrack dimension repair

G2 step17500 保持冻结，在两个真实 DanceTrack 序列各训练一个 transition、共 2
个 optimizer steps；官方 `hierarchy_dynamic` 解码在 4 个 transition 上均返回
合法框。C0/C2 recall@IoU0.5 分别为 1/7→2/7（0001）和 1/6→2/6（0016），
只作为受限耦合开发证据。回归发现官方 evaluator 把 DanceTrack-0016 写成
1280×720，实际 `seqinfo.ini` 是 1920×1080；已修正并用同一 checkpoint 重跑，
没有触碰正式 G2。未报告 HOTA/IDF1/TETA 或独立泛化提升。证据：
`docs/GROUNDJEV_COUPLED/PHASE_C_REAL_G2_DENSE_COUPLING_STEP17500_20261021.{json,md}`。

## 2026-10-21 — formal G2 remains live; no downstream restart

单次权威状态检查确认正式 G2 launcher PID `1799359` 仍存在；终态 receipt 与
`adapter_projector_final.pt` 尚未生成，最新可用周期 adapter 仍为
`adapter-step017500.pt`，完整 recovery state 保留。G1 GPU5、G2 bundle、GPU7
独立 held-out、TAO native shards 和 native 三任务 owner 分别仍处于注册的等待
状态，三任务 owner optimizer updates 为 0。没有重启、覆盖或提前晋升非终态
checkpoint。证据：`docs/GROUNDJEV/G2_LIVE_WAIT_20261021.{json,md}`。

## 2026-10-21 — formal G2 event step 17593 live progress

重新读取同一个正式 event stream，确认 G2 已从 step17500 继续推进到
`17593/18227`；最后事件 loss=`0.171383`、gradient norm=`10.1324`、peak
memory=`12.829 GiB`，最近 500 步吞吐约 `307.1 steps/hour`，线性剩余约
`2.1 h`。launcher PID `1799359` 仍存活，终态 receipt/final adapter 仍缺失，
所以没有启动或改写任何下游 owner。证据：
`docs/GROUNDJEV/G2_PROGRESS_LIVE_20261021_STEP17593.{json,md}`。

## 2026-10-21 — downstream coupling regression remains green during G2 wait

在正式 G2 继续占用 GPU3/5、终态 receipt 尚未生成期间，对
`locatemot/groundjev/coupled`、`tools/groundjev/coupled` 和
`scripts/groundjev` 执行 registered Python 环境的 `compileall`；随后运行
11 个核心 coupling 单元测试（official DLM memory、causal prefill、memory
conditioned input、Phase-A core），结果全部通过。没有启动 GPU、修改 G2 或
改变下游等待条件；该验证只证明代码入口可重建，不构成性能结果。

## 2026-10-21 — G2 latest recovery state CPU integrity recheck

在正式 G2 仍运行到 event step17615、终态 receipt 尚缺失期间，用 registered
ground-train Python 环境只读加载 `checkpoints/latest_training_state.pt`。
state 报告 durable step17500、world size2、293 个 trainable tensors，
optimizer/scheduler、sampler position、四类 RNG 和训练合同字段均存在；所有
浮点 trainable tensors 有限，文件 SHA=`9aad3f6bd6b4bfcab386d1ec1e39ea03420fc0a202df0d67a447e158f4e66139`。
未改写 checkpoint、未启动 GPU，确认正式 G2 仍可安全恢复。

## 2026-10-21 — independent held-out image-overlap gate cleared

在 G2 终态前用 CPU-only 直接调用注册的 held-out `overlap_audit()`，对
MOT20-05 validation-half 1,657 帧、RMOT single 800 帧、RMOT multi 800 帧和
TAO OVMOT 40 帧逐项比对 G2 Clean TRAIN rows；四个范围 exact image overlap
均为 0，TRAIN manifest SHA 为
`4a8e39d29a30fe1ef7df10484be57a10ab5f0baff0638130ff93fa9c1181ab6a`。没有模型
推理或 GPU 分配；该结果只清除独立性门槛，不代表任何性能指标。证据：
`docs/GROUNDJEV/HELDOUT_OVERLAP_AUDIT_20261021.{json,md}`。

## 2026-10-21 — independent held-out input preflight passed

对注册的四个独立 held-out manifest 做 CPU-only 文件检查：MOT20-05 validation
1,657 帧、RMOT single 800 帧、RMOT multi 800 帧和 TAO OVMOT 40 帧图片路径全部
存在（missing=0），且均声明 `training_on_this_scope=false`。没有 GPU 推理或
指标计算；这只保证终态 G2 bundle 生成后 held-out owner 不会因输入路径缺失
失败。证据：`docs/GROUNDJEV/HELDOUT_INPUT_PREFLIGHT_20261021.{json,md}`。

## 2026-10-21 — G0/G200/G1 comparison matrix aligned with the coupled mainline

按最新 GroundJEV 单主干耦合方向重新核对并登记三组对照：G0 的 MOT20-05
1,658 帧合法 TRAIN 长导出已完成；G200 是物理 GPU5 上完成的 200 步隔离定位
控制，保留 MOT17-02 四帧、RMOT single8/multi12 的短功能 receipt；新 G1
GPU5 owner 仍为 `WAITING_G2_CLEAN_SUCCESS`，optimizer updates 为 0，未启动
训练。耦合实现（官方 dense 多框解码、typed JEV、身份/框状态 memory、固定候选
视频回放）已有受限真实数据证据，但没有把短片或跨序列诊断升级为独立性能提升。
证据：`docs/GROUNDJEV/G0_G200_G1_COMPARISON_MATRIX_20261021.{json,md}`。

## 2026-10-21 — YesTrack route corrected to auxiliary offline reference

按新的 GroundJEV 单主干耦合任务重新校正执行路线：独立 YesTrack/OneJev
不再作为在线主模型，也不应阻塞 G2、JEV 结构化决策或持续身份记忆的耦合训练。
YesTrack 的 Yes/No 数据组织和概率读取只作为共享 GroundAnything 主干上的辅助监督
与离线参考对照；保留既有历史结果，不启动新的独立 YesTrack 长训练。正式主线仍是
GroundAnything hidden states + typed JEV choice/NEW/NO_MATCH + persistent identity
memory + one-to-one assignment。

## 2026-10-21 — independent native OneJev owner paused and coupled replacement registered

审计发现 `real_three_task_joint_native_v1_queue` 虽然尚未启动 GPU，但终态后会调用
独立 `run_source_manifest_sft.py`，与新的单主干 GroundJEV 路线冲突。该 owner 已安全
停在 `PAUSED_FOR_GROUNDJEV_SINGLE_BACKBONE`，原 manifest/config 和 owner 状态保留，
没有任何 optimizer update。新增 `queue_groundjev_native_three_task.py` 与
`train_native_three_task.py`：依赖精确 G2 Clean 终态和 native TAO 三任务 manifest，
在 GPU4 上只加载一个 G2 主干，按 MOT/RMOT/OVMOT 均衡样本提取一次视觉 hidden state，
训练共享 typed JEV、因果 memory projector 和小型 OVMOT category readout；不启动
独立 OneJev/YesTrack。CPU loader、loss/backward synthetic regression、compileall 和
11 个既有 coupling tests 均通过，尚未启动 GPU 训练，待 native manifest 与 G2 终态门
打开后自动恢复。

## 2026-10-21 — coupled three-task owner confirmed at dependency gate

新的 `queue_groundjev_native_three_task.py` 已在独立 owner 会话中运行，当前状态为
`WAITING_G2_NATIVE_MANIFEST`，`optimizer_updates=0`、`gpu_launched=false`。它没有占用
GPU，也没有重复启动旧 OneJev；G2 Clean 终态和 native TAO 三任务 manifest 仍是唯一
启动条件。该等待是可恢复的真实 owner 状态，不是训练结果或指标。

## 2026-10-21 — G2 Clean live progress at step 17728

同一个受保护 G2 Clean 训练句柄仍在推进，event stream 已到
`17728/18227`（`2026-10-21 21:33:28 +0800`），loss=`0.1114838467`、
gradient norm=`10.3513269`。最近 100/500/1000 事件吞吐约为
`286.73/303.04/307.59 steps/hour`，线性剩余约 `1.74/1.647/1.622 h`。
终态 receipt 和终态 adapter 尚未出现，持久恢复对仍为
`adapter-step017500.pt` 与 `checkpoints/latest_training_state.pt`；没有启动或
改写任何下游 owner。这是训练健康/ETA 证据，不是 MOT/RMOT/OVMOT 指标结果。
证据：`docs/GROUNDJEV/G2_PROGRESS_LIVE_20261021_STEP17728.{json,md}`。

## 2026-10-21 — G2 Clean live progress at step 17750

event stream 已到 `17750/18227`（`2026-10-21 21:40:20 +0800`），最新
loss=`0.0942097213`、gradient norm=`7.7979364`，均有限。终态 receipt 和
终态 adapter 尚未生成，持久恢复对仍为 `adapter-step017500.pt` 与
`checkpoints/latest_training_state.pt`；G1、TAO native、held-out 和共享三任务
owner 继续受依赖门保护。这是训练健康证据，不是 MOT/RMOT/OVMOT 指标结果。
证据：`docs/GROUNDJEV/G2_PROGRESS_LIVE_20261021_STEP17750.{json,md}`。

## 2026-10-21 — G0 MOT20-05 长时导出终态复核

重新读取 G0 权威 owner 与完成 receipt：`mot20_05_full_validation_gpu5_v1` 为
`NATURAL_MOT20_SCOPE_AND_FROZEN_MEMORY_CONTROLS_COMPLETE`，fresh source 输出覆盖
1657 帧，官方指标 receipt 完整且归档校验通过（HOTA=0.07334696、DetA=0.09026983、
AssA=0.05967427、IDF1=0.05959845、IDSW=1933、FP=417、FN=288297）。该 G0 结果
保留作为对照，不被新的 GroundJEV 单主干耦合训练覆盖。

## 2026-10-21 — G2 Clean live progress at step 17854

同一受保护 G2 Clean 句柄推进至 `17854/18227`（`2026-10-21 22:04:25 +0800`），
近 500/1000 事件吞吐约 `0.0792/0.0825 steps/s`，线性剩余约 `1.31/1.26 h`。
`final_receipt.json` 与 `adapter_projector_final.pt` 尚未出现；下游 G1、TAO native、
held-out 和共享三任务 owner 仍处于依赖等待，未启动重复训练或独立 YesTrack。

## 2026-10-21 — G2 Clean exact-coverage training terminal

受保护的 G2 Clean exact-coverage 训练已到达终态 `18227/18227`，正式 receipt
为 `G2_EXACT_COVERAGE_TRAINING_COMPLETE`。训练合同为 world size 2、effective
global batch 32、exact-once coverage；manifest SHA256 为
`f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71`。终态
adapter SHA256 为
`cf1b5a6f626810ead94eb4e16adaf3feb6c6da3f99ca5023b6b5c234726f5742`，恢复
state SHA256 为
`cab791def46c264393636c5697f452ec81dd0a242cfd9408644490b92cf643dd`。最后
训练事件 loss=`0.6241224888`、gradient norm=`6.61599`、peak memory
`17.374252 GiB` 均有限；这只是训练终态与可恢复性证据，不宣称任何
MOT/RMOT/OVMOT 性能提升。G2 bundle 已完成，TAO native 导出和 GPU5 G1
启动由持久 owner 自动推进；共享三任务训练仍等待原生 manifest，未手动重复启动。

## 2026-10-21 — TAO native export resource-gate repair

TAO native OVMOT 的八个输入 shard manifest 和 G2 native merged bundle 已经
存在，但原注册 owner 固定等待 GPU4/6；这两张卡持续被外部 PRISM/ShiftRMOT
作业占用，因此没有产生 native candidate。未终止或修改这些外部作业。按当前
显存实测，将 owner 安全停止并从未完成的 shard 边界恢复到 GPU1/3；新的资源
合同要求每张选定卡至少保留 49152 MiB free，允许记录共存进程但不结束它们。
G2 checkpoint 未重训，三任务 owner 仍只等待合并后的 native manifest；这是
资源门修复，不是模型指标结果。
